//! ACP (Agent Client Protocol) bridge for sandboxed execution.
//!
//! Spawns any ACP-compliant agent (Goose, Codex, Gemini CLI, etc.) as a
//! subprocess inside a Docker container and communicates via the standard
//! ACP protocol (JSON-RPC over stdio). Agent output is translated into
//! IronClaw's `JobEventPayload` stream and posted to the orchestrator.
//!
//! Security model: the Docker container is the primary security boundary
//! (cap-drop ALL, non-root user, memory limits, network isolation). By
//! default, ACP `request_permission` calls are auto-approved in the
//! container — the sandbox is the only layer.
//!
//! For an optional second, user-consent layer, set
//! `surface_permissions=true` on a per-agent basis in
//! `~/.ironclaw/acp-agents.json` (or the DB `acp_agents` setting). The
//! host propagates this to the container as
//! `IRONCLAW_ACP_SURFACE_PERMISSIONS=true`. When enabled, the bridge
//! defers each `request_permission` to the orchestrator over HTTP, which
//! inserts a `PendingGate` (gate_name = `acp_permission`) and broadcasts
//! `AppEvent::GateRequired`. The user resolves via the existing
//! `/api/chat/gate/resolve` web endpoint; the resolve handler
//! short-circuits `acp_permission` gates and forwards a `PermissionDecision`
//! back to the bridge, which returns it as the ACP
//! `RequestPermissionOutcome`. On timeout (10 minutes), the bridge
//! returns `Cancelled` — safe default on user inactivity.
//!
//! ```text
//! ┌──────────────────────────────────────────────┐
//! │ Docker Container                              │
//! │                                               │
//! │  ironclaw acp-bridge --job-id <uuid>          │
//! │    └─ spawns ACP agent subprocess             │
//! │    └─ ACP handshake (initialize + session)    │
//! │    └─ sends job description via prompt()      │
//! │    └─ translates ACP events → JobEventPayload │
//! │    └─ POSTs events to orchestrator            │
//! │    └─ polls for follow-up prompts             │
//! └──────────────────────────────────────────────┘
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use agent_client_protocol::{self as acp, Agent as _};
use serde_json::json;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};
use uuid::Uuid;

use crate::error::WorkerError;
use crate::worker::api::{
    CompletionReport, JobEventPayload, PermissionDecision, PermissionOptionDto,
    PermissionOptionKindDto, PermissionRequestRegister, WorkerHttpClient,
};

/// Configuration for the ACP bridge runtime.
pub struct AcpBridgeConfig {
    pub job_id: Uuid,
    pub orchestrator_url: String,
    pub timeout: Duration,
    /// Command to spawn the ACP agent.
    pub agent_command: String,
    /// Arguments for the agent command.
    pub agent_args: Vec<String>,
    /// Extra environment variables for the agent process.
    pub agent_env: HashMap<String, String>,
    /// Whether to surface `request_permission` calls from the ACP agent to
    /// the end user via the orchestrator, instead of auto-approving the
    /// first option. The Docker sandbox is still the primary security
    /// boundary; this adds a second, user-consent layer.
    pub surface_permissions: bool,
}

/// The ACP bridge runtime.
pub struct AcpBridgeRuntime {
    config: AcpBridgeConfig,
    client: Arc<WorkerHttpClient>,
}

impl AcpBridgeRuntime {
    /// Create a new bridge runtime.
    ///
    /// Reads `IRONCLAW_WORKER_TOKEN` from the environment for auth.
    pub fn new(config: AcpBridgeConfig) -> Result<Self, WorkerError> {
        let client = Arc::new(WorkerHttpClient::from_env(
            config.orchestrator_url.clone(),
            config.job_id,
        )?);

        Ok(Self { config, client })
    }

    /// Run the bridge: fetch job, spawn ACP agent, stream events, handle follow-ups.
    pub async fn run(&self) -> Result<(), WorkerError> {
        // Fetch the job description from the orchestrator
        let job = self.client.get_job().await?;

        tracing::info!(
            job_id = %self.config.job_id,
            "Starting ACP bridge for: {}",
            truncate(&job.description, 100)
        );

        // Fetch credentials for injection into the spawned Command
        let credentials = self.client.fetch_credentials().await?;
        let mut extra_env = self.config.agent_env.clone();
        for cred in &credentials {
            extra_env.insert(cred.env_var.clone(), cred.value.clone());
        }
        if !credentials.is_empty() {
            tracing::info!(
                job_id = %self.config.job_id,
                "Fetched {} credential(s) for child process injection",
                credentials.len()
            );
        }

        // Report that we're running
        self.client
            .report_status(&crate::worker::api::StatusUpdate {
                state: "running".to_string(),
                message: Some(format!("Spawning ACP agent: {}", self.config.agent_command)),
                iteration: 0,
            })
            .await?;

        // Run the ACP session, then emit exactly one terminal "result" event
        // (so job monitors transition state) followed by a completion report.
        let (success, message) = match self.run_acp_session(&job.description, &extra_env).await {
            Ok(()) => (true, "ACP agent session completed".to_string()),
            Err(e) => {
                tracing::error!(job_id = %self.config.job_id, "ACP session failed: {}", e);
                (false, format!("ACP agent failed: {e}"))
            }
        };
        self.client
            .post_event(&JobEventPayload {
                event_type: "result".to_string(),
                data: json!({
                    "status": if success { "completed" } else { "error" },
                    "message": &message,
                }),
            })
            .await;
        self.client
            .report_complete(&CompletionReport {
                success,
                message: Some(message),
                iterations: 1,
            })
            .await?;

        Ok(())
    }

    /// Spawn the ACP agent and run the protocol lifecycle.
    async fn run_acp_session(
        &self,
        prompt: &str,
        extra_env: &HashMap<String, String>,
    ) -> Result<(), WorkerError> {
        let mut cmd = Command::new(&self.config.agent_command);
        cmd.args(&self.config.agent_args);
        cmd.envs(extra_env);
        cmd.current_dir("/workspace")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);

        let mut child = cmd.spawn().map_err(|e| WorkerError::ExecutionFailed {
            reason: format!(
                "failed to spawn ACP agent '{}': {}",
                self.config.agent_command, e
            ),
        })?;

        let child_stdin = child
            .stdin
            .take()
            .ok_or_else(|| WorkerError::ExecutionFailed {
                reason: "failed to capture ACP agent stdin".to_string(),
            })?;
        let child_stdout = child
            .stdout
            .take()
            .ok_or_else(|| WorkerError::ExecutionFailed {
                reason: "failed to capture ACP agent stdout".to_string(),
            })?;
        let child_stderr = child
            .stderr
            .take()
            .ok_or_else(|| WorkerError::ExecutionFailed {
                reason: "failed to capture ACP agent stderr".to_string(),
            })?;

        // Spawn stderr reader that forwards lines as status events
        let client_for_stderr = Arc::clone(&self.client);
        let job_id = self.config.job_id;
        let stderr_handle = tokio::spawn(async move {
            let reader = BufReader::new(child_stderr);
            let mut lines = reader.lines();
            while let Ok(Some(line)) = lines.next_line().await {
                tracing::debug!(job_id = %job_id, "acp agent stderr: {}", line);
                let payload = JobEventPayload {
                    event_type: "status".to_string(),
                    data: json!({ "message": line }),
                };
                client_for_stderr.post_event(&payload).await;
            }
        });

        // Run the ACP protocol inside a LocalSet (SDK futures are !Send)
        let client_for_acp = Arc::clone(&self.client);
        let prompt_owned = prompt.to_string();
        let job_id = self.config.job_id;
        let timeout = self.config.timeout;
        let surface_permissions = self.config.surface_permissions;

        // Clone client for follow-up loop
        let client_for_followup = Arc::clone(&self.client);

        // Monitor the child process so the follow-up loop can exit if the agent dies.
        // The oneshot is Send, so it crosses the LocalSet boundary cleanly.
        let (child_exit_rx, kill_tx) = spawn_child_monitor(child);

        let local_set = tokio::task::LocalSet::new();
        let acp_result = local_set
            .run_until(async move {
                let outgoing = child_stdin.compat_write();
                let incoming = child_stdout.compat();

                // Create ACP connection. The bridge runs inside a Docker
                // container; `surface_permissions` determines whether
                // `request_permission` is deferred to the host (user click in
                // the web UI) or auto-approved.
                let ironclaw_client =
                    IronClawAcpClient::new(Arc::clone(&client_for_acp), surface_permissions);

                let (conn, handle_io) =
                    acp::ClientSideConnection::new(ironclaw_client, outgoing, incoming, |fut| {
                        tokio::task::spawn_local(fut);
                    });
                tokio::task::spawn_local(handle_io);

                conn.initialize(ironclaw_init_request())
                    .await
                    .map_err(|e| WorkerError::ExecutionFailed {
                        reason: format!("ACP initialize failed: {}", e),
                    })?;

                tracing::info!(job_id = %job_id, "ACP handshake complete");

                // Create a new session
                let workspace = std::env::current_dir().unwrap_or_else(|e| {
                    tracing::warn!(
                        job_id = %job_id,
                        error = %e,
                        "failed to read current_dir; falling back to /workspace",
                    );
                    "/workspace".into()
                });
                let session_response = conn
                    .new_session(acp::NewSessionRequest::new(workspace))
                    .await
                    .map_err(|e| WorkerError::ExecutionFailed {
                        reason: format!("ACP new_session failed: {}", e),
                    })?;

                let session_id = session_response.session_id.clone();
                tracing::info!(job_id = %job_id, session_id = %session_id, "ACP session created");

                // Send the job description as a prompt
                let prompt_result = tokio::time::timeout(timeout, async {
                    conn.prompt(acp::PromptRequest::new(
                        session_id.clone(),
                        vec![prompt_owned.into()],
                    ))
                    .await
                })
                .await;

                let prompt_response = match prompt_result {
                    Ok(Ok(resp)) => resp,
                    Ok(Err(e)) => {
                        return Err(WorkerError::ExecutionFailed {
                            reason: format!("ACP prompt failed: {}", e),
                        });
                    }
                    Err(_) => {
                        return Err(WorkerError::ExecutionFailed {
                            reason: "ACP prompt timed out".to_string(),
                        });
                    }
                };

                // Report prompt result
                let result_payload = stop_reason_to_turn_event(
                    &prompt_response.stop_reason,
                    &session_id.to_string(),
                );
                client_for_acp.post_event(&result_payload).await;

                // Follow-up loop: poll for prompts, send additional prompt() calls.
                // Exits when: orchestrator sends done, or agent process exits.
                let prompt_sender = ConnPromptSender { conn: &conn };
                run_follow_up_loop(
                    &client_for_followup,
                    &prompt_sender,
                    &client_for_acp,
                    &session_id,
                    child_exit_rx,
                    job_id,
                )
                .await?;

                Ok::<(), WorkerError>(())
            })
            .await;

        // Kill the child on protocol failure so stderr closes (see kill channel above).
        if acp_result.is_err() {
            let _ = kill_tx.send(());
        }

        let _ = stderr_handle.await;

        acp_result
    }
}

// ==================== ACP Client trait implementation ====================

/// Sink for ACP events translated from session notifications.
///
/// The bridge posts events to the orchestrator via HTTP; the CLI test
/// command prints them to stdout. Both share the same `IronClawAcpClient`.
#[doc(hidden)]
pub trait AcpEventSink: 'static {
    fn emit_event(&self, payload: &JobEventPayload) -> impl std::future::Future<Output = ()>;
}

impl AcpEventSink for Arc<WorkerHttpClient> {
    async fn emit_event(&self, payload: &JobEventPayload) {
        self.post_event(payload).await;
    }
}

/// Gateway for deferring ACP permission decisions to the orchestrator.
///
/// The bridge holds one of these so `request_permission` can register a
/// pending gate and long-poll for the user's decision, instead of
/// auto-approving.
#[doc(hidden)]
pub trait AcpPermissionGateway: 'static {
    fn register_permission(
        &self,
        req: &PermissionRequestRegister,
    ) -> impl std::future::Future<Output = Result<(), WorkerError>>;

    fn poll_permission(
        &self,
        permission_id: Uuid,
    ) -> impl std::future::Future<Output = Result<Option<PermissionDecision>, WorkerError>>;
}

impl AcpPermissionGateway for Arc<WorkerHttpClient> {
    async fn register_permission(
        &self,
        req: &PermissionRequestRegister,
    ) -> Result<(), WorkerError> {
        WorkerHttpClient::register_permission(self, req).await
    }

    async fn poll_permission(
        &self,
        permission_id: Uuid,
    ) -> Result<Option<PermissionDecision>, WorkerError> {
        WorkerHttpClient::poll_permission(self, permission_id).await
    }
}

/// Source of follow-up prompts from the orchestrator.
pub(crate) trait FollowUpPromptSource {
    fn poll_prompt(
        &self,
    ) -> impl std::future::Future<Output = Result<Option<crate::worker::api::PromptResponse>, WorkerError>>;
}

impl FollowUpPromptSource for Arc<WorkerHttpClient> {
    async fn poll_prompt(&self) -> Result<Option<crate::worker::api::PromptResponse>, WorkerError> {
        WorkerHttpClient::poll_prompt(self).await
    }
}

/// Sends a follow-up prompt to the ACP agent.
trait AcpPromptSender {
    fn send_prompt(
        &self,
        session_id: acp::SessionId,
        content: String,
    ) -> impl std::future::Future<Output = acp::Result<acp::PromptResponse>>;
}

/// IronClaw's implementation of the ACP Client trait.
///
/// Handles callbacks from the agent: session notifications (streaming output)
/// and permission requests. When `surface_permissions` is `true`, permission
/// requests are deferred to the orchestrator via `AcpPermissionGateway` and
/// the agent blocks until the user resolves the gate in the web UI. When
/// `false`, permission requests are auto-approved (legacy: container is the
/// sole boundary).
///
/// Generic over the event sink + gateway so both the container bridge and
/// CLI test command can reuse it.
#[doc(hidden)]
pub struct IronClawAcpClient<S: AcpEventSink + AcpPermissionGateway> {
    sink: S,
    surface_permissions: bool,
    permission_timeout: Duration,
}

impl<S: AcpEventSink + AcpPermissionGateway> IronClawAcpClient<S> {
    #[doc(hidden)]
    pub fn new(sink: S, surface_permissions: bool) -> Self {
        Self {
            sink,
            surface_permissions,
            permission_timeout: ACP_PERMISSION_TIMEOUT,
        }
    }

    /// Override the overall permission-request timeout. Intended for tests
    /// that exercise the inactivity-to-Cancelled branch in bounded time.
    #[cfg(test)]
    pub(crate) fn with_permission_timeout(mut self, timeout: Duration) -> Self {
        self.permission_timeout = timeout;
        self
    }
}

/// Overall deadline for a single ACP permission request when
/// `surface_permissions=true`. If the user hasn't resolved the gate within
/// this window, we return `Cancelled` to the ACP agent — safe default on
/// inactivity.
const ACP_PERMISSION_TIMEOUT: Duration = Duration::from_secs(600);
/// Backoff between poll retries after a transient error.
const ACP_PERMISSION_POLL_BACKOFF: Duration = Duration::from_secs(2);

#[async_trait::async_trait(?Send)]
impl<S: AcpEventSink + AcpPermissionGateway> acp::Client for IronClawAcpClient<S> {
    async fn request_permission(
        &self,
        args: acp::RequestPermissionRequest,
    ) -> acp::Result<acp::RequestPermissionResponse> {
        if !self.surface_permissions {
            // Legacy behavior: auto-select first option. Docker container is
            // the security boundary.
            let Some(first_option) = args.options.first() else {
                return Err(acp::Error::invalid_params());
            };
            let option_id = first_option.option_id.clone();
            return Ok(acp::RequestPermissionResponse::new(
                acp::RequestPermissionOutcome::Selected(acp::SelectedPermissionOutcome::new(
                    option_id,
                )),
            ));
        }

        if args.options.is_empty() {
            return Err(acp::Error::invalid_params());
        }

        let permission_id = Uuid::new_v4();
        let payload = build_register_payload(permission_id, &args);

        if let Err(e) = self.sink.register_permission(&payload).await {
            tracing::warn!(
                permission_id = %permission_id,
                "ACP permission registration failed: {}. Defaulting to Cancelled.",
                e
            );
            return Ok(acp::RequestPermissionResponse::new(
                acp::RequestPermissionOutcome::Cancelled,
            ));
        }

        let deadline = tokio::time::Instant::now() + self.permission_timeout;
        loop {
            if tokio::time::Instant::now() >= deadline {
                tracing::warn!(
                    permission_id = %permission_id,
                    "ACP permission request timed out after {:?}. Returning Cancelled.",
                    self.permission_timeout,
                );
                return Ok(acp::RequestPermissionResponse::new(
                    acp::RequestPermissionOutcome::Cancelled,
                ));
            }

            match self.sink.poll_permission(permission_id).await {
                Ok(Some(PermissionDecision::Selected { option_id })) => {
                    return Ok(acp::RequestPermissionResponse::new(
                        acp::RequestPermissionOutcome::Selected(
                            acp::SelectedPermissionOutcome::new(acp::PermissionOptionId::new(
                                option_id,
                            )),
                        ),
                    ));
                }
                Ok(Some(PermissionDecision::Cancelled)) => {
                    return Ok(acp::RequestPermissionResponse::new(
                        acp::RequestPermissionOutcome::Cancelled,
                    ));
                }
                Ok(None) => {
                    // 204 No Content — server poll window elapsed. Yield
                    // briefly so we don't tight-loop when the client is
                    // mocked / the server returns 204 instantly.
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
                Err(e) => {
                    tracing::warn!(
                        permission_id = %permission_id,
                        "ACP permission poll failed: {}. Retrying.",
                        e
                    );
                    tokio::time::sleep(ACP_PERMISSION_POLL_BACKOFF).await;
                }
            }
        }
    }

    async fn session_notification(&self, args: acp::SessionNotification) -> acp::Result<()> {
        if let Some(payload) = session_update_to_payload(&args.update) {
            self.sink.emit_event(&payload).await;
        }
        Ok(())
    }
}

/// Translate an ACP `PermissionOptionKind` to the wire-stable enum.
fn permission_option_kind_to_dto(kind: acp::PermissionOptionKind) -> PermissionOptionKindDto {
    match kind {
        acp::PermissionOptionKind::AllowOnce => PermissionOptionKindDto::AllowOnce,
        acp::PermissionOptionKind::AllowAlways => PermissionOptionKindDto::AllowAlways,
        acp::PermissionOptionKind::RejectOnce => PermissionOptionKindDto::RejectOnce,
        acp::PermissionOptionKind::RejectAlways => PermissionOptionKindDto::RejectAlways,
        // The ACP enum is `#[non_exhaustive]`; treat unknowns as a rejection
        // hint so we never silently widen permission.
        _ => PermissionOptionKindDto::RejectOnce,
    }
}

fn build_register_payload(
    permission_id: Uuid,
    args: &acp::RequestPermissionRequest,
) -> PermissionRequestRegister {
    let options = args
        .options
        .iter()
        .map(|opt| PermissionOptionDto {
            option_id: opt.option_id.to_string(),
            name: opt.name.clone(),
            kind: permission_option_kind_to_dto(opt.kind),
        })
        .collect();
    let tool_name = args
        .tool_call
        .fields
        .title
        .clone()
        .unwrap_or_else(|| "(unknown)".to_string());
    let tool_kind = args
        .tool_call
        .fields
        .kind
        .and_then(|k| serde_json::to_value(k).ok())
        .and_then(|v| v.as_str().map(|s| s.to_string()));
    let locations = args
        .tool_call
        .fields
        .locations
        .as_ref()
        .map(|ls| ls.iter().map(|l| l.path.display().to_string()).collect())
        .unwrap_or_default();
    PermissionRequestRegister {
        permission_id,
        tool_call_id: args.tool_call.tool_call_id.to_string(),
        tool_name,
        tool_kind,
        raw_input: args.tool_call.fields.raw_input.clone(),
        locations,
        options,
    }
}

/// Build the standard IronClaw ACP initialization request.
#[doc(hidden)]
pub fn ironclaw_init_request() -> acp::InitializeRequest {
    acp::InitializeRequest::new(acp::ProtocolVersion::V1).client_info(
        acp::Implementation::new("ironclaw", env!("CARGO_PKG_VERSION")).title("IronClaw"),
    )
}

// ==================== Child process monitor ====================

/// Spawn a background task that monitors a child process for natural exit
/// or a kill signal. Returns `(child_exit_rx, kill_tx)`.
///
/// - `child_exit_rx`: fires with the exit code when the child terminates
/// - `kill_tx`: send `()` to terminate the child (e.g. on protocol failure
///   so stderr closes and the stderr reader task can finish)
fn spawn_child_monitor(
    mut child: tokio::process::Child,
) -> (
    tokio::sync::oneshot::Receiver<Option<i32>>,
    tokio::sync::oneshot::Sender<()>,
) {
    let (child_exit_tx, child_exit_rx) = tokio::sync::oneshot::channel();
    let (kill_tx, kill_rx) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        let exit_code_of =
            |r: std::io::Result<std::process::ExitStatus>| r.ok().and_then(|s| s.code());
        let exit_code = tokio::select! {
            status = child.wait() => exit_code_of(status),
            // Only fire on explicit send, not on sender drop.
            Ok(()) = kill_rx => {
                let _ = child.kill().await;
                exit_code_of(child.wait().await)
            }
        };
        let _ = child_exit_tx.send(exit_code);
    });
    (child_exit_rx, kill_tx)
}

// ==================== Follow-up loop ====================

/// Wrapper that implements `AcpPromptSender` for an ACP `ClientSideConnection`.
struct ConnPromptSender<'a, C: acp::Agent> {
    conn: &'a C,
}

impl<C: acp::Agent + 'static> AcpPromptSender for ConnPromptSender<'_, C> {
    async fn send_prompt(
        &self,
        session_id: acp::SessionId,
        content: String,
    ) -> acp::Result<acp::PromptResponse> {
        self.conn
            .prompt(acp::PromptRequest::new(session_id, vec![content.into()]))
            .await
    }
}

/// Maximum consecutive transient poll errors before giving up.
const MAX_CONSECUTIVE_POLL_ERRORS: u32 = 5;

/// Run the follow-up prompt loop.
///
/// Polls for follow-up prompts from the orchestrator, sends them to the ACP
/// agent, and translates results into events. Returns `Err` if any follow-up
/// prompt fails, so the caller can report `success: false`.
async fn run_follow_up_loop(
    prompt_source: &impl FollowUpPromptSource,
    agent: &impl AcpPromptSender,
    sink: &impl AcpEventSink,
    session_id: &acp::SessionId,
    mut child_exit_rx: tokio::sync::oneshot::Receiver<Option<i32>>,
    job_id: Uuid,
) -> Result<(), WorkerError> {
    let session_id_str = session_id.to_string();
    let mut consecutive_poll_errors: u32 = 0;
    loop {
        // Race poll_prompt against child exit so we detect process death
        // even during a long-poll HTTP request to the orchestrator.
        let poll_result = tokio::select! {
            result = prompt_source.poll_prompt() => result,
            exit_code = &mut child_exit_rx => {
                let code = exit_code.ok().flatten();
                tracing::debug!(job_id = %job_id, exit_code = ?code, "ACP agent exited, ending follow-up loop");
                break;
            }
        };

        let backoff = match poll_result {
            Ok(Some(follow_up)) => {
                consecutive_poll_errors = 0;
                if follow_up.done {
                    tracing::debug!(job_id = %job_id, "Orchestrator signaled done");
                    break;
                }
                tracing::debug!(job_id = %job_id, "Got follow-up prompt");

                let follow_result = agent
                    .send_prompt(session_id.clone(), follow_up.content)
                    .await;

                match follow_result {
                    Ok(resp) => {
                        let payload = stop_reason_to_turn_event(&resp.stop_reason, &session_id_str);
                        sink.emit_event(&payload).await;
                    }
                    Err(e) => {
                        let msg = format!("Follow-up prompt failed: {e}");
                        tracing::error!(job_id = %job_id, "{}", msg);
                        return Err(WorkerError::ExecutionFailed { reason: msg });
                    }
                }
                continue;
            }
            Ok(None) => {
                consecutive_poll_errors = 0;
                Duration::from_secs(2)
            }
            Err(e) => {
                // Permanent errors fail immediately; transient errors retry
                // with a cap. Follows the is_retryable pattern from llm/retry.rs.
                let is_retryable = matches!(&e, WorkerError::ConnectionFailed { .. });
                if !is_retryable {
                    let msg = format!("Prompt polling failed (permanent): {e}");
                    tracing::error!(job_id = %job_id, "{}", msg);
                    return Err(WorkerError::ExecutionFailed { reason: msg });
                }
                consecutive_poll_errors += 1;
                if consecutive_poll_errors >= MAX_CONSECUTIVE_POLL_ERRORS {
                    let msg = format!(
                        "Prompt polling exhausted {} retries: {e}",
                        MAX_CONSECUTIVE_POLL_ERRORS,
                    );
                    tracing::error!(job_id = %job_id, "{}", msg);
                    return Err(WorkerError::ExecutionFailed { reason: msg });
                }
                tracing::warn!(
                    job_id = %job_id,
                    attempt = consecutive_poll_errors,
                    "Transient poll error: {}", e,
                );
                Duration::from_secs(5)
            }
        };

        tokio::select! {
            _ = tokio::time::sleep(backoff) => {}
            exit_code = &mut child_exit_rx => {
                let code = exit_code.ok().flatten();
                tracing::debug!(job_id = %job_id, exit_code = ?code, "ACP agent exited, ending follow-up loop");
                break;
            }
        }
    }
    Ok(())
}

// ==================== Event translation ====================

/// Convert an ACP `SessionUpdate` into an IronClaw `JobEventPayload`.
fn session_update_to_payload(update: &acp::SessionUpdate) -> Option<JobEventPayload> {
    match update {
        acp::SessionUpdate::AgentMessageChunk(chunk) => text_from_content_block(&chunk.content)
            .map(|text| JobEventPayload {
                event_type: "message".to_string(),
                data: json!({
                    "role": "assistant",
                    "content": text,
                }),
            }),
        acp::SessionUpdate::AgentThoughtChunk(chunk) => text_from_content_block(&chunk.content)
            .map(|text| JobEventPayload {
                event_type: "status".to_string(),
                data: json!({
                    "message": text,
                    "type": "thought",
                }),
            }),
        acp::SessionUpdate::ToolCall(tool_call) => Some(JobEventPayload {
            event_type: "tool_use".to_string(),
            data: json!({
                "tool_name": tool_call.title,
                "tool_use_id": tool_call.tool_call_id.to_string(),
            }),
        }),
        acp::SessionUpdate::ToolCallUpdate(update) => Some(JobEventPayload {
            event_type: "tool_result".to_string(),
            data: json!({
                "tool_use_id": update.tool_call_id.to_string(),
            }),
        }),
        _ => Some(JobEventPayload {
            event_type: "status".to_string(),
            data: json!({ "message": "ACP session update" }),
        }),
    }
}

/// Extract text from a `ContentBlock`, returning `None` for non-text blocks.
fn text_from_content_block(block: &acp::ContentBlock) -> Option<&str> {
    match block {
        acp::ContentBlock::Text(text_content) => Some(&text_content.text),
        _ => None,
    }
}

/// Convert an ACP `StopReason` into a non-terminal per-turn event.
///
/// Uses `event_type: "turn_result"` so the orchestrator maps it to
/// `AppEvent::JobStatus` (not `JobResult`). The terminal `"result"` event
/// is emitted exactly once by `run()` after the entire session ends.
fn stop_reason_to_turn_event(reason: &acp::StopReason, session_id: &str) -> JobEventPayload {
    let (status, message) = match reason {
        acp::StopReason::EndTurn => ("completed", "Agent completed successfully"),
        acp::StopReason::MaxTokens => ("error", "Agent reached max tokens"),
        acp::StopReason::MaxTurnRequests => ("error", "Agent reached max turn requests"),
        acp::StopReason::Refusal => ("error", "Agent refused to continue"),
        acp::StopReason::Cancelled => ("cancelled", "Agent was cancelled"),
        _ => ("completed", "Agent finished"),
    };
    JobEventPayload {
        event_type: "turn_result".to_string(),
        data: json!({
            "status": status,
            "session_id": session_id,
            "message": message,
        }),
    }
}

fn truncate(s: &str, max_len: usize) -> &str {
    if s.len() <= max_len {
        s
    } else {
        let mut end = max_len;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        &s[..end] // safety: end is validated by is_char_boundary loop above
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_update_agent_message_text() {
        let update = acp::SessionUpdate::AgentMessageChunk(acp::ContentChunk::new(
            acp::ContentBlock::Text(acp::TextContent::new("Hello world")),
        ));
        let payload = session_update_to_payload(&update).unwrap();
        assert_eq!(payload.event_type, "message");
        assert_eq!(payload.data["role"], "assistant");
        assert_eq!(payload.data["content"], "Hello world");
    }

    #[test]
    fn test_session_update_agent_thought_text() {
        let update = acp::SessionUpdate::AgentThoughtChunk(acp::ContentChunk::new(
            acp::ContentBlock::Text(acp::TextContent::new("Thinking...")),
        ));
        let payload = session_update_to_payload(&update).unwrap();
        assert_eq!(payload.event_type, "status");
        assert_eq!(payload.data["type"], "thought");
        assert_eq!(payload.data["message"], "Thinking...");
    }

    #[test]
    fn test_session_update_agent_message_image_ignored() {
        let update = acp::SessionUpdate::AgentMessageChunk(acp::ContentChunk::new(
            acp::ContentBlock::Image(acp::ImageContent::new("base64data", "image/png")),
        ));
        assert!(session_update_to_payload(&update).is_none());
    }

    #[test]
    fn test_stop_reason_end_turn() {
        let payload = stop_reason_to_turn_event(&acp::StopReason::EndTurn, "sid-1");
        assert_eq!(payload.event_type, "turn_result");
        assert_eq!(payload.data["status"], "completed");
    }

    #[test]
    fn test_stop_reason_max_tokens() {
        let payload = stop_reason_to_turn_event(&acp::StopReason::MaxTokens, "sid-1");
        assert_eq!(payload.data["status"], "error");
    }

    #[test]
    fn test_stop_reason_cancelled() {
        let payload = stop_reason_to_turn_event(&acp::StopReason::Cancelled, "sid-1");
        assert_eq!(payload.data["status"], "cancelled");
    }

    #[test]
    fn test_stop_reason_refusal() {
        let payload = stop_reason_to_turn_event(&acp::StopReason::Refusal, "sid-1");
        assert_eq!(payload.data["status"], "error");
        assert_eq!(payload.data["message"], "Agent refused to continue");
    }

    #[test]
    fn test_session_update_tool_call() {
        let update = acp::SessionUpdate::ToolCall(acp::ToolCall::new("tc-1", "Running tests"));
        let payload = session_update_to_payload(&update).unwrap();
        assert_eq!(payload.event_type, "tool_use");
        assert_eq!(payload.data["tool_name"], "Running tests");
        assert_eq!(payload.data["tool_use_id"], "tc-1");
    }

    #[test]
    fn test_session_update_tool_call_update() {
        let fields = acp::ToolCallUpdateFields::new();
        let update = acp::SessionUpdate::ToolCallUpdate(acp::ToolCallUpdate::new("tc-1", fields));
        let payload = session_update_to_payload(&update).unwrap();
        assert_eq!(payload.event_type, "tool_result");
        assert_eq!(payload.data["tool_use_id"], "tc-1");
    }

    #[test]
    fn test_session_update_thought_image_ignored() {
        let update = acp::SessionUpdate::AgentThoughtChunk(acp::ContentChunk::new(
            acp::ContentBlock::Image(acp::ImageContent::new("data", "image/png")),
        ));
        assert!(session_update_to_payload(&update).is_none());
    }

    #[test]
    fn test_stop_reason_max_turn_requests() {
        let payload = stop_reason_to_turn_event(&acp::StopReason::MaxTurnRequests, "sid-1");
        assert_eq!(payload.data["status"], "error");
        assert_eq!(payload.data["message"], "Agent reached max turn requests");
    }

    #[test]
    fn test_stop_reason_includes_session_id() {
        let payload = stop_reason_to_turn_event(&acp::StopReason::EndTurn, "my-session-42");
        assert_eq!(payload.data["session_id"], "my-session-42");
    }

    #[test]
    fn test_text_from_content_block_text() {
        let block = acp::ContentBlock::Text(acp::TextContent::new("hello"));
        assert_eq!(text_from_content_block(&block), Some("hello"));
    }

    #[test]
    fn test_text_from_content_block_image_returns_none() {
        let block = acp::ContentBlock::Image(acp::ImageContent::new("data", "image/png"));
        assert!(text_from_content_block(&block).is_none());
    }

    #[test]
    fn test_truncate() {
        assert_eq!(truncate("hello", 10), "hello");
        assert_eq!(truncate("hello world", 5), "hello");
        assert_eq!(truncate("", 5), "");
    }

    #[test]
    fn test_truncate_multibyte_safe() {
        // 2-byte UTF-8 char: "é" is 0xC3 0xA9
        let s = "café";
        assert_eq!(truncate(s, 3), "caf"); // doesn't split the é
        assert_eq!(truncate(s, 5), "café"); // includes full char
    }

    // ==================== Follow-up loop stubs & tests ====================

    use crate::worker::api::PromptResponse;
    use std::sync::Mutex;

    /// Stub event sink that collects emitted events for assertion.
    struct CollectingSink {
        events: Mutex<Vec<JobEventPayload>>,
    }

    impl CollectingSink {
        fn new() -> Self {
            Self {
                events: Mutex::new(Vec::new()),
            }
        }

        fn events(&self) -> Vec<JobEventPayload> {
            self.events.lock().unwrap().clone()
        }
    }

    impl AcpEventSink for CollectingSink {
        async fn emit_event(&self, payload: &JobEventPayload) {
            self.events.lock().unwrap().push(payload.clone());
        }
    }

    /// Stub prompt source that yields a sequence of responses then returns None.
    struct StubPromptSource {
        responses: Mutex<Vec<Result<Option<PromptResponse>, WorkerError>>>,
    }

    impl StubPromptSource {
        fn new(mut responses: Vec<Result<Option<PromptResponse>, WorkerError>>) -> Self {
            responses.reverse(); // so we can pop (FIFO)
            Self {
                responses: Mutex::new(responses),
            }
        }
    }

    impl FollowUpPromptSource for StubPromptSource {
        async fn poll_prompt(&self) -> Result<Option<PromptResponse>, WorkerError> {
            self.responses.lock().unwrap().pop().unwrap_or(Ok(None))
        }
    }

    /// Stub ACP prompt sender that returns pre-configured results.
    struct StubAcpPromptSender {
        results: Mutex<Vec<acp::Result<acp::PromptResponse>>>,
    }

    impl StubAcpPromptSender {
        fn new(mut results: Vec<acp::Result<acp::PromptResponse>>) -> Self {
            results.reverse(); // so we can pop (FIFO)
            Self {
                results: Mutex::new(results),
            }
        }
    }

    impl AcpPromptSender for StubAcpPromptSender {
        async fn send_prompt(
            &self,
            _session_id: acp::SessionId,
            _content: String,
        ) -> acp::Result<acp::PromptResponse> {
            self.results
                .lock()
                .unwrap()
                .pop()
                .unwrap_or(Ok(acp::PromptResponse::new(acp::StopReason::EndTurn)))
        }
    }

    /// Regression test for #1915: follow-up prompt failure must return Err
    /// so that run() reports success: false.
    #[tokio::test]
    async fn follow_up_prompt_failure_returns_error() {
        let sink = CollectingSink::new();

        let prompt_source = StubPromptSource::new(vec![Ok(Some(PromptResponse {
            content: "follow up".to_string(),
            done: false,
        }))]);

        let agent = StubAcpPromptSender::new(vec![Err(acp::Error::internal_error())]);

        let (_tx, rx) = tokio::sync::oneshot::channel();
        let session_id = acp::SessionId::new("test-session");

        let result =
            run_follow_up_loop(&prompt_source, &agent, &sink, &session_id, rx, Uuid::nil()).await;

        assert!(
            result.is_err(),
            "Follow-up prompt failure must propagate as Err"
        );
        let err = result.unwrap_err();
        assert!(
            matches!(err, WorkerError::ExecutionFailed { .. }),
            "Error must be ExecutionFailed, got: {err}"
        );

        // No per-turn events should be emitted on failure — the terminal
        // "result" event is emitted by run(), not the follow-up loop.
        let events = sink.events();
        assert!(
            events.is_empty(),
            "Follow-up loop should not emit events on failure (terminal event comes from run())"
        );
    }

    /// Verify that a successful follow-up followed by orchestrator "done"
    /// signal returns Ok.
    #[tokio::test]
    async fn follow_up_prompt_success_then_done_returns_ok() {
        let sink = CollectingSink::new();

        let prompt_source = StubPromptSource::new(vec![
            Ok(Some(PromptResponse {
                content: "do more work".to_string(),
                done: false,
            })),
            Ok(Some(PromptResponse {
                content: String::new(),
                done: true,
            })),
        ]);

        let agent =
            StubAcpPromptSender::new(vec![Ok(acp::PromptResponse::new(acp::StopReason::EndTurn))]);

        let (_tx, rx) = tokio::sync::oneshot::channel();
        let session_id = acp::SessionId::new("test-session");

        let result =
            run_follow_up_loop(&prompt_source, &agent, &sink, &session_id, rx, Uuid::nil()).await;

        assert!(
            result.is_ok(),
            "Successful follow-up then done should be Ok"
        );

        let events = sink.events();
        assert!(
            events.iter().any(|e| e.event_type == "turn_result"),
            "Should emit turn_result (non-terminal) event for successful prompt"
        );
        assert!(
            !events.iter().any(|e| e.event_type == "result"),
            "Must NOT emit terminal result event (that comes from run())"
        );
    }

    /// Stub that blocks forever on poll_prompt, simulating a long-poll HTTP
    /// request that never returns. Only cancellation (via select!) can end this.
    struct ForeverPromptSource;

    impl FollowUpPromptSource for ForeverPromptSource {
        async fn poll_prompt(&self) -> Result<Option<PromptResponse>, WorkerError> {
            std::future::pending::<()>().await;
            unreachable!()
        }
    }

    /// Regression test: child exit must be detected even when poll_prompt()
    /// is blocked on a long-poll HTTP request. Without the select! fix around
    /// poll_prompt, this test times out because the loop never checks
    /// child_exit_rx during active polling.
    #[tokio::test]
    async fn follow_up_loop_exits_during_long_poll_when_child_dies() {
        let sink = CollectingSink::new();
        let prompt_source = ForeverPromptSource;
        let agent = StubAcpPromptSender::new(vec![]);

        let (tx, rx) = tokio::sync::oneshot::channel();
        let session_id = acp::SessionId::new("test-session");

        // Simulate child exit after a short delay
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            let _ = tx.send(Some(0));
        });

        let result = tokio::time::timeout(
            Duration::from_secs(5),
            run_follow_up_loop(&prompt_source, &agent, &sink, &session_id, rx, Uuid::nil()),
        )
        .await;

        assert!(
            result.is_ok(),
            "Loop should exit promptly when child dies during long poll, not timeout"
        );
        assert!(
            result.unwrap().is_ok(),
            "Child exit during poll should be a clean exit (Ok), not an error"
        );
    }

    /// Regression test for #1981 review feedback: when the ACP protocol fails
    /// while the subprocess is still alive, the kill channel must terminate the
    /// child so the stderr reader hits EOF and cleanup completes. Without the
    /// kill mechanism, stderr_handle.await blocks forever and the job hangs.
    #[tokio::test]
    async fn child_monitor_kills_process_so_stderr_reader_completes() {
        let mut child = tokio::process::Command::new("sleep")
            .arg("30")
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .expect("spawn sleep");

        let child_stderr = child.stderr.take().unwrap();

        // Use the actual production function
        let (child_exit_rx, kill_tx) = spawn_child_monitor(child);

        // Stderr reader — same pattern as production (lines 176-187)
        let stderr_handle = tokio::spawn(async move {
            let reader = BufReader::new(child_stderr);
            let mut lines = reader.lines();
            while let Ok(Some(_)) = lines.next_line().await {}
        });

        // Simulate protocol failure → send kill signal
        let _ = kill_tx.send(());

        // Both must complete; would hang forever without the kill channel
        let result = tokio::time::timeout(Duration::from_secs(5), async {
            let _ = child_exit_rx.await;
            let _ = stderr_handle.await;
        })
        .await;

        assert!(
            result.is_ok(),
            "kill signal should terminate child and unblock stderr reader"
        );
    }

    /// Verify the loop recovers from a transient polling error and
    /// processes the next prompt successfully.
    #[tokio::test(start_paused = true)]
    async fn follow_up_loop_recovers_from_transient_poll_error() {
        let sink = CollectingSink::new();

        // First poll returns an error, second returns a real prompt, third signals done.
        let prompt_source = StubPromptSource::new(vec![
            Err(WorkerError::ConnectionFailed {
                url: "http://test".to_string(),
                reason: "transient".to_string(),
            }),
            Ok(Some(PromptResponse {
                content: "do work".to_string(),
                done: false,
            })),
            Ok(Some(PromptResponse {
                content: String::new(),
                done: true,
            })),
        ]);

        let agent =
            StubAcpPromptSender::new(vec![Ok(acp::PromptResponse::new(acp::StopReason::EndTurn))]);

        let (_tx, rx) = tokio::sync::oneshot::channel();
        let session_id = acp::SessionId::new("test-session");

        let result =
            run_follow_up_loop(&prompt_source, &agent, &sink, &session_id, rx, Uuid::nil()).await;

        assert!(
            result.is_ok(),
            "Loop should recover from transient poll error"
        );

        let events = sink.events();
        assert!(
            events.iter().any(|e| e.event_type == "turn_result"),
            "Should emit turn_result event after recovery"
        );
    }

    /// Regression test for #1981: successful follow-up must emit "turn_result"
    /// (non-terminal), not "result" (terminal). Terminal events are emitted
    /// by run() so the job monitor sees exactly one completion signal.
    #[tokio::test]
    async fn follow_up_success_emits_turn_event_not_terminal() {
        let sink = CollectingSink::new();

        let prompt_source = StubPromptSource::new(vec![
            Ok(Some(PromptResponse {
                content: "do work".to_string(),
                done: false,
            })),
            Ok(Some(PromptResponse {
                content: String::new(),
                done: true,
            })),
        ]);

        let agent =
            StubAcpPromptSender::new(vec![Ok(acp::PromptResponse::new(acp::StopReason::EndTurn))]);

        let (_tx, rx) = tokio::sync::oneshot::channel();
        let session_id = acp::SessionId::new("test-session");

        let result =
            run_follow_up_loop(&prompt_source, &agent, &sink, &session_id, rx, Uuid::nil()).await;
        assert!(result.is_ok());

        let events = sink.events();
        assert_eq!(events.len(), 1, "Exactly one per-turn event expected");
        assert_eq!(events[0].event_type, "turn_result");
        assert_eq!(events[0].data["status"], "completed");
    }

    /// Regression test for #1981: permanent poll errors (OrchestratorRejected,
    /// LlmProxyFailed) must fail immediately without retry.
    #[tokio::test]
    async fn follow_up_loop_fails_on_permanent_poll_error() {
        let sink = CollectingSink::new();

        let prompt_source = StubPromptSource::new(vec![Err(WorkerError::OrchestratorRejected {
            job_id: Uuid::nil(),
            reason: "prompt endpoint returned 404 Not Found".to_string(),
        })]);

        let agent = StubAcpPromptSender::new(vec![]);

        let (_tx, rx) = tokio::sync::oneshot::channel();
        let session_id = acp::SessionId::new("test-session");

        let result =
            run_follow_up_loop(&prompt_source, &agent, &sink, &session_id, rx, Uuid::nil()).await;

        assert!(result.is_err(), "Permanent error must fail immediately");
        let err = result.unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("permanent"),
            "Error message should indicate permanent failure, got: {msg}"
        );
    }

    // ==================== request_permission tests ====================

    use acp::Client as _;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::sync::Mutex as AsyncMutex;

    /// Stub permission gateway that drives `request_permission`.
    ///
    /// `poll_results` is a queue — each call to `poll_permission` pops one.
    /// When drained, returns `Ok(None)` forever, which exercises the timeout
    /// branch when paired with `tokio::test(start_paused = true)`.
    struct StubPermissionGateway {
        register_result: AsyncMutex<Option<Result<(), WorkerError>>>,
        register_calls: AtomicUsize,
        poll_results: AsyncMutex<Vec<Result<Option<PermissionDecision>, WorkerError>>>,
    }

    impl StubPermissionGateway {
        fn new(
            register_result: Result<(), WorkerError>,
            mut poll_results: Vec<Result<Option<PermissionDecision>, WorkerError>>,
        ) -> Self {
            poll_results.reverse(); // pop() takes from the end
            Self {
                register_result: AsyncMutex::new(Some(register_result)),
                register_calls: AtomicUsize::new(0),
                poll_results: AsyncMutex::new(poll_results),
            }
        }
    }

    impl AcpEventSink for StubPermissionGateway {
        async fn emit_event(&self, _payload: &JobEventPayload) {}
    }

    impl AcpPermissionGateway for StubPermissionGateway {
        async fn register_permission(
            &self,
            _req: &PermissionRequestRegister,
        ) -> Result<(), WorkerError> {
            self.register_calls.fetch_add(1, Ordering::SeqCst);
            self.register_result.lock().await.take().unwrap_or(Ok(()))
        }

        async fn poll_permission(
            &self,
            _permission_id: Uuid,
        ) -> Result<Option<PermissionDecision>, WorkerError> {
            self.poll_results.lock().await.pop().unwrap_or(Ok(None))
        }
    }

    fn sample_request() -> acp::RequestPermissionRequest {
        let fields = acp::ToolCallUpdateFields::new()
            .title(Some("shell".to_string()))
            .kind(Some(acp::ToolKind::Execute));
        acp::RequestPermissionRequest::new(
            acp::SessionId::new("test-session"),
            acp::ToolCallUpdate::new("tc-1", fields),
            vec![
                acp::PermissionOption::new(
                    "allow-once",
                    "Allow once",
                    acp::PermissionOptionKind::AllowOnce,
                ),
                acp::PermissionOption::new(
                    "reject",
                    "Reject",
                    acp::PermissionOptionKind::RejectOnce,
                ),
            ],
        )
    }

    #[tokio::test]
    async fn request_permission_auto_approves_when_surface_permissions_false() {
        let gateway = StubPermissionGateway::new(Ok(()), vec![]);
        let client = IronClawAcpClient::new(gateway, false);

        let resp = client.request_permission(sample_request()).await.unwrap();
        match resp.outcome {
            acp::RequestPermissionOutcome::Selected(sel) => {
                assert_eq!(sel.option_id.to_string(), "allow-once");
            }
            other => panic!("expected Selected, got {other:?}"),
        }
        // No HTTP calls in legacy mode.
        assert_eq!(
            client.sink.register_calls.load(Ordering::SeqCst),
            0,
            "legacy mode must not call the orchestrator"
        );
    }

    #[tokio::test]
    async fn request_permission_returns_selected_when_gateway_yields_decision() {
        let gateway = StubPermissionGateway::new(
            Ok(()),
            vec![Ok(Some(PermissionDecision::Selected {
                option_id: "reject".to_string(),
            }))],
        );
        let client = IronClawAcpClient::new(gateway, true);

        let resp = client.request_permission(sample_request()).await.unwrap();
        match resp.outcome {
            acp::RequestPermissionOutcome::Selected(sel) => {
                assert_eq!(sel.option_id.to_string(), "reject");
            }
            other => panic!("expected Selected, got {other:?}"),
        }
        assert_eq!(client.sink.register_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn request_permission_returns_cancelled_when_gateway_yields_cancelled() {
        let gateway =
            StubPermissionGateway::new(Ok(()), vec![Ok(Some(PermissionDecision::Cancelled))]);
        let client = IronClawAcpClient::new(gateway, true);

        let resp = client.request_permission(sample_request()).await.unwrap();
        assert!(matches!(
            resp.outcome,
            acp::RequestPermissionOutcome::Cancelled
        ));
    }

    #[tokio::test]
    async fn request_permission_returns_cancelled_when_register_fails() {
        let gateway = StubPermissionGateway::new(
            Err(WorkerError::ConnectionFailed {
                url: "http://test".to_string(),
                reason: "boom".to_string(),
            }),
            vec![],
        );
        let client = IronClawAcpClient::new(gateway, true);

        let resp = client.request_permission(sample_request()).await.unwrap();
        assert!(
            matches!(resp.outcome, acp::RequestPermissionOutcome::Cancelled),
            "register failure must default to Cancelled (safe default)"
        );
    }

    /// Timeout branch: gateway always returns `Ok(None)`; a short injected
    /// timeout ensures the loop exits quickly.
    #[tokio::test]
    async fn request_permission_returns_cancelled_on_overall_timeout() {
        let gateway = StubPermissionGateway::new(Ok(()), vec![]);
        let client = IronClawAcpClient::new(gateway, true)
            .with_permission_timeout(Duration::from_millis(50));

        let resp = tokio::time::timeout(
            Duration::from_secs(5),
            client.request_permission(sample_request()),
        )
        .await
        .expect("did not resolve within real timeout")
        .expect("request_permission returned Err, expected Ok(Cancelled)");
        assert!(matches!(
            resp.outcome,
            acp::RequestPermissionOutcome::Cancelled
        ));
    }

    #[tokio::test]
    async fn request_permission_rejects_empty_options_when_surfacing() {
        let gateway = StubPermissionGateway::new(Ok(()), vec![]);
        let client = IronClawAcpClient::new(gateway, true);

        let req = acp::RequestPermissionRequest::new(
            acp::SessionId::new("test-session"),
            acp::ToolCallUpdate::new("tc-1", acp::ToolCallUpdateFields::default()),
            vec![], // no options
        );
        let err = client.request_permission(req).await.unwrap_err();
        // `acp::Error::invalid_params()` — verify via debug repr.
        assert!(format!("{err:?}").to_lowercase().contains("invalid"));
    }

    #[test]
    fn permission_option_kind_maps_all_variants() {
        assert_eq!(
            permission_option_kind_to_dto(acp::PermissionOptionKind::AllowOnce),
            PermissionOptionKindDto::AllowOnce
        );
        assert_eq!(
            permission_option_kind_to_dto(acp::PermissionOptionKind::AllowAlways),
            PermissionOptionKindDto::AllowAlways
        );
        assert_eq!(
            permission_option_kind_to_dto(acp::PermissionOptionKind::RejectOnce),
            PermissionOptionKindDto::RejectOnce
        );
        assert_eq!(
            permission_option_kind_to_dto(acp::PermissionOptionKind::RejectAlways),
            PermissionOptionKindDto::RejectAlways
        );
    }

    #[test]
    fn build_register_payload_captures_tool_fields() {
        let req = sample_request();
        let permission_id = Uuid::new_v4();
        let payload = build_register_payload(permission_id, &req);

        assert_eq!(payload.permission_id, permission_id);
        assert_eq!(payload.tool_call_id, "tc-1");
        assert_eq!(payload.tool_name, "shell");
        assert_eq!(payload.tool_kind.as_deref(), Some("execute"));
        assert_eq!(payload.options.len(), 2);
        assert_eq!(payload.options[0].option_id, "allow-once");
        assert_eq!(payload.options[0].kind, PermissionOptionKindDto::AllowOnce);
        assert_eq!(payload.options[1].kind, PermissionOptionKindDto::RejectOnce);
    }

    #[test]
    fn build_register_payload_falls_back_when_title_missing() {
        let req = acp::RequestPermissionRequest::new(
            acp::SessionId::new("s"),
            acp::ToolCallUpdate::new("tc-1", acp::ToolCallUpdateFields::default()),
            vec![acp::PermissionOption::new(
                "a",
                "A",
                acp::PermissionOptionKind::AllowOnce,
            )],
        );
        let payload = build_register_payload(Uuid::new_v4(), &req);
        assert_eq!(payload.tool_name, "(unknown)");
        assert!(payload.tool_kind.is_none());
    }

    // ==================== end request_permission tests ====================

    /// Regression test for #1981: repeated transient poll errors must eventually
    /// give up after MAX_CONSECUTIVE_POLL_ERRORS retries.
    #[tokio::test(start_paused = true)]
    async fn follow_up_loop_exhausts_transient_retries() {
        let sink = CollectingSink::new();

        // Generate more errors than the retry cap
        let errors: Vec<Result<Option<PromptResponse>, WorkerError>> = (0
            ..MAX_CONSECUTIVE_POLL_ERRORS + 1)
            .map(|_| {
                Err(WorkerError::ConnectionFailed {
                    url: "http://test".to_string(),
                    reason: "connection refused".to_string(),
                })
            })
            .collect();
        let prompt_source = StubPromptSource::new(errors);

        let agent = StubAcpPromptSender::new(vec![]);

        let (_tx, rx) = tokio::sync::oneshot::channel();
        let session_id = acp::SessionId::new("test-session");

        let result =
            run_follow_up_loop(&prompt_source, &agent, &sink, &session_id, rx, Uuid::nil()).await;

        assert!(result.is_err(), "Should fail after exhausting retries");
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("exhausted"),
            "Error should mention retry exhaustion, got: {msg}"
        );
    }
}
