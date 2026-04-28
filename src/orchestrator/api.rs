//! Internal HTTP API for worker-to-orchestrator communication.
//!
//! This runs on a separate port (default 50051) from the web gateway.
//! All endpoints are authenticated via per-job bearer tokens.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, broadcast};
use uuid::Uuid;

use crate::channels::web::types::ToolDecisionDto;
use crate::db::Database;
use crate::gate::pending::PendingGate;
use crate::gate::store::PendingGateStore;
use crate::llm::{CompletionRequest, LlmProvider, ToolCompletionRequest};
use crate::orchestrator::acp_permissions::AcpPermissionStore;
use crate::orchestrator::auth::{TokenStore, worker_auth_middleware};
use crate::orchestrator::job_manager::ContainerJobManager;
use crate::secrets::SecretsStore;
use crate::worker::api::JobEventPayload;
use crate::worker::api::{
    CompletionReport, CredentialResponse, JobDescription, PermissionRequestRegister,
    ProxyCompletionRequest, ProxyCompletionResponse, ProxyToolCompletionRequest,
    ProxyToolCompletionResponse, StatusUpdate,
};
use ironclaw_common::{AppEvent, JobResultStatus};

/// A follow-up prompt queued for a Claude Code bridge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingPrompt {
    pub content: String,
    pub done: bool,
}

/// Shared state for the orchestrator API.
#[derive(Clone)]
pub struct OrchestratorState {
    pub llm: Arc<dyn LlmProvider>,
    pub job_manager: Arc<ContainerJobManager>,
    pub token_store: TokenStore,
    /// Broadcast channel for job events (consumed by the web gateway SSE).
    /// Tuple: (job_id, user_id, event).
    pub job_event_tx: Option<broadcast::Sender<(Uuid, String, AppEvent)>>,
    /// Buffered follow-up prompts for sandbox jobs, keyed by job_id.
    pub prompt_queue: Arc<Mutex<HashMap<Uuid, VecDeque<PendingPrompt>>>>,
    /// Database handle for persisting job events.
    pub store: Option<Arc<dyn Database>>,
    /// Encrypted secrets store for credential injection into containers.
    pub secrets_store: Option<Arc<dyn SecretsStore + Send + Sync>>,
    /// In-memory cache of job_id → user_id for SSE scoping and credential
    /// lookup. Lazily populated on first read via `resolve_job_owner`, which
    /// falls back to `get_sandbox_job` when the entry is missing. The DB is
    /// always the source of truth; the cache just avoids a round-trip on
    /// every job event for long-running jobs.
    pub job_owner_cache: Arc<std::sync::RwLock<HashMap<Uuid, String>>>,
    /// Pending gate store shared with the web gateway. `None` in minimal
    /// test setups; required for ACP permission prompt surfacing.
    pub pending_gates: Option<Arc<PendingGateStore>>,
    /// Per-ACP-permission correlation store. The bridge long-polls here
    /// while the orchestrator waits for a user to resolve the matching
    /// `PendingGate` via `/api/chat/gate/resolve`.
    pub acp_permissions: Arc<AcpPermissionStore>,
}

/// Maximum entries in the job_owner_cache before arbitrary entries are
/// evicted. HashMap iteration order is unspecified, so eviction is not
/// LRU/FIFO — for the cache's purpose (avoid a DB hit per SSE event of a
/// live job), this is adequate: the next read of an evicted entry just
/// refills from DB. Upgrade to `IndexMap` or `lru::LruCache` if we ever
/// need real recency ordering.
const MAX_JOB_OWNER_CACHE_SIZE: usize = 10_000;

impl OrchestratorState {
    /// Look up the job owner from cache, falling back to DB.
    async fn resolve_job_owner(&self, job_id: Uuid) -> Option<String> {
        let cached = self
            .job_owner_cache
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get(&job_id)
            .cloned();
        if let Some(uid) = cached {
            return Some(uid);
        }
        let uid = match self.store.as_ref() {
            Some(store) => store
                .get_sandbox_job(job_id)
                .await
                .ok()
                .flatten()
                .map(|j| j.user_id),
            None => None,
        };
        // Don't cache empty user_id — treat the same as None
        let uid = uid.filter(|u| !u.is_empty());
        if let Some(ref uid) = uid {
            self.cache_job_owner(job_id, uid.clone());
        }
        uid
    }

    fn cache_job_owner(&self, job_id: Uuid, user_id: String) {
        let mut cache = self
            .job_owner_cache
            .write()
            .unwrap_or_else(|e| e.into_inner());
        if cache.len() >= MAX_JOB_OWNER_CACHE_SIZE {
            let to_remove: Vec<Uuid> = cache
                .keys()
                .take(MAX_JOB_OWNER_CACHE_SIZE / 10)
                .copied()
                .collect();
            for k in to_remove {
                cache.remove(&k);
            }
        }
        cache.insert(job_id, user_id);
    }
}

/// The orchestrator's internal API server.
pub struct OrchestratorApi;

impl OrchestratorApi {
    /// Build the axum router for the internal API.
    pub fn router(state: OrchestratorState) -> Router {
        Router::new()
            // Worker routes: authenticated via route_layer middleware.
            .route("/worker/{job_id}/job", get(get_job))
            .route("/worker/{job_id}/llm/complete", post(llm_complete))
            .route(
                "/worker/{job_id}/llm/complete_with_tools",
                post(llm_complete_with_tools),
            )
            .route("/worker/{job_id}/status", post(report_status))
            .route("/worker/{job_id}/complete", post(report_complete))
            .route("/worker/{job_id}/event", post(job_event_handler))
            .route("/worker/{job_id}/prompt", get(get_prompt_handler))
            .route("/worker/{job_id}/credentials", get(get_credentials_handler))
            .route(
                "/worker/{job_id}/permission",
                post(register_permission_handler),
            )
            .route(
                "/worker/{job_id}/permission/{permission_id}",
                get(poll_permission_handler),
            )
            .route_layer(axum::middleware::from_fn_with_state(
                state.token_store.clone(),
                worker_auth_middleware,
            ))
            // Unauthenticated routes (added after the layer).
            .route("/health", get(health_check))
            .with_state(state)
    }

    /// Start the internal API server on the given port.
    ///
    /// On macOS/Windows (Docker Desktop), binds to loopback only because
    /// Docker Desktop routes `host.docker.internal` through its VM to the
    /// host's `127.0.0.1`.
    ///
    /// On Linux, containers reach the host via the docker bridge gateway
    /// (`172.17.0.1`), which is NOT loopback. Binding to `127.0.0.1`
    /// would reject container traffic. We bind to all interfaces instead
    /// and rely on `worker_auth_middleware` (applied as a route_layer on
    /// every `/worker/` endpoint) to reject unauthenticated requests.
    pub async fn start(
        state: OrchestratorState,
        port: u16,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let router = Self::router(state);
        let addr = if cfg!(target_os = "linux") {
            std::net::SocketAddr::from(([0, 0, 0, 0], port))
        } else {
            std::net::SocketAddr::from(([127, 0, 0, 1], port))
        };

        tracing::info!("Orchestrator internal API listening on {}", addr);

        let listener = tokio::net::TcpListener::bind(addr).await?;
        axum::serve(listener, router).await?;

        Ok(())
    }
}

// -- Handlers --
//
// All /worker/ handlers below are behind the worker_auth_middleware route_layer,
// so they don't need to validate tokens themselves.

async fn health_check() -> &'static str {
    "ok"
}

async fn get_job(
    State(state): State<OrchestratorState>,
    Path(job_id): Path<Uuid>,
) -> Result<Json<JobDescription>, StatusCode> {
    let handle = state
        .job_manager
        .get_handle(job_id)
        .await
        .ok_or(StatusCode::NOT_FOUND)?;

    Ok(Json(JobDescription {
        title: format!("Job {}", job_id),
        description: handle.task_description,
        project_dir: handle.project_dir.map(|p| p.display().to_string()),
    }))
}

async fn llm_complete(
    State(state): State<OrchestratorState>,
    Path(job_id): Path<Uuid>,
    Json(req): Json<ProxyCompletionRequest>,
) -> Result<Json<ProxyCompletionResponse>, StatusCode> {
    let completion_req = CompletionRequest {
        messages: req.messages,
        model: req.model,
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        stop_sequences: req.stop_sequences,
        metadata: std::collections::HashMap::new(),
    };

    let resp = state.llm.complete(completion_req).await.map_err(|e| {
        tracing::error!("LLM completion failed for job {}: {}", job_id, e);
        StatusCode::BAD_GATEWAY
    })?;

    Ok(Json(ProxyCompletionResponse {
        content: resp.content,
        input_tokens: resp.input_tokens,
        output_tokens: resp.output_tokens,
        finish_reason: format_finish_reason(resp.finish_reason),
        cache_read_input_tokens: resp.cache_read_input_tokens,
        cache_creation_input_tokens: resp.cache_creation_input_tokens,
    }))
}

async fn llm_complete_with_tools(
    State(state): State<OrchestratorState>,
    Path(job_id): Path<Uuid>,
    Json(req): Json<ProxyToolCompletionRequest>,
) -> Result<Json<ProxyToolCompletionResponse>, StatusCode> {
    let tool_req = ToolCompletionRequest {
        messages: req.messages,
        tools: req.tools,
        model: req.model,
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        stop_sequences: req.stop_sequences,
        tool_choice: req.tool_choice,
        metadata: std::collections::HashMap::new(),
    };

    let resp = state.llm.complete_with_tools(tool_req).await.map_err(|e| {
        tracing::error!("LLM tool completion failed for job {}: {}", job_id, e);
        StatusCode::BAD_GATEWAY
    })?;

    Ok(Json(ProxyToolCompletionResponse {
        content: resp.content,
        tool_calls: resp.tool_calls,
        input_tokens: resp.input_tokens,
        output_tokens: resp.output_tokens,
        finish_reason: format_finish_reason(resp.finish_reason),
        cache_read_input_tokens: resp.cache_read_input_tokens,
        cache_creation_input_tokens: resp.cache_creation_input_tokens,
    }))
}

async fn report_status(
    State(state): State<OrchestratorState>,
    Path(job_id): Path<Uuid>,
    Json(update): Json<StatusUpdate>,
) -> Result<StatusCode, StatusCode> {
    tracing::debug!(
        job_id = %job_id,
        state = %update.state,
        iteration = update.iteration,
        "Worker status update"
    );

    state
        .job_manager
        .update_worker_status(job_id, update.message, update.iteration)
        .await;

    Ok(StatusCode::OK)
}

async fn report_complete(
    State(state): State<OrchestratorState>,
    Path(job_id): Path<Uuid>,
    Json(report): Json<CompletionReport>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    if report.success {
        tracing::info!(
            job_id = %job_id,
            "Worker reported job complete"
        );
    } else {
        tracing::warn!(
            job_id = %job_id,
            message = ?report.message,
            "Worker reported job failure"
        );
    }

    // Store the result and clean up the container
    let result = crate::orchestrator::job_manager::CompletionResult {
        success: report.success,
        message: report.message.clone(),
    };
    if let Err(e) = state.job_manager.complete_job(job_id, result).await {
        tracing::error!(job_id = %job_id, "Failed to complete job cleanup: {}", e);
    }

    Ok(Json(serde_json::json!({"status": "ok"})))
}

// -- Sandbox job event handlers --

/// Receive a job event from a worker or Claude Code bridge and broadcast + persist it.
async fn job_event_handler(
    State(state): State<OrchestratorState>,
    Path(job_id): Path<Uuid>,
    Json(payload): Json<JobEventPayload>,
) -> Result<StatusCode, StatusCode> {
    tracing::debug!(
        job_id = %job_id,
        event_type = %payload.event_type,
        "Job event received"
    );

    // Persist to DB (fire-and-forget)
    if let Some(ref store) = state.store {
        let store = Arc::clone(store);
        let event_type = payload.event_type.clone();
        let data = payload.data.clone();
        tokio::spawn(async move {
            if let Err(e) = store.save_job_event(job_id, &event_type, &data).await {
                tracing::warn!(job_id = %job_id, "Failed to persist job event: {}", e);
            }
        });
    }

    // Convert to app event and broadcast
    let job_id_str = job_id.to_string();
    let app_event = match payload.event_type.as_str() {
        "message" => AppEvent::JobMessage {
            job_id: job_id_str,
            role: payload
                .data
                .get("role")
                .and_then(|v| v.as_str())
                .unwrap_or("assistant")
                .to_string(),
            content: payload
                .data
                .get("content")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
        },
        "tool_use" => AppEvent::JobToolUse {
            job_id: job_id_str,
            tool_name: payload
                .data
                .get("tool_name")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string(),
            input: payload
                .data
                .get("input")
                .cloned()
                .unwrap_or(serde_json::Value::Null),
        },
        "tool_result" => AppEvent::JobToolResult {
            job_id: job_id_str,
            tool_name: payload
                .data
                .get("tool_name")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string(),
            output: payload
                .data
                .get("output")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
        },
        "result" => {
            // JSON payloads from sandbox containers are a trust
            // boundary — parse the wire string into the typed enum and
            // fall back to `Failed` (not `Completed`) on unknown values
            // so a mis-labeled container run cannot silently register
            // as success.
            let raw_status = payload
                .data
                .get("status")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let status = raw_status.parse::<JobResultStatus>().unwrap_or_else(|_| {
                tracing::warn!(
                    job_id = %job_id,
                    raw_status = %raw_status,
                    "unknown job result status from container; defaulting to Failed"
                );
                JobResultStatus::Failed
            });
            AppEvent::JobResult {
                job_id: job_id_str,
                status,
                session_id: payload
                    .data
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string()),
                // NOTE: `fallback_deliverable` is currently always None in SSE events.
                // In-memory jobs store fallback data in JobContext.metadata (accessed via job_status tool).
                // Sandbox containers don't yet emit fallback data in their event payloads.
                // This field is forward-compatible infrastructure for when container workers
                // gain context/memory tracking capabilities.
                fallback_deliverable: payload.data.get("fallback_deliverable").cloned(),
            }
        }
        "reasoning" => {
            let narrative = payload
                .data
                .get("narrative")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let decisions = ToolDecisionDto::from_json_array(&payload.data["decisions"]);
            AppEvent::JobReasoning {
                job_id: job_id_str,
                narrative,
                decisions,
            }
        }
        _ => AppEvent::JobStatus {
            job_id: job_id_str,
            message: payload
                .data
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
        },
    };

    // Broadcast via the channel (if configured).
    if let Some(ref tx) = state.job_event_tx {
        // silent-ok: SSE broadcast is best-effort and already handles the
        // empty-user_id case below by sending with an empty scope string,
        // which the gateway filters out. A transient DB miss here should
        // not block the worker from reporting progress.
        let user_id = state.resolve_job_owner(job_id).await.unwrap_or_default();

        if user_id.is_empty() {
            let _ = tx.send((job_id, String::new(), app_event));
        } else {
            let _ = tx.send((job_id, user_id, app_event));
        }
    }

    Ok(StatusCode::OK)
}

/// Return the next queued follow-up prompt for a Claude Code bridge.
/// Returns 204 No Content if no prompt is available.
async fn get_prompt_handler(
    State(state): State<OrchestratorState>,
    Path(job_id): Path<Uuid>,
) -> Result<(StatusCode, Json<serde_json::Value>), StatusCode> {
    let mut queue = state.prompt_queue.lock().await;
    if let Some(prompts) = queue.get_mut(&job_id)
        && let Some(prompt) = prompts.pop_front()
    {
        return Ok((
            StatusCode::OK,
            Json(serde_json::json!({
                "content": prompt.content,
                "done": prompt.done,
            })),
        ));
    }

    // Return 204 with an empty body. The Json wrapper requires some value
    // but the status code signals "nothing here".
    Ok((StatusCode::NO_CONTENT, Json(serde_json::Value::Null)))
}

/// Serve decrypted credentials for a job's granted secrets.
///
/// Returns 204 if no grants exist, 503 if no secrets store is configured,
/// or a JSON array of `{ env_var, value }` pairs.
///
/// Credential lookup uses the job creator's user_id (resolved from
/// `job_owner_cache` or the database), NOT a global owner ID. This prevents
/// cross-tenant credential leakage where one user's sandbox job could
/// access another user's secrets. (#2068)
async fn get_credentials_handler(
    State(state): State<OrchestratorState>,
    Path(job_id): Path<Uuid>,
) -> Result<(StatusCode, Json<serde_json::Value>), StatusCode> {
    let grants = match state.token_store.get_grants(job_id).await {
        Some(g) if !g.is_empty() => g,
        _ => return Ok((StatusCode::NO_CONTENT, Json(serde_json::Value::Null))),
    };

    let secrets = state.secrets_store.as_ref().ok_or_else(|| {
        tracing::error!("Credentials requested but no secrets store configured");
        StatusCode::SERVICE_UNAVAILABLE
    })?;

    // Resolve the job creator's user_id from cache or DB. Fail closed.
    let job_user_id = state.resolve_job_owner(job_id).await.ok_or_else(|| {
        tracing::error!(
            job_id = %job_id,
            "Cannot resolve job owner for credential lookup; refusing to serve credentials"
        );
        StatusCode::FORBIDDEN
    })?;

    if job_user_id.is_empty() {
        tracing::error!(
            job_id = %job_id,
            "Job owner resolved to empty string; refusing to serve credentials"
        );
        return Err(StatusCode::FORBIDDEN);
    }

    let mut credentials: Vec<CredentialResponse> = Vec::with_capacity(grants.len());

    for grant in &grants {
        let decrypted = secrets
            .get_decrypted(&job_user_id, &grant.secret_name)
            .await
            .map_err(|e| {
                // Internal log is warn (not error) because the common case is
                // a benign "user has no such secret" miss. Kept above debug so
                // a real crypto/keychain failure surfaces in production logs —
                // 403 telemetry alone can't distinguish "wrong tenant" from
                // "crypto broken".
                tracing::warn!(
                    job_id = %job_id,
                    user_id = %job_user_id,
                    env_var = %grant.env_var,
                    "Failed to decrypt secret for credential grant: {}", e
                );
                // Wire response is 403 for all secret failures to avoid
                // leaking whether a secret exists under a different user's
                // scope.
                StatusCode::FORBIDDEN
            })?;

        // Record usage for audit trail
        if let Ok(secret) = secrets.get(&job_user_id, &grant.secret_name).await
            && let Err(e) = secrets.record_usage(secret.id).await
        {
            tracing::warn!(
                job_id = %job_id,
                "Failed to record credential usage: {}", e
            );
        }

        tracing::debug!(
            job_id = %job_id,
            env_var = %grant.env_var,
            "Serving credential to container"
        );

        credentials.push(CredentialResponse {
            env_var: grant.env_var.clone(),
            value: decrypted.expose().to_string(),
        });
    }

    Ok((
        StatusCode::OK,
        Json(serde_json::to_value(&credentials).unwrap_or(serde_json::Value::Null)),
    ))
}

// -- ACP permission handlers --

/// Namespace for derived ACP thread IDs.
///
/// Gates in `PendingGateStore` are keyed by `(user_id, ThreadId)`. ACP jobs
/// don't have an engine thread, so we derive a stable UUID from the job id
/// using UUIDv5. Keeping the derivation function-local (not exported)
/// prevents callers from relying on the scheme outside this module.
const ACP_THREAD_NAMESPACE: Uuid = Uuid::from_u128(0x8a4f_1c42_e6b2_4b5a_9c6d_3f8e_11ac_deaf);

/// How long a surfaced ACP permission request can sit unresolved. Should
/// exceed the bridge's overall request_permission timeout so that the
/// orchestrator's slot outlives the bridge's waiting call — if the bridge
/// gave up, we don't want the user to still be clicking "approve" on a
/// ghost.
const ACP_PERMISSION_GATE_TTL: chrono::Duration = chrono::Duration::minutes(15);

fn derive_acp_thread_id(job_id: Uuid) -> ironclaw_engine::ThreadId {
    let derived = Uuid::new_v5(&ACP_THREAD_NAMESPACE, job_id.as_bytes());
    ironclaw_engine::ThreadId(derived)
}

fn build_acp_pending_gate(
    job_id: Uuid,
    user_id: &str,
    req: &PermissionRequestRegister,
) -> PendingGate {
    use crate::worker::api::PermissionOptionKindDto;

    // ACP exposes allow-always per-option, not as a separate outcome. Pass
    // `allow_always: true` to the engine's ResumeKind when the agent
    // offered any AllowAlways option — that's the signal the UI uses to
    // render the "always approve" button.
    let allow_always = req
        .options
        .iter()
        .any(|o| matches!(o.kind, PermissionOptionKindDto::AllowAlways));

    let now = chrono::Utc::now();
    PendingGate {
        request_id: Uuid::new_v4(),
        gate_name: crate::bridge::ACP_PERMISSION_GATE_NAME.to_string(),
        user_id: user_id.to_string(),
        thread_id: derive_acp_thread_id(job_id),
        scope_thread_id: None,
        conversation_id: ironclaw_engine::ConversationId::new(),
        source_channel: "web".to_string(),
        action_name: req.tool_name.clone(),
        call_id: req.tool_call_id.clone(),
        parameters: req.raw_input.clone().unwrap_or(serde_json::Value::Null),
        display_parameters: None,
        description: format!("ACP agent wants to run {}", req.tool_name),
        resume_kind: ironclaw_engine::ResumeKind::Approval { allow_always },
        created_at: now,
        expires_at: now + ACP_PERMISSION_GATE_TTL,
        original_message: None,
        resume_output: None,
        paused_lease: None,
        approval_already_granted: false,
        job_id: Some(job_id),
    }
}

/// Register a permission request deferred by the ACP bridge.
///
/// Inserts a `PendingGate` for the user's web UI to render, broadcasts
/// `AppEvent::GateRequired` so the SSE stream reflects the new gate, and
/// parks a slot the bridge can long-poll until the user resolves the gate
/// via `/api/chat/gate/resolve`.
async fn register_permission_handler(
    State(state): State<OrchestratorState>,
    Path(job_id): Path<Uuid>,
    Json(req): Json<PermissionRequestRegister>,
) -> Result<StatusCode, StatusCode> {
    // Prefer an explicitly-injected store (tests); otherwise look up the
    // engine's canonical store lazily — engine init runs after orchestrator
    // setup so we can't thread it through at construction time.
    let pending_gates = match state.pending_gates.clone() {
        Some(store) => store,
        None => match crate::bridge::engine_pending_gate_store().await {
            Some(store) => store,
            None => {
                tracing::error!(
                    job_id = %job_id,
                    "ACP permission surfaced but engine not initialized yet"
                );
                return Err(StatusCode::SERVICE_UNAVAILABLE);
            }
        },
    };

    // Resolve and validate the job owner up front. Fail closed: any job
    // whose owner we can't identify cannot surface gates to a user.
    let user_id = state.resolve_job_owner(job_id).await.ok_or_else(|| {
        tracing::error!(
            job_id = %job_id,
            "Cannot resolve job owner for ACP permission; refusing"
        );
        StatusCode::FORBIDDEN
    })?;
    if user_id.is_empty() {
        return Err(StatusCode::FORBIDDEN);
    }

    let pending = build_acp_pending_gate(job_id, &user_id, &req);

    // Register the slot **before** inserting the gate so a fast user
    // resolution can't wake a slot that doesn't exist yet.
    state
        .acp_permissions
        .register(
            job_id,
            req.permission_id,
            pending.request_id,
            req.options.clone(),
            std::time::Duration::from_secs(3600),
        )
        .await
        .map_err(|e| {
            tracing::error!(
                job_id = %job_id,
                permission_id = %req.permission_id,
                "ACP permission registration collided: {}. Bridge likely reused an id.", e,
            );
            StatusCode::CONFLICT
        })?;

    pending_gates.insert(pending.clone()).await.map_err(|e| {
        tracing::warn!(
            job_id = %job_id,
            "Failed to insert ACP pending gate: {}", e
        );
        StatusCode::CONFLICT
    })?;

    if let Some(ref tx) = state.job_event_tx {
        // projection-exempt: acp_permission, source log is the PendingGate
        // insertion above; this orchestrator broadcast mirrors the bridge
        // router's `notify_pending_gate` projection.
        let _ = tx.send((
            job_id,
            user_id.clone(),
            crate::bridge::pending_gate_to_app_event(&pending, None),
        ));
    }

    Ok(StatusCode::ACCEPTED)
}

const ACP_PERMISSION_POLL_WINDOW: std::time::Duration = std::time::Duration::from_secs(25);

/// Long-poll for the user's decision on an ACP permission request.
///
/// Returns `200` with the decision when resolved, `204` if the poll window
/// elapsed without a decision (bridge should retry), or `404` if the
/// permission id is unknown (stale or expired).
async fn poll_permission_handler(
    State(state): State<OrchestratorState>,
    Path((job_id, permission_id)): Path<(Uuid, Uuid)>,
) -> Result<(StatusCode, Json<serde_json::Value>), StatusCode> {
    let Some(slot) = state.acp_permissions.get_slot(job_id, permission_id).await else {
        return Err(StatusCode::NOT_FOUND);
    };

    let decision = tokio::select! {
        d = slot.wait() => Some(d),
        _ = tokio::time::sleep(ACP_PERMISSION_POLL_WINDOW) => None,
    };

    match decision {
        Some(d) => {
            let value = serde_json::to_value(&d).map_err(|e| {
                tracing::error!(
                    job_id = %job_id,
                    permission_id = %permission_id,
                    error = %e,
                    "Failed to serialize PermissionDecision",
                );
                StatusCode::INTERNAL_SERVER_ERROR
            })?;
            Ok((StatusCode::OK, Json(value)))
        }
        None => Ok((StatusCode::NO_CONTENT, Json(serde_json::Value::Null))),
    }
}

fn format_finish_reason(reason: crate::llm::FinishReason) -> String {
    match reason {
        crate::llm::FinishReason::Stop => "stop".to_string(),
        crate::llm::FinishReason::Length => "length".to_string(),
        crate::llm::FinishReason::ToolUse => "tool_use".to_string(),
        crate::llm::FinishReason::ContentFilter => "content_filter".to_string(),
        crate::llm::FinishReason::Unknown => "unknown".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;
    use uuid::Uuid;

    use crate::orchestrator::auth::TokenStore;
    use crate::orchestrator::job_manager::{ContainerJobConfig, ContainerJobManager};
    use crate::testing::StubLlm;

    use super::*;

    fn test_state() -> OrchestratorState {
        let token_store = TokenStore::new();
        let jm = ContainerJobManager::new(ContainerJobConfig::default(), token_store.clone());
        OrchestratorState {
            llm: Arc::new(StubLlm::default()),
            job_manager: Arc::new(jm),
            token_store,
            job_event_tx: None,
            prompt_queue: Arc::new(Mutex::new(HashMap::new())),
            store: None,
            secrets_store: None,
            job_owner_cache: Arc::new(std::sync::RwLock::new(HashMap::new())),
            pending_gates: None,
            acp_permissions: Arc::new(AcpPermissionStore::new()),
        }
    }

    #[tokio::test]
    async fn health_requires_no_auth() {
        let state = test_state();
        let router = OrchestratorApi::router(state);

        let req = Request::builder()
            .uri("/health")
            .body(Body::empty())
            .unwrap();

        let resp = router.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn worker_route_rejects_missing_token() {
        let state = test_state();
        let router = OrchestratorApi::router(state);

        let job_id = Uuid::new_v4();
        let req = Request::builder()
            .uri(format!("/worker/{}/job", job_id))
            .body(Body::empty())
            .unwrap();

        let resp = router.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn worker_route_rejects_wrong_token() {
        let state = test_state();
        let router = OrchestratorApi::router(state);

        let job_id = Uuid::new_v4();
        let req = Request::builder()
            .uri(format!("/worker/{}/job", job_id))
            .header("Authorization", "Bearer totally-bogus")
            .body(Body::empty())
            .unwrap();

        let resp = router.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn worker_route_accepts_valid_token() {
        let state = test_state();
        let job_id = Uuid::new_v4();
        let token = state.token_store.create_token(job_id).await;

        let router = OrchestratorApi::router(state);

        let req = Request::builder()
            .uri(format!("/worker/{}/job", job_id))
            .header("Authorization", format!("Bearer {}", token))
            .body(Body::empty())
            .unwrap();

        let resp = router.oneshot(req).await.unwrap();
        // 404 because no container exists for this job_id, but NOT 401.
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn token_for_job_a_rejected_on_job_b() {
        let state = test_state();
        let job_a = Uuid::new_v4();
        let job_b = Uuid::new_v4();
        let token_a = state.token_store.create_token(job_a).await;

        let router = OrchestratorApi::router(state);

        // Use job_a's token to hit job_b's endpoint
        let req = Request::builder()
            .uri(format!("/worker/{}/job", job_b))
            .header("Authorization", format!("Bearer {}", token_a))
            .body(Body::empty())
            .unwrap();

        let resp = router.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    // -- Prompt queue tests --

    #[tokio::test]
    async fn prompt_returns_204_when_queue_empty() {
        let state = test_state();
        let job_id = Uuid::new_v4();
        let token = state.token_store.create_token(job_id).await;
        let router = OrchestratorApi::router(state);

        let req = Request::builder()
            .uri(format!("/worker/{}/prompt", job_id))
            .header("Authorization", format!("Bearer {}", token))
            .body(Body::empty())
            .unwrap();

        let resp = router.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NO_CONTENT);
    }

    #[tokio::test]
    async fn prompt_returns_queued_prompt() {
        let state = test_state();
        let job_id = Uuid::new_v4();
        let token = state.token_store.create_token(job_id).await;

        // Queue a prompt
        {
            let mut q = state.prompt_queue.lock().await;
            q.entry(job_id).or_default().push_back(PendingPrompt {
                content: "What is the status?".to_string(),
                done: false,
            });
        }

        let router = OrchestratorApi::router(state);
        let req = Request::builder()
            .uri(format!("/worker/{}/prompt", job_id))
            .header("Authorization", format!("Bearer {}", token))
            .body(Body::empty())
            .unwrap();

        let resp = router.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["content"], "What is the status?");
        assert_eq!(json["done"], false);
    }

    // -- Credentials handler tests --

    #[tokio::test]
    async fn credentials_returns_204_when_no_grants() {
        let state = test_state();
        let job_id = Uuid::new_v4();
        let token = state.token_store.create_token(job_id).await;
        let router = OrchestratorApi::router(state);

        let req = Request::builder()
            .uri(format!("/worker/{}/credentials", job_id))
            .header("Authorization", format!("Bearer {}", token))
            .body(Body::empty())
            .unwrap();

        let resp = router.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NO_CONTENT);
    }

    #[tokio::test]
    async fn credentials_returns_503_when_no_secrets_store() {
        let state = test_state();
        let job_id = Uuid::new_v4();
        let token = state.token_store.create_token(job_id).await;

        // Store grants so we get past the 204 check
        state
            .token_store
            .store_grants(
                job_id,
                vec![crate::orchestrator::auth::CredentialGrant {
                    secret_name: "test_secret".to_string(),
                    env_var: "TEST_SECRET".to_string(),
                }],
            )
            .await;

        // Populate job_owner_cache so credential handler can resolve user
        state
            .job_owner_cache
            .write()
            .unwrap()
            .insert(job_id, "test_user".to_string());

        let router = OrchestratorApi::router(state);
        let req = Request::builder()
            .uri(format!("/worker/{}/credentials", job_id))
            .header("Authorization", format!("Bearer {}", token))
            .body(Body::empty())
            .unwrap();

        let resp = router.oneshot(req).await.unwrap();
        // No secrets_store configured → 503
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn credentials_returns_secrets_when_store_configured() {
        use crate::testing::credentials::test_secrets_store;
        use secrecy::SecretString;
        let secrets_store = Arc::new(test_secrets_store());

        let job_creator = "alice_user";

        // Create a secret under the job creator's user_id
        secrets_store
            .create(
                job_creator,
                crate::secrets::CreateSecretParams {
                    name: "test_secret".to_string(),
                    value: SecretString::from("supersecretvalue".to_string()),
                    provider: None,
                    expires_at: None,
                },
            )
            .await
            .unwrap();

        let token_store = TokenStore::new();
        let jm = ContainerJobManager::new(ContainerJobConfig::default(), token_store.clone());
        let job_id = Uuid::new_v4();
        let token = token_store.create_token(job_id).await;
        token_store
            .store_grants(
                job_id,
                vec![crate::orchestrator::auth::CredentialGrant {
                    secret_name: "test_secret".to_string(),
                    env_var: "MY_SECRET".to_string(),
                }],
            )
            .await;

        // Pre-populate the job_owner_cache with the job creator's user_id
        let job_owner_cache = Arc::new(std::sync::RwLock::new(HashMap::new()));
        job_owner_cache
            .write()
            .unwrap()
            .insert(job_id, job_creator.to_string());

        let state = OrchestratorState {
            llm: Arc::new(StubLlm::default()),
            job_manager: Arc::new(jm),
            token_store,
            job_event_tx: None,
            prompt_queue: Arc::new(Mutex::new(HashMap::new())),
            store: None,
            secrets_store: Some(secrets_store),
            job_owner_cache,
            pending_gates: None,
            acp_permissions: Arc::new(AcpPermissionStore::new()),
        };

        let router = OrchestratorApi::router(state);
        let req = Request::builder()
            .uri(format!("/worker/{}/credentials", job_id))
            .header("Authorization", format!("Bearer {}", token))
            .body(Body::empty())
            .unwrap();

        let resp = router.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
        let json: Vec<serde_json::Value> = serde_json::from_slice(&body).unwrap();
        assert_eq!(json.len(), 1);
        assert_eq!(json[0]["env_var"], "MY_SECRET");
        assert_eq!(json[0]["value"], "supersecretvalue");
    }

    /// Regression test for #2068: credentials handler must use job creator's
    /// user_id, not a global owner. When no owner can be resolved, the handler
    /// must refuse (403) rather than falling back to any default.
    #[tokio::test]
    async fn credentials_returns_403_when_job_owner_unknown() {
        use crate::testing::credentials::test_secrets_store;
        use secrecy::SecretString;
        let secrets_store = Arc::new(test_secrets_store());

        // Store a secret under global owner -- must NOT be accessible
        secrets_store
            .create(
                "global_owner",
                crate::secrets::CreateSecretParams {
                    name: "test_secret".to_string(),
                    value: SecretString::from("should_not_leak".to_string()),
                    provider: None,
                    expires_at: None,
                },
            )
            .await
            .unwrap();

        let token_store = TokenStore::new();
        let jm = ContainerJobManager::new(ContainerJobConfig::default(), token_store.clone());
        let job_id = Uuid::new_v4();
        let token = token_store.create_token(job_id).await;
        token_store
            .store_grants(
                job_id,
                vec![crate::orchestrator::auth::CredentialGrant {
                    secret_name: "test_secret".to_string(),
                    env_var: "MY_SECRET".to_string(),
                }],
            )
            .await;

        // Deliberately do NOT populate job_owner_cache -- simulates unknown owner
        let state = OrchestratorState {
            llm: Arc::new(StubLlm::default()),
            job_manager: Arc::new(jm),
            token_store,
            job_event_tx: None,
            prompt_queue: Arc::new(Mutex::new(HashMap::new())),
            store: None,
            secrets_store: Some(secrets_store),
            job_owner_cache: Arc::new(std::sync::RwLock::new(HashMap::new())),
            pending_gates: None,
            acp_permissions: Arc::new(AcpPermissionStore::new()),
        };

        let router = OrchestratorApi::router(state);
        let req = Request::builder()
            .uri(format!("/worker/{}/credentials", job_id))
            .header("Authorization", format!("Bearer {}", token))
            .body(Body::empty())
            .unwrap();

        let resp = router.oneshot(req).await.unwrap();
        // No owner resolved and no DB → 403 Forbidden (fail closed)
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    /// Regression test for #2068: sandbox job credentials must be scoped to
    /// the job creator, not cross-tenant accessible.
    #[tokio::test]
    async fn credentials_uses_job_creator_not_other_user() {
        use crate::testing::credentials::test_secrets_store;
        use secrecy::SecretString;
        let secrets_store = Arc::new(test_secrets_store());

        // Store a secret under "owner_bob" -- the global owner
        secrets_store
            .create(
                "owner_bob",
                crate::secrets::CreateSecretParams {
                    name: "api_key".to_string(),
                    value: SecretString::from("bob_secret".to_string()),
                    provider: None,
                    expires_at: None,
                },
            )
            .await
            .unwrap();

        // Job was created by "alice" -- who does NOT have this secret
        let token_store = TokenStore::new();
        let jm = ContainerJobManager::new(ContainerJobConfig::default(), token_store.clone());
        let job_id = Uuid::new_v4();
        let token = token_store.create_token(job_id).await;
        token_store
            .store_grants(
                job_id,
                vec![crate::orchestrator::auth::CredentialGrant {
                    secret_name: "api_key".to_string(),
                    env_var: "API_KEY".to_string(),
                }],
            )
            .await;

        let job_owner_cache = Arc::new(std::sync::RwLock::new(HashMap::new()));
        job_owner_cache
            .write()
            .unwrap()
            .insert(job_id, "alice".to_string());

        let state = OrchestratorState {
            llm: Arc::new(StubLlm::default()),
            job_manager: Arc::new(jm),
            token_store,
            job_event_tx: None,
            prompt_queue: Arc::new(Mutex::new(HashMap::new())),
            store: None,
            secrets_store: Some(secrets_store),
            job_owner_cache,
            pending_gates: None,
            acp_permissions: Arc::new(AcpPermissionStore::new()),
        };

        let router = OrchestratorApi::router(state);
        let req = Request::builder()
            .uri(format!("/worker/{}/credentials", job_id))
            .header("Authorization", format!("Bearer {}", token))
            .body(Body::empty())
            .unwrap();

        let resp = router.oneshot(req).await.unwrap();
        // Alice has no "api_key" secret -- handler must refuse (403) rather
        // than serve bob's value. All secret-lookup failures map to FORBIDDEN
        // so the response does not leak whether a secret exists under any
        // other user's scope.
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    // -- Job event handler tests --

    #[tokio::test]
    async fn job_event_broadcasts_message() {
        let (tx, mut rx) = broadcast::channel(16);
        let token_store = TokenStore::new();
        let jm = ContainerJobManager::new(ContainerJobConfig::default(), token_store.clone());
        let state = OrchestratorState {
            llm: Arc::new(StubLlm::default()),
            job_manager: Arc::new(jm),
            token_store: token_store.clone(),
            job_event_tx: Some(tx),
            prompt_queue: Arc::new(Mutex::new(HashMap::new())),
            store: None,
            secrets_store: None,
            job_owner_cache: Arc::new(std::sync::RwLock::new(HashMap::new())),
            pending_gates: None,
            acp_permissions: Arc::new(AcpPermissionStore::new()),
        };

        let job_id = Uuid::new_v4();
        let token = token_store.create_token(job_id).await;
        let router = OrchestratorApi::router(state);

        let payload = serde_json::json!({
            "event_type": "message",
            "data": {
                "role": "assistant",
                "content": "Hello from worker"
            }
        });

        let req = Request::builder()
            .method("POST")
            .uri(format!("/worker/{}/event", job_id))
            .header("Authorization", format!("Bearer {}", token))
            .header("Content-Type", "application/json")
            .body(Body::from(serde_json::to_vec(&payload).unwrap()))
            .unwrap();

        let resp = router.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let (recv_id, recv_uid, event) = rx.recv().await.unwrap();
        assert_eq!(recv_id, job_id);
        // No store configured, so user_id falls back to empty string.
        assert_eq!(recv_uid, "");
        match event {
            AppEvent::JobMessage {
                job_id: jid,
                role,
                content,
            } => {
                assert_eq!(jid, job_id.to_string());
                assert_eq!(role, "assistant");
                assert_eq!(content, "Hello from worker");
            }
            other => panic!("Expected JobMessage, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn job_event_handles_tool_use() {
        let (tx, mut rx) = broadcast::channel(16);
        let token_store = TokenStore::new();
        let jm = ContainerJobManager::new(ContainerJobConfig::default(), token_store.clone());
        let state = OrchestratorState {
            llm: Arc::new(StubLlm::default()),
            job_manager: Arc::new(jm),
            token_store: token_store.clone(),
            job_event_tx: Some(tx),
            prompt_queue: Arc::new(Mutex::new(HashMap::new())),
            store: None,
            secrets_store: None,
            job_owner_cache: Arc::new(std::sync::RwLock::new(HashMap::new())),
            pending_gates: None,
            acp_permissions: Arc::new(AcpPermissionStore::new()),
        };

        let job_id = Uuid::new_v4();
        let token = token_store.create_token(job_id).await;
        let router = OrchestratorApi::router(state);

        let payload = serde_json::json!({
            "event_type": "tool_use",
            "data": {
                "tool_name": "shell",
                "input": {"command": "ls"}
            }
        });

        let req = Request::builder()
            .method("POST")
            .uri(format!("/worker/{}/event", job_id))
            .header("Authorization", format!("Bearer {}", token))
            .header("Content-Type", "application/json")
            .body(Body::from(serde_json::to_vec(&payload).unwrap()))
            .unwrap();

        let resp = router.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let (_recv_id, _recv_uid, event) = rx.recv().await.unwrap();
        match event {
            AppEvent::JobToolUse { tool_name, .. } => {
                assert_eq!(tool_name, "shell");
            }
            other => panic!("Expected JobToolUse, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn job_event_handles_unknown_type() {
        let (tx, mut rx) = broadcast::channel(16);
        let token_store = TokenStore::new();
        let jm = ContainerJobManager::new(ContainerJobConfig::default(), token_store.clone());
        let state = OrchestratorState {
            llm: Arc::new(StubLlm::default()),
            job_manager: Arc::new(jm),
            token_store: token_store.clone(),
            job_event_tx: Some(tx),
            prompt_queue: Arc::new(Mutex::new(HashMap::new())),
            store: None,
            secrets_store: None,
            job_owner_cache: Arc::new(std::sync::RwLock::new(HashMap::new())),
            pending_gates: None,
            acp_permissions: Arc::new(AcpPermissionStore::new()),
        };

        let job_id = Uuid::new_v4();
        let token = token_store.create_token(job_id).await;
        let router = OrchestratorApi::router(state);

        let payload = serde_json::json!({
            "event_type": "custom_thing",
            "data": { "message": "something custom" }
        });

        let req = Request::builder()
            .method("POST")
            .uri(format!("/worker/{}/event", job_id))
            .header("Authorization", format!("Bearer {}", token))
            .header("Content-Type", "application/json")
            .body(Body::from(serde_json::to_vec(&payload).unwrap()))
            .unwrap();

        let resp = router.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let (_recv_id, _recv_uid, event) = rx.recv().await.unwrap();
        // Unknown event types fall through to JobStatus
        assert!(matches!(event, AppEvent::JobStatus { .. }));
    }

    // -- ACP permission handler tests --

    use crate::worker::api::{PermissionDecision, PermissionOptionDto, PermissionOptionKindDto};

    fn sample_register_payload() -> PermissionRequestRegister {
        PermissionRequestRegister {
            permission_id: Uuid::new_v4(),
            tool_call_id: "tc-1".to_string(),
            tool_name: "shell".to_string(),
            tool_kind: Some("execute".to_string()),
            raw_input: Some(serde_json::json!({"cmd": "ls"})),
            locations: vec![],
            options: vec![
                PermissionOptionDto {
                    option_id: "allow-once".into(),
                    name: "Allow once".into(),
                    kind: PermissionOptionKindDto::AllowOnce,
                },
                PermissionOptionDto {
                    option_id: "allow-always".into(),
                    name: "Allow always".into(),
                    kind: PermissionOptionKindDto::AllowAlways,
                },
                PermissionOptionDto {
                    option_id: "reject".into(),
                    name: "Reject".into(),
                    kind: PermissionOptionKindDto::RejectOnce,
                },
            ],
        }
    }

    fn state_with_acp_surface(user_id: &str, job_id: Uuid) -> OrchestratorState {
        let token_store = TokenStore::new();
        let jm = ContainerJobManager::new(ContainerJobConfig::default(), token_store.clone());
        let (tx, _) = broadcast::channel(16);
        let job_owner_cache = Arc::new(std::sync::RwLock::new(HashMap::new()));
        job_owner_cache
            .write()
            .unwrap()
            .insert(job_id, user_id.to_string());
        OrchestratorState {
            llm: Arc::new(StubLlm::default()),
            job_manager: Arc::new(jm),
            token_store,
            job_event_tx: Some(tx),
            prompt_queue: Arc::new(Mutex::new(HashMap::new())),
            store: None,
            secrets_store: None,
            job_owner_cache,
            pending_gates: Some(Arc::new(PendingGateStore::in_memory())),
            acp_permissions: Arc::new(AcpPermissionStore::new()),
        }
    }

    #[tokio::test]
    async fn register_permission_returns_503_when_gate_store_missing() {
        // `register_permission_handler` falls back to the engine's global
        // `PendingGateStore` when `state.pending_gates` is None. Acquire the
        // shared test lock + clear engine state so a concurrent test can't
        // leave a store installed and turn our 503 into a 202.
        let _lock = crate::bridge::test_support::ENGINE_STATE_TEST_LOCK
            .lock()
            .await;
        crate::bridge::test_support::clear_engine_state().await;

        let job_id = Uuid::new_v4();
        let state = {
            let mut s = state_with_acp_surface("user-1", job_id);
            s.pending_gates = None;
            s
        };
        let token = state.token_store.create_token(job_id).await;
        let router = OrchestratorApi::router(state);

        let req = Request::builder()
            .method("POST")
            .uri(format!("/worker/{}/permission", job_id))
            .header("Authorization", format!("Bearer {}", token))
            .header("Content-Type", "application/json")
            .body(Body::from(
                serde_json::to_vec(&sample_register_payload()).unwrap(),
            ))
            .unwrap();
        let resp = router.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn register_permission_returns_403_when_owner_unknown() {
        let token_store = TokenStore::new();
        let jm = ContainerJobManager::new(ContainerJobConfig::default(), token_store.clone());
        let (tx, _) = broadcast::channel(16);
        let state = OrchestratorState {
            llm: Arc::new(StubLlm::default()),
            job_manager: Arc::new(jm),
            token_store: token_store.clone(),
            job_event_tx: Some(tx),
            prompt_queue: Arc::new(Mutex::new(HashMap::new())),
            store: None,
            secrets_store: None,
            // No owner cache entry → unknown owner → 403.
            job_owner_cache: Arc::new(std::sync::RwLock::new(HashMap::new())),
            pending_gates: Some(Arc::new(PendingGateStore::in_memory())),
            acp_permissions: Arc::new(AcpPermissionStore::new()),
        };
        let job_id = Uuid::new_v4();
        let token = token_store.create_token(job_id).await;
        let router = OrchestratorApi::router(state);

        let req = Request::builder()
            .method("POST")
            .uri(format!("/worker/{}/permission", job_id))
            .header("Authorization", format!("Bearer {}", token))
            .header("Content-Type", "application/json")
            .body(Body::from(
                serde_json::to_vec(&sample_register_payload()).unwrap(),
            ))
            .unwrap();
        let resp = router.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn register_permission_inserts_gate_and_broadcasts() {
        let job_id = Uuid::new_v4();
        let user_id = "user-1";
        let state = state_with_acp_surface(user_id, job_id);
        let mut rx = state.job_event_tx.as_ref().unwrap().subscribe();
        let pending_gates = Arc::clone(state.pending_gates.as_ref().unwrap());
        let acp_permissions = Arc::clone(&state.acp_permissions);
        let token = state.token_store.create_token(job_id).await;
        let payload = sample_register_payload();
        let router = OrchestratorApi::router(state);

        let req = Request::builder()
            .method("POST")
            .uri(format!("/worker/{}/permission", job_id))
            .header("Authorization", format!("Bearer {}", token))
            .header("Content-Type", "application/json")
            .body(Body::from(serde_json::to_vec(&payload).unwrap()))
            .unwrap();
        let resp = router.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::ACCEPTED);

        // PendingGate was inserted.
        let gates = pending_gates.list_for_user(user_id).await;
        assert_eq!(gates.len(), 1);
        assert_eq!(gates[0].gate_name, crate::bridge::ACP_PERMISSION_GATE_NAME);
        assert_eq!(gates[0].action_name, "shell");
        // allow_always must propagate because AllowAlways option was present.
        match &gates[0].resume_kind {
            ironclaw_engine::ResumeKind::Approval { allow_always } => {
                assert!(*allow_always);
            }
            other => panic!("expected Approval ResumeKind, got {other:?}"),
        }

        // Broadcast fired.
        let (recv_id, recv_uid, ev) = rx.recv().await.unwrap();
        assert_eq!(recv_id, job_id);
        assert_eq!(recv_uid, user_id);
        assert!(matches!(ev, AppEvent::GateRequired { .. }));

        // Slot registered for bridge polling.
        let opts = acp_permissions
            .options_for_request_id(gates[0].request_id)
            .await
            .unwrap();
        assert_eq!(opts.len(), 3);
    }

    #[tokio::test]
    async fn poll_permission_returns_404_for_unknown_id() {
        let job_id = Uuid::new_v4();
        let state = state_with_acp_surface("user-1", job_id);
        let token = state.token_store.create_token(job_id).await;
        let router = OrchestratorApi::router(state);

        let req = Request::builder()
            .uri(format!("/worker/{}/permission/{}", job_id, Uuid::new_v4()))
            .header("Authorization", format!("Bearer {}", token))
            .body(Body::empty())
            .unwrap();
        let resp = router.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn poll_permission_returns_decision_when_completed() {
        let job_id = Uuid::new_v4();
        let state = state_with_acp_surface("user-1", job_id);
        let token = state.token_store.create_token(job_id).await;

        // Pre-register + complete a slot so the handler returns immediately.
        let permission_id = Uuid::new_v4();
        let request_id = Uuid::new_v4();
        state
            .acp_permissions
            .register(
                job_id,
                permission_id,
                request_id,
                vec![],
                std::time::Duration::from_secs(60),
            )
            .await
            .expect("register");
        assert!(
            state
                .acp_permissions
                .complete_by_request_id(
                    request_id,
                    PermissionDecision::Selected {
                        option_id: "allow-once".to_string(),
                    },
                )
                .await
        );

        let router = OrchestratorApi::router(state);
        let req = Request::builder()
            .uri(format!("/worker/{}/permission/{}", job_id, permission_id))
            .header("Authorization", format!("Bearer {}", token))
            .body(Body::empty())
            .unwrap();
        let resp = router.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["outcome"], "selected");
        assert_eq!(json["option_id"], "allow-once");
    }

    // -- Status update test --

    #[tokio::test]
    async fn report_status_updates_handle() {
        let state = test_state();
        let job_id = Uuid::new_v4();
        let token = state.token_store.create_token(job_id).await;

        // Insert a handle so update_worker_status has something to update
        {
            let mut containers = state.job_manager.containers.write().await;
            containers.insert(
                job_id,
                crate::orchestrator::job_manager::ContainerHandle {
                    job_id,
                    container_id: "test-container".to_string(),
                    state: crate::orchestrator::job_manager::ContainerState::Running,
                    mode: crate::orchestrator::job_manager::JobMode::Worker,
                    created_at: chrono::Utc::now(),
                    project_dir: None,
                    task_description: "test".to_string(),
                    last_worker_status: None,
                    worker_iteration: 0,
                    completion_result: None,
                },
            );
        }

        let jm = Arc::clone(&state.job_manager);
        let router = OrchestratorApi::router(state);

        let update = serde_json::json!({
            "state": "in_progress",
            "message": "Iteration 5",
            "iteration": 5
        });

        let req = Request::builder()
            .method("POST")
            .uri(format!("/worker/{}/status", job_id))
            .header("Authorization", format!("Bearer {}", token))
            .header("Content-Type", "application/json")
            .body(Body::from(serde_json::to_vec(&update).unwrap()))
            .unwrap();

        let resp = router.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let handle = jm.get_handle(job_id).await.unwrap();
        assert_eq!(handle.worker_iteration, 5);
        assert_eq!(handle.last_worker_status.as_deref(), Some("Iteration 5"));
    }
}
