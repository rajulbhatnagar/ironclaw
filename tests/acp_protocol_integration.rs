//! End-to-end protocol test for the ACP permission round-trip.
//!
//! Wires a minimal in-process ACP agent (mirrors the `simple_agent`
//! example in the upstream `agent-client-protocol` crate) to the real
//! IronClaw ACP client via two `tokio::io::duplex` pipes — no subprocess,
//! no HTTP, but the full ACP JSON-RPC protocol runs on both sides:
//!
//!     initialize
//!     session/new
//!     session/prompt
//!         ↓ agent calls back:
//!         session/request_permission    ← ACP wire format
//!           ↓ IronClawAcpClient registers with AcpPermissionStore
//!           ↓ simulated user resolves the slot
//!           ↑ client returns Selected(option_id) / Cancelled
//!         ↑ agent receives outcome over ACP wire format
//!     ← prompt response with StopReason
//!
//! Exercised surface:
//! - `IronClawAcpClient::request_permission` — full flow including
//!   register → long-poll → map decision → ACP `RequestPermissionResponse`.
//! - `AcpPermissionStore::register` + `complete_by_request_id` as the
//!   correlation layer.
//! - Real serde encoding/decoding of `RequestPermissionRequest` and
//!   `RequestPermissionResponse` over a pipe (catches any wire-shape
//!   drift between what the bridge sends and what an ACP agent expects).

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Duration;

use agent_client_protocol::{self as acp, Agent as _};
use tokio::sync::Mutex as AsyncMutex;
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};
use uuid::Uuid;

use ironclaw::error::WorkerError;
use ironclaw::orchestrator::AcpPermissionStore;
use ironclaw::worker::acp_bridge::{
    AcpEventSink, AcpPermissionGateway, IronClawAcpClient, ironclaw_init_request,
};
use ironclaw::worker::api::{JobEventPayload, PermissionDecision, PermissionRequestRegister};

/// Minimal ACP agent: on prompt, calls back into the client via the real
/// protocol to request permission, and returns a StopReason derived from
/// the user's outcome. Trimmed clone of `examples/agent.rs` from the
/// upstream `agent-client-protocol` crate.
struct SimpleAcpAgent {
    /// Installed after the `AgentSideConnection` is constructed so the
    /// agent's `prompt` handler can call back to the client.
    conn: Rc<RefCell<Option<Rc<acp::AgentSideConnection>>>>,
    /// Captures the outcome the agent observed, for assertions.
    last_outcome: Rc<RefCell<Option<acp::RequestPermissionOutcome>>>,
}

#[async_trait::async_trait(?Send)]
impl acp::Agent for SimpleAcpAgent {
    async fn initialize(
        &self,
        _args: acp::InitializeRequest,
    ) -> acp::Result<acp::InitializeResponse> {
        Ok(acp::InitializeResponse::new(acp::ProtocolVersion::V1)
            .agent_info(acp::Implementation::new("simple-agent", "0.1.0").title("Simple Agent")))
    }

    async fn authenticate(
        &self,
        _args: acp::AuthenticateRequest,
    ) -> acp::Result<acp::AuthenticateResponse> {
        Ok(acp::AuthenticateResponse::default())
    }

    async fn new_session(
        &self,
        _args: acp::NewSessionRequest,
    ) -> acp::Result<acp::NewSessionResponse> {
        Ok(acp::NewSessionResponse::new("sess-1"))
    }

    async fn prompt(&self, args: acp::PromptRequest) -> acp::Result<acp::PromptResponse> {
        let conn = self
            .conn
            .borrow()
            .as_ref()
            .expect("connection installed before prompt arrived")
            .clone();
        let fields = acp::ToolCallUpdateFields::new()
            .title(Some("shell".to_string()))
            .kind(Some(acp::ToolKind::Execute));
        let req = acp::RequestPermissionRequest::new(
            args.session_id.clone(),
            acp::ToolCallUpdate::new("tc-1", fields),
            vec![
                acp::PermissionOption::new(
                    "allow-once",
                    "Allow once",
                    acp::PermissionOptionKind::AllowOnce,
                ),
                acp::PermissionOption::new(
                    "reject-once",
                    "Reject once",
                    acp::PermissionOptionKind::RejectOnce,
                ),
            ],
        );
        let resp =
            <acp::AgentSideConnection as acp::Client>::request_permission(&conn, req).await?;
        let stop = match &resp.outcome {
            acp::RequestPermissionOutcome::Selected(_) => acp::StopReason::EndTurn,
            acp::RequestPermissionOutcome::Cancelled => acp::StopReason::Cancelled,
            _ => acp::StopReason::EndTurn,
        };
        *self.last_outcome.borrow_mut() = Some(resp.outcome);
        Ok(acp::PromptResponse::new(stop))
    }

    async fn cancel(&self, _args: acp::CancelNotification) -> acp::Result<()> {
        Ok(())
    }
}

/// Gateway/sink that backs the client with a real `AcpPermissionStore`.
/// The bridge's production gateway (`Arc<WorkerHttpClient>`) wraps the
/// same logical operations over HTTP; this mock skips the HTTP layer and
/// talks to the store directly, so the test stays in-process.
struct StoreGateway {
    store: Arc<AcpPermissionStore>,
    job_id: Uuid,
    /// Tracks the `pending_request_id` each `permission_id` was registered
    /// with, so the simulated-user task can resolve the correct slot.
    request_ids: Arc<AsyncMutex<HashMap<Uuid, Uuid>>>,
}

impl AcpEventSink for StoreGateway {
    async fn emit_event(&self, _payload: &JobEventPayload) {}
}

impl AcpPermissionGateway for StoreGateway {
    async fn register_permission(
        &self,
        req: &PermissionRequestRegister,
    ) -> Result<(), WorkerError> {
        let pending_request_id = Uuid::new_v4();
        self.request_ids
            .lock()
            .await
            .insert(req.permission_id, pending_request_id);
        self.store
            .register(
                self.job_id,
                req.permission_id,
                pending_request_id,
                req.options.clone(),
                Duration::from_secs(60),
            )
            .await
            .map_err(|e| WorkerError::ExecutionFailed {
                reason: format!("register failed: {e}"),
            })
    }

    async fn poll_permission(
        &self,
        permission_id: Uuid,
    ) -> Result<Option<PermissionDecision>, WorkerError> {
        let Some(slot) = self.store.get_slot(self.job_id, permission_id).await else {
            return Ok(None);
        };
        // Bounded wait so a dropped decision fails loudly instead of
        // hanging the test runner.
        match tokio::time::timeout(Duration::from_secs(5), slot.wait()).await {
            Ok(decision) => Ok(Some(decision)),
            Err(_) => Ok(None),
        }
    }
}

async fn run_round_trip(user_decision: PermissionDecision) -> acp::RequestPermissionOutcome {
    let store = Arc::new(AcpPermissionStore::new());
    let job_id = Uuid::new_v4();
    let request_ids: Arc<AsyncMutex<HashMap<Uuid, Uuid>>> =
        Arc::new(AsyncMutex::new(HashMap::new()));

    // Two duplex pipes mirror the stdin/stdout split. Writes from the
    // agent end up at the client's read side and vice versa.
    let (agent_incoming, client_outgoing) = tokio::io::duplex(64 * 1024);
    let (client_incoming, agent_outgoing) = tokio::io::duplex(64 * 1024);

    let store_for_user = Arc::clone(&store);
    let request_ids_for_user = Arc::clone(&request_ids);

    // ACP SDK futures are `!Send`; both sides must live inside a
    // LocalSet.
    let local = tokio::task::LocalSet::new();
    local
        .run_until(async move {
            // --- Agent side ---
            let agent_conn_slot: Rc<RefCell<Option<Rc<acp::AgentSideConnection>>>> =
                Rc::new(RefCell::new(None));
            let last_outcome: Rc<RefCell<Option<acp::RequestPermissionOutcome>>> =
                Rc::new(RefCell::new(None));
            let agent = SimpleAcpAgent {
                conn: Rc::clone(&agent_conn_slot),
                last_outcome: Rc::clone(&last_outcome),
            };
            let (agent_conn, agent_io) = acp::AgentSideConnection::new(
                agent,
                agent_outgoing.compat_write(),
                agent_incoming.compat(),
                |fut| {
                    tokio::task::spawn_local(fut);
                },
            );
            let agent_conn = Rc::new(agent_conn);
            *agent_conn_slot.borrow_mut() = Some(Rc::clone(&agent_conn));
            tokio::task::spawn_local(async move {
                let _ = agent_io.await;
            });

            // --- Client side (IronClaw) ---
            let gateway = StoreGateway {
                store: Arc::clone(&store),
                job_id,
                request_ids: Arc::clone(&request_ids),
            };
            let client = IronClawAcpClient::new(gateway, /* surface_permissions */ true);
            let (client_conn, client_io) = acp::ClientSideConnection::new(
                client,
                client_outgoing.compat_write(),
                client_incoming.compat(),
                |fut| {
                    tokio::task::spawn_local(fut);
                },
            );
            tokio::task::spawn_local(async move {
                let _ = client_io.await;
            });

            // Simulated user: wait for the client to register a permission,
            // then resolve the slot. In production this is the user
            // clicking in the web UI, which triggers
            // `chat_gate_resolve_handler` → `complete_by_request_id`.
            let user_task = tokio::task::spawn_local(async move {
                let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
                loop {
                    let entry = request_ids_for_user
                        .lock()
                        .await
                        .iter()
                        .next()
                        .map(|(k, v)| (*k, *v));
                    if let Some((_permission_id, request_id)) = entry {
                        let ok = store_for_user
                            .complete_by_request_id(request_id, user_decision.clone())
                            .await;
                        assert!(ok, "user resolution should wake the slot");
                        return;
                    }
                    if tokio::time::Instant::now() >= deadline {
                        panic!("user task: no permission registered within deadline");
                    }
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            });

            client_conn
                .initialize(ironclaw_init_request())
                .await
                .expect("initialize");
            let session = client_conn
                .new_session(acp::NewSessionRequest::new(std::path::PathBuf::from(
                    "/tmp",
                )))
                .await
                .expect("new_session");
            let _prompt_resp = client_conn
                .prompt(acp::PromptRequest::new(
                    session.session_id,
                    vec!["please run shell".into()],
                ))
                .await
                .expect("prompt");
            user_task.await.expect("user task");

            last_outcome
                .borrow()
                .clone()
                .expect("agent observed permission outcome")
        })
        .await
}

#[tokio::test]
async fn acp_e2e_user_approves_ends_turn() {
    let outcome = run_round_trip(PermissionDecision::Selected {
        option_id: "allow-once".to_string(),
    })
    .await;
    match outcome {
        acp::RequestPermissionOutcome::Selected(sel) => {
            assert_eq!(sel.option_id.to_string(), "allow-once");
        }
        other => panic!("expected Selected(allow-once), got {other:?}"),
    }
}

#[tokio::test]
async fn acp_e2e_user_cancels_cancels_turn() {
    let outcome = run_round_trip(PermissionDecision::Cancelled).await;
    assert!(
        matches!(outcome, acp::RequestPermissionOutcome::Cancelled),
        "expected Cancelled, got {outcome:?}",
    );
}
