//! Minimal ACP agent for e2e tests.
//!
//! Speaks ACP over stdio like `examples/agent` from the upstream
//! `agent-client-protocol` crate. On every `session/prompt` it calls
//! `session/request_permission` with two options. With the host's
//! `surface_permissions=true` flag this produces a real `PendingGate`
//! the Jobs-tab UI can render and resolve; the agent returns a
//! `StopReason` derived from the user's outcome.
//!
//! This is the test counterpart of `SimpleAcpAgent` in
//! `tests/acp_protocol_integration.rs`, packaged as a standalone
//! binary so it can run inside the container the orchestrator spawns
//! for ACP jobs.
//!
//! Run standalone (for local sanity checks):
//!
//! ```bash
//! cargo build -p test_acp_agent --release
//! ```

use std::cell::RefCell;
use std::rc::Rc;

use agent_client_protocol::{self as acp};
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};

struct TestAgent {
    conn: Rc<RefCell<Option<Rc<acp::AgentSideConnection>>>>,
}

#[async_trait::async_trait(?Send)]
impl acp::Agent for TestAgent {
    async fn initialize(
        &self,
        _args: acp::InitializeRequest,
    ) -> acp::Result<acp::InitializeResponse> {
        Ok(
            acp::InitializeResponse::new(acp::ProtocolVersion::V1).agent_info(
                acp::Implementation::new("test-acp-agent", env!("CARGO_PKG_VERSION"))
                    .title("IronClaw e2e Test Agent"),
            ),
        )
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
        Ok(acp::NewSessionResponse::new("test-session"))
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
            acp::ToolCallUpdate::new("tc-e2e", fields),
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
        let stop = match resp.outcome {
            acp::RequestPermissionOutcome::Selected(_) => acp::StopReason::EndTurn,
            acp::RequestPermissionOutcome::Cancelled => acp::StopReason::Cancelled,
            _ => acp::StopReason::EndTurn,
        };
        Ok(acp::PromptResponse::new(stop))
    }

    async fn cancel(&self, _args: acp::CancelNotification) -> acp::Result<()> {
        Ok(())
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> acp::Result<()> {
    let outgoing = tokio::io::stdout().compat_write();
    let incoming = tokio::io::stdin().compat();

    let local = tokio::task::LocalSet::new();
    local
        .run_until(async move {
            let conn_slot: Rc<RefCell<Option<Rc<acp::AgentSideConnection>>>> =
                Rc::new(RefCell::new(None));
            let agent = TestAgent {
                conn: Rc::clone(&conn_slot),
            };
            let (conn, handle_io) =
                acp::AgentSideConnection::new(agent, outgoing, incoming, |fut| {
                    tokio::task::spawn_local(fut);
                });
            *conn_slot.borrow_mut() = Some(Rc::new(conn));
            handle_io.await
        })
        .await
}
