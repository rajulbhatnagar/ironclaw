//! In-orchestrator correlation store for ACP permission requests.
//!
//! The ACP bridge (running in a Docker container) makes a synchronous
//! `request_permission` call to the ACP agent and cannot return until it
//! has a decision. The bridge registers the request here via
//! `POST /worker/{job_id}/permission`, then long-polls
//! `GET /worker/{job_id}/permission/{permission_id}` until a decision is
//! available.
//!
//! The user-facing path runs through the existing `PendingGateStore` +
//! `/api/chat/gate/resolve`. When the user resolves, the web handler calls
//! `AcpPermissionStore::complete_by_request_id` with the mapped
//! `PermissionDecision`, which wakes the long-poller.
//!
//! This module owns **only** the transport-level correlation (options offered
//! by the agent, waker, decision). The durable source of truth for what the
//! user sees is the `PendingGate` in `PendingGateStore`.
//!
//! Keys used:
//! - `(job_id, permission_id)` — the bridge's correlation id.
//! - `pending_request_id` (from the `PendingGate`) — how the web resolve
//!   handler completes a slot without needing the bridge's id.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use tokio::sync::{Mutex, Notify};
use uuid::Uuid;

use crate::worker::api::{PermissionDecision, PermissionOptionDto};

/// Sentinel key type so both indexes can share the same underlying slot.
type SlotKey = (Uuid, Uuid);

/// Reason a `register` call was rejected.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum DuplicateRegistration {
    #[error("slot already exists for this (job_id, permission_id)")]
    SlotKey,
    #[error("slot already exists for this pending_request_id")]
    RequestId,
}

/// Internal state for a single pending permission request.
pub struct Slot {
    options: Vec<PermissionOptionDto>,
    notify: Notify,
    decision: OnceLock<PermissionDecision>,
    expires_at: Instant,
}

impl Slot {
    /// Wait for the decision to be set.
    ///
    /// Returns `Some(decision)` once set. The caller is responsible for any
    /// surrounding timeout (the orchestrator's long-poll handler wraps this
    /// in a `tokio::select!` against a poll window).
    pub async fn wait(&self) -> PermissionDecision {
        loop {
            if let Some(d) = self.decision.get() {
                return d.clone();
            }
            self.notify.notified().await;
        }
    }

    /// Snapshot the options originally offered by the agent.
    pub fn options(&self) -> Vec<PermissionOptionDto> {
        self.options.clone()
    }
}

/// Correlation store for in-flight ACP permission requests.
pub struct AcpPermissionStore {
    inner: Mutex<Inner>,
}

struct Inner {
    slots: HashMap<SlotKey, Arc<Slot>>,
    by_request_id: HashMap<Uuid, SlotKey>,
}

impl AcpPermissionStore {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(Inner {
                slots: HashMap::new(),
                by_request_id: HashMap::new(),
            }),
        }
    }

    /// Register a new permission request.
    ///
    /// Must be called **before** inserting the corresponding `PendingGate`,
    /// so a fast-resolving user can't wake a slot that doesn't exist yet.
    ///
    /// Returns `Err` if a slot already exists for either
    /// `(job_id, permission_id)` or `pending_request_id` — the bridge
    /// generates a fresh v4 UUID per request, so a collision means the
    /// caller accidentally reused an id and would silently orphan an
    /// existing waiter.
    pub async fn register(
        &self,
        job_id: Uuid,
        permission_id: Uuid,
        pending_request_id: Uuid,
        options: Vec<PermissionOptionDto>,
        ttl: Duration,
    ) -> Result<(), DuplicateRegistration> {
        let slot = Arc::new(Slot {
            options,
            notify: Notify::new(),
            decision: OnceLock::new(),
            expires_at: Instant::now() + ttl,
        });

        let mut inner = self.inner.lock().await;
        let key = (job_id, permission_id);
        if inner.slots.contains_key(&key) {
            return Err(DuplicateRegistration::SlotKey);
        }
        if inner.by_request_id.contains_key(&pending_request_id) {
            return Err(DuplicateRegistration::RequestId);
        }
        inner.slots.insert(key, slot);
        inner.by_request_id.insert(pending_request_id, key);
        Ok(())
    }

    /// Look up a slot by its `(job_id, permission_id)` correlation id.
    pub async fn get_slot(&self, job_id: Uuid, permission_id: Uuid) -> Option<Arc<Slot>> {
        let inner = self.inner.lock().await;
        inner.slots.get(&(job_id, permission_id)).cloned()
    }

    /// Snapshot the options originally registered for a gate request id.
    ///
    /// Used by the web resolve handler to pick the right `option_id` when
    /// mapping `GateResolutionPayload::Approved{always}` onto the ACP
    /// `PermissionDecision::Selected` shape.
    pub async fn options_for_request_id(
        &self,
        pending_request_id: Uuid,
    ) -> Option<Vec<PermissionOptionDto>> {
        let inner = self.inner.lock().await;
        let key = inner.by_request_id.get(&pending_request_id)?;
        inner.slots.get(key).map(|s| s.options.clone())
    }

    /// Mark a permission request as resolved with the given decision.
    ///
    /// Returns `true` if a slot was found and the decision was recorded,
    /// `false` if the request id is unknown (stale, expired, or already
    /// consumed). Wakes any waiting long-pollers.
    pub async fn complete_by_request_id(
        &self,
        pending_request_id: Uuid,
        decision: PermissionDecision,
    ) -> bool {
        let mut inner = self.inner.lock().await;
        let Some(key) = inner.by_request_id.remove(&pending_request_id) else {
            return false;
        };
        let Some(slot) = inner.slots.get(&key).cloned() else {
            return false;
        };
        let set_ok = slot.decision.set(decision).is_ok();
        slot.notify.notify_waiters();
        set_ok
    }

    /// Remove stale entries whose expiry has passed.
    ///
    /// Returns the number of entries removed. Intended to be called from a
    /// periodic sweep task.
    pub async fn expire_stale(&self) -> usize {
        let now = Instant::now();
        let mut inner = self.inner.lock().await;
        let mut removed_keys: HashSet<SlotKey> = HashSet::new();
        inner.slots.retain(|key, slot| {
            if slot.expires_at <= now {
                removed_keys.insert(*key);
                false
            } else {
                true
            }
        });
        inner.by_request_id.retain(|_, v| !removed_keys.contains(v));
        removed_keys.len()
    }
}

impl Default for AcpPermissionStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::timeout;

    fn opts() -> Vec<PermissionOptionDto> {
        vec![PermissionOptionDto {
            option_id: "allow-once".to_string(),
            name: "Allow once".to_string(),
            kind: crate::worker::api::PermissionOptionKindDto::AllowOnce,
        }]
    }

    #[tokio::test]
    async fn register_and_complete_wakes_waiter() {
        let store = Arc::new(AcpPermissionStore::new());
        let job_id = Uuid::new_v4();
        let permission_id = Uuid::new_v4();
        let request_id = Uuid::new_v4();

        store
            .register(
                job_id,
                permission_id,
                request_id,
                opts(),
                Duration::from_secs(60),
            )
            .await
            .expect("register");

        let store2 = Arc::clone(&store);
        let waiter = tokio::spawn(async move {
            let slot = store2.get_slot(job_id, permission_id).await.expect("slot");
            slot.wait().await
        });

        tokio::time::sleep(Duration::from_millis(20)).await;
        let completed = store
            .complete_by_request_id(
                request_id,
                PermissionDecision::Selected {
                    option_id: "allow-once".to_string(),
                },
            )
            .await;
        assert!(completed);

        let decision = timeout(Duration::from_millis(500), waiter)
            .await
            .expect("wait timed out")
            .expect("waiter panicked");
        assert_eq!(
            decision,
            PermissionDecision::Selected {
                option_id: "allow-once".to_string()
            }
        );
    }

    #[tokio::test]
    async fn unknown_permission_id_returns_none() {
        let store = AcpPermissionStore::new();
        let res = store.get_slot(Uuid::new_v4(), Uuid::new_v4()).await;
        assert!(res.is_none());
    }

    #[tokio::test]
    async fn complete_unknown_request_id_returns_false() {
        let store = AcpPermissionStore::new();
        let completed = store
            .complete_by_request_id(Uuid::new_v4(), PermissionDecision::Cancelled)
            .await;
        assert!(!completed);
    }

    #[tokio::test]
    async fn complete_is_idempotent_after_first() {
        let store = AcpPermissionStore::new();
        let job_id = Uuid::new_v4();
        let permission_id = Uuid::new_v4();
        let request_id = Uuid::new_v4();

        store
            .register(
                job_id,
                permission_id,
                request_id,
                opts(),
                Duration::from_secs(60),
            )
            .await
            .expect("register");

        assert!(
            store
                .complete_by_request_id(request_id, PermissionDecision::Cancelled)
                .await
        );
        assert!(
            !store
                .complete_by_request_id(request_id, PermissionDecision::Cancelled)
                .await
        );
    }

    #[tokio::test]
    async fn options_for_request_id_returns_registered_options() {
        let store = AcpPermissionStore::new();
        let job_id = Uuid::new_v4();
        let permission_id = Uuid::new_v4();
        let request_id = Uuid::new_v4();

        store
            .register(
                job_id,
                permission_id,
                request_id,
                opts(),
                Duration::from_secs(60),
            )
            .await
            .expect("register");

        let fetched = store
            .options_for_request_id(request_id)
            .await
            .expect("opts");
        assert_eq!(fetched.len(), 1);
        assert_eq!(fetched[0].option_id, "allow-once");
    }

    #[tokio::test]
    async fn expire_stale_removes_expired_entries() {
        let store = AcpPermissionStore::new();
        let job_id = Uuid::new_v4();
        let permission_id = Uuid::new_v4();
        let request_id = Uuid::new_v4();

        store
            .register(
                job_id,
                permission_id,
                request_id,
                opts(),
                Duration::from_millis(1),
            )
            .await
            .expect("register");

        tokio::time::sleep(Duration::from_millis(10)).await;
        let removed = store.expire_stale().await;
        assert_eq!(removed, 1);
        assert!(store.get_slot(job_id, permission_id).await.is_none());
        assert!(store.options_for_request_id(request_id).await.is_none());
    }

    #[tokio::test]
    async fn register_rejects_duplicate_slot_key() {
        let store = AcpPermissionStore::new();
        let job_id = Uuid::new_v4();
        let permission_id = Uuid::new_v4();
        store
            .register(
                job_id,
                permission_id,
                Uuid::new_v4(),
                opts(),
                Duration::from_secs(60),
            )
            .await
            .expect("first register");
        let err = store
            .register(
                job_id,
                permission_id,
                Uuid::new_v4(),
                opts(),
                Duration::from_secs(60),
            )
            .await
            .unwrap_err();
        assert_eq!(err, DuplicateRegistration::SlotKey);
    }

    #[tokio::test]
    async fn register_rejects_duplicate_request_id() {
        let store = AcpPermissionStore::new();
        let request_id = Uuid::new_v4();
        store
            .register(
                Uuid::new_v4(),
                Uuid::new_v4(),
                request_id,
                opts(),
                Duration::from_secs(60),
            )
            .await
            .expect("first register");
        let err = store
            .register(
                Uuid::new_v4(),
                Uuid::new_v4(),
                request_id,
                opts(),
                Duration::from_secs(60),
            )
            .await
            .unwrap_err();
        assert_eq!(err, DuplicateRegistration::RequestId);
    }

    #[tokio::test]
    async fn concurrent_waiters_all_see_decision() {
        let store = Arc::new(AcpPermissionStore::new());
        let job_id = Uuid::new_v4();
        let permission_id = Uuid::new_v4();
        let request_id = Uuid::new_v4();

        store
            .register(
                job_id,
                permission_id,
                request_id,
                opts(),
                Duration::from_secs(60),
            )
            .await
            .expect("register");

        let make_waiter = || {
            let store = Arc::clone(&store);
            tokio::spawn(async move {
                let slot = store.get_slot(job_id, permission_id).await.expect("slot");
                slot.wait().await
            })
        };
        let w1 = make_waiter();
        let w2 = make_waiter();

        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(
            store
                .complete_by_request_id(
                    request_id,
                    PermissionDecision::Selected {
                        option_id: "allow-once".to_string()
                    }
                )
                .await
        );

        let d1 = timeout(Duration::from_millis(500), w1)
            .await
            .unwrap()
            .unwrap();
        let d2 = timeout(Duration::from_millis(500), w2)
            .await
            .unwrap()
            .unwrap();
        let expected = PermissionDecision::Selected {
            option_id: "allow-once".to_string(),
        };
        assert_eq!(d1, expected);
        assert_eq!(d2, expected);
    }
}
