"""E2E scenario: ACP permission prompts render on the Jobs tab.

Exercises the full surface shipped for the ACP permission Jobs-tab
feature:

1. Trigger an ACP sandbox job. The mock LLM answers a "run test acp
   job" user message with a ``create_job`` tool call (mode=acp,
   agent=test-agent), which spins up a container running the
   ``test_acp_agent`` binary.
2. The test agent's first ``session/prompt`` unconditionally calls
   ``session/request_permission``. With ``surface_permissions=true``
   on the agent config, the IronClaw host raises a ``PendingGate``
   instead of auto-approving inside the container.
3. The gate carries the originating job_id (via
   ``PendingGate.job_id``). ``GET /api/jobs/pending-gates`` surfaces
   it. The Jobs tab renders a ``.job-pending-badge`` on the row and
   an approval card inside ``#job-pending-gates`` on the Activity
   subtab.
4. Clicking Approve routes through ``/api/chat/gate/resolve``; the
   gate clears from the list.

Skips cleanly when Docker or the ``ironclaw-test-acp:latest`` image
isn't available locally — see the ``acp_e2e_server`` fixture for the
build commands.
"""

import asyncio

import pytest

from helpers import AUTH_TOKEN, SEL, api_get, api_post


pytestmark = pytest.mark.timeout(180)


async def _open_tab(page, tab: str) -> None:
    button = page.locator(SEL["tab_button"].format(tab=tab))
    await button.click()
    await page.locator(SEL["tab_panel"].format(tab=tab)).wait_for(
        state="visible", timeout=5000
    )


async def _trigger_acp_job(base_url: str) -> str:
    """Send the mock-LLM trigger message and return the created job_id.

    The mock LLM's canned ``create_job`` response for ``run test acp
    job`` kicks off a sandbox job in ACP mode. We poll
    ``/api/jobs`` rather than waiting for the chat to finish because
    ``wait=false`` makes the tool return immediately with the job id.
    """
    thread = await api_post(base_url, "/api/chat/thread/new")
    thread.raise_for_status()
    thread_id = thread.json()["id"]

    send = await api_post(
        base_url,
        "/api/chat/send",
        json={"content": "run test acp job", "thread_id": thread_id},
        timeout=30,
    )
    assert send.status_code in (200, 202), send.text[:400]

    for _ in range(60):
        jobs = await api_get(base_url, "/api/jobs")
        jobs.raise_for_status()
        entries = jobs.json().get("jobs", [])
        if entries:
            return entries[0]["id"]
        await asyncio.sleep(0.5)
    raise AssertionError("ACP job was not created within 30s")


async def _wait_for_pending_gate(base_url: str, job_id: str, timeout: float = 30.0):
    """Poll the gate list until the orchestrator has registered our ACP gate."""
    for _ in range(int(timeout * 2)):
        resp = await api_get(base_url, "/api/jobs/pending-gates")
        resp.raise_for_status()
        gates = resp.json().get("gates", [])
        for gate in gates:
            if gate["job_id"] == job_id:
                return gate
        await asyncio.sleep(0.5)
    raise AssertionError(
        f"no pending gate registered for job {job_id} within {timeout}s"
    )


async def test_acp_permission_renders_and_approves_on_jobs_tab(acp_e2e_page, acp_e2e_server):
    job_id = await _trigger_acp_job(acp_e2e_server)
    gate = await _wait_for_pending_gate(acp_e2e_server, job_id)
    assert gate["gate_name"] == "acp_permission"

    # Jobs tab shows the attention badge on the row.
    await _open_tab(acp_e2e_page, "jobs")
    badge = acp_e2e_page.locator(SEL["job_pending_badge"])
    await badge.first.wait_for(state="visible", timeout=10000)

    # Click the row → Activity subtab should already be the default.
    row = acp_e2e_page.locator(SEL["job_row"]).first
    await row.click()

    # Approval card renders inside the job detail.
    card = acp_e2e_page.locator(SEL["job_pending_gate_card"])
    await card.wait_for(state="visible", timeout=10000)
    header_text = await card.locator(".approval-tool-name").inner_text()
    assert header_text, "approval card should show a tool name"

    # Approve. The card either removes itself after the resolved delay
    # or refetches empty; either way the gate list should clear.
    await acp_e2e_page.locator(SEL["job_pending_approve"]).click()

    for _ in range(20):
        resp = await api_get(acp_e2e_server, "/api/jobs/pending-gates")
        resp.raise_for_status()
        remaining = [
            g for g in resp.json().get("gates", []) if g["job_id"] == job_id
        ]
        if not remaining:
            break
        await asyncio.sleep(0.5)
    else:
        pytest.fail(f"gate for job {job_id} was not cleared after Approve click")


async def test_acp_permission_deny_clears_gate(acp_e2e_page, acp_e2e_server):
    job_id = await _trigger_acp_job(acp_e2e_server)
    await _wait_for_pending_gate(acp_e2e_server, job_id)

    await _open_tab(acp_e2e_page, "jobs")
    await acp_e2e_page.locator(SEL["job_pending_badge"]).first.wait_for(
        state="visible", timeout=10000
    )
    row = acp_e2e_page.locator(SEL["job_row"]).first
    await row.click()
    await acp_e2e_page.locator(SEL["job_pending_gate_card"]).wait_for(
        state="visible", timeout=10000
    )
    await acp_e2e_page.locator(SEL["job_pending_deny"]).click()

    for _ in range(20):
        resp = await api_get(acp_e2e_server, "/api/jobs/pending-gates")
        resp.raise_for_status()
        remaining = [
            g for g in resp.json().get("gates", []) if g["job_id"] == job_id
        ]
        if not remaining:
            return
        await asyncio.sleep(0.5)
    pytest.fail(f"gate for job {job_id} was not cleared after Deny click")
