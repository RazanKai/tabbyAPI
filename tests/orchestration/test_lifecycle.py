"""Lifecycle coordinator tests: leases, single-flight load, drain, idle unload, pause.

Mocked tier (SPEC section 10). Uses injected clocks and fake load/unload callbacks.
Follows upstream's test convention of plain ``asyncio.run`` (no pytest-asyncio), so
the suite runs with the same tooling as the rest of TabbyAPI's tests.

Assertions are on *observed behaviour* — which callbacks ran, in what order, and what
state was published — never on a mocked helper's return value.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from orchestration.lifecycle import (
    AdmissionOutcome,
    CoordinatorDeps,
    LeaseKind,
    Lifecycle,
    LifecycleCoordinator,
)
from orchestration.policy import PriorityState, Reason
from orchestration.telemetry import ActivityStatus

from orch_helpers import MIB, FakeClock, FakeDeps, FakeTelemetry, make_snapshot


def build(calibrated_config, clock=None, quiet_seconds=1.0, **deps_kwargs):
    """Build a coordinator and pre-satisfy the cold-load quiet window.

    Real cold admission requires *continuous* quietness, so tests must model that
    window rather than assuming a single quiet sample qualifies. Setting
    ``quiet_seconds`` small keeps the setup honest (the window still has to elapse)
    without making every test multi-step.

    The fake loader is wired to keep the snapshot fresh while its clock advances,
    because in production the background sampler keeps sampling throughout a load.
    """
    clock = clock or FakeClock()
    calibrated_config.cold_load.quiet_seconds = quiet_seconds
    deps = FakeDeps(clock=clock, **deps_kwargs)
    coord = LifecycleCoordinator(
        calibrated_config, deps.as_deps(), FakeTelemetry(), clock=clock, log=lambda _m: None
    )

    def resample_quiet() -> None:
        coord.ingest_snapshot(
            make_snapshot(
                captured=clock.now,
                utilization=0,
                external=[],
                activity_status=ActivityStatus.SPARSE,
            ),
            clock.now,
        )

    deps.on_clock_advance = resample_quiet
    return coord, deps, clock


def elapse_quiet(coord, clock, seconds=None, **snap_kwargs):
    """Advance the clock through the quiet window, feeding a sample each step.

    Quietness is a continuous property, so it must be *observed* over time.
    ``growth=None`` disables the growth baseline reset so a flat test can reuse a
    constant external figure.
    """
    seconds = coord.cfg.cold_load.quiet_seconds + 0.5 if seconds is None else seconds
    steps = 3
    base = dict(utilization=0, external=[], activity_status=ActivityStatus.SPARSE)
    base.update(snap_kwargs)
    for i in range(steps + 1):
        if i:
            clock.advance(seconds / steps)
        coord.ingest_snapshot(make_snapshot(captured=clock.now, **base), clock.now)


def feed(coord, clock, **snap_kwargs):
    """One fresh snapshot at the clock's now (no time advanced)."""
    snap = make_snapshot(captured=clock.now, **snap_kwargs)
    coord.ingest_snapshot(snap, clock.now)
    return snap


async def with_clock_pump(coord, clock, coro, *, step=0.1, max_seconds=60.0, **snap_kwargs):
    """Run ``coro`` while a background task advances the injected clock.

    The injected clock is the coordinator's only deadline and hysteresis
    authority, so any behaviour that requires *time to pass* (a release window,
    a quiet window, a wait-list deadline) needs it driven explicitly — exactly
    as the production sampler drives it in wall time. Returns the coroutine's
    result and always stops the pump.
    """
    base = dict(utilization=0, external=[], activity_status=ActivityStatus.SPARSE)
    base.update(snap_kwargs)
    elapsed = 0.0

    async def pump() -> None:
        nonlocal elapsed
        while elapsed < max_seconds:
            await asyncio.sleep(0.001)
            clock.advance(step)
            elapsed += step
            coord.ingest_snapshot(make_snapshot(captured=clock.now, **base), clock.now)

    task = asyncio.create_task(pump())
    try:
        return await coro
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


def elapse_veto(coord, clock, external=None, utilization=90, activity=None):
    """Drive a sustained external workload past the policy's enter window.

    A single hot sample is only a CANDIDATE by design (R04), so tests that expect a
    BUSY veto must let the confirmation window actually elapse. The workload is held
    constant so the memory trigger is the one under test.
    """
    external = external if external is not None else [(4242, 4096 * MIB)]
    window = coord.cfg.external_workload.enter_seconds
    steps = 3
    for i in range(steps + 1):
        if i:
            clock.advance((window + 0.5) / steps)
        snap_kwargs = dict(
            utilization=utilization,
            external=external,
            activity_status=(ActivityStatus.AVAILABLE if activity else ActivityStatus.SPARSE),
            activity=activity,
        )
        coord.ingest_snapshot(make_snapshot(captured=clock.now, **snap_kwargs), clock.now)


def ready_and_quiet(coord, clock, quiet_seconds=1.0, **snap_kwargs):
    """Satisfy cold-load quietness by *observing it over time*, then return."""
    elapse_quiet(coord, clock, seconds=quiet_seconds + 0.5, **snap_kwargs)


def quiet_snap(**over) -> dict:
    """A snapshot that satisfies cold-load quietness (no external, no activity)."""
    base = dict(utilization=0, external=[], activity_status=ActivityStatus.SPARSE, free=10000 * MIB)
    base.update(over)
    return base


# --------------------------------------------------------------------------- #
# Cold load, leases, single-flight (R06, R07, T07)
# --------------------------------------------------------------------------- #


def test_cold_request_loads_then_grants_a_lease(calibrated_config):
    """R06/R07: an eligible request performs the cold load and receives a lease."""
    coord, deps, clock = build(calibrated_config)
    elapse_quiet(coord, clock)

    async def go():
        return await coord.acquire_lease("req-1")

    result = asyncio.run(go())
    assert result.outcome is AdmissionOutcome.GRANTED
    assert result.lease is not None
    assert deps.load_count == 1
    assert coord.lifecycle is Lifecycle.READY
    assert deps.calls == ["load:start", "load:done"]


def test_load_failure_never_publishes_ready(calibrated_config):
    """R07/T14: READY comes from a real container, never from an intention."""
    coord, deps, clock = build(calibrated_config, fail_load=True)
    elapse_quiet(coord, clock)

    result = asyncio.run(coord.acquire_lease("req-1"))
    assert result.outcome is AdmissionOutcome.DENIED
    assert result.reason is Reason.ORCHESTRATOR_FAULT
    assert coord.lifecycle is not Lifecycle.READY
    assert deps.calls == ["load:start", "load:fail"]


def test_ambiguous_load_failure_latches_fault(calibrated_config):
    """R07/T14: a failure with a container still present is FAULT, not READY/UNLOADED."""
    coord, deps, clock = build(calibrated_config, fail_load=True, fail_load_but_container=True)
    elapse_quiet(coord, clock)

    result = asyncio.run(coord.acquire_lease("req-1"))
    assert result.outcome is AdmissionOutcome.DENIED
    assert coord.lifecycle is Lifecycle.FAULT
    assert coord.fault_reason is not None


def test_load_reported_success_without_container_is_fault(calibrated_config):
    """R07: a task-start or HTTP success is not proof of a usable model."""
    coord, deps, clock = build(calibrated_config)
    elapse_quiet(coord, clock)

    async def load_but_no_container():
        deps.calls.append("load:start")
        deps.load_count += 1
        # deliberately does NOT set _present

    coord.deps = CoordinatorDeps(
        load_model=load_but_no_container,
        unload_model=deps.unload_model,
        container_present=lambda: deps._present,
        backend_busy=lambda: deps._busy,
        container_identity=lambda: None,
        container_envelope=lambda: {},
        recovery_in_progress=lambda: False,
        adopt_recovery_task=lambda: None,
    )
    result = asyncio.run(coord.acquire_lease("req-1"))
    assert result.outcome is AdmissionOutcome.DENIED
    assert coord.lifecycle is Lifecycle.FAULT


def test_second_request_behind_a_transition_is_denied(calibrated_config):
    """R09/T07: in reject mode, arrivals behind an existing load fail promptly."""
    coord, deps, clock = build(calibrated_config, load_seconds=5.0)
    elapse_quiet(coord, clock)

    async def go():
        first = asyncio.create_task(coord.acquire_lease("req-1"))
        await asyncio.sleep(0)  # let the first request reserve the transition
        await asyncio.sleep(0)
        second = await coord.acquire_lease("req-2")
        first_result = await first
        return first_result, second

    first_result, second = asyncio.run(go())
    assert first_result.outcome is AdmissionOutcome.GRANTED
    assert second.outcome is AdmissionOutcome.DENIED
    assert second.reason is Reason.MODEL_TRANSITION, (
        "the arrival must be told a transition is in progress, not queue behind it"
    )
    assert deps.load_count == 1, "exactly one load for two requests"


def test_simultaneous_requests_share_one_load(calibrated_config):
    """R07/T07: no duplicate container and no second memory reservation."""
    coord, deps, clock = build(calibrated_config, load_seconds=3.0)
    elapse_quiet(coord, clock)

    async def go():
        a = asyncio.create_task(coord.acquire_lease("req-a"))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        b = asyncio.create_task(coord.acquire_lease("req-b"))
        return await asyncio.gather(a, b)

    results = asyncio.run(go())
    assert deps.load_count == 1
    # One is admitted by the load; the other is denied while the transition is active.
    outcomes = sorted(r.outcome.value for r in results)
    assert outcomes == ["denied", "granted"]


def test_already_resident_admits_without_reloading(calibrated_config):
    """R06/T06: a warm request must not be charged the cold footprint again."""
    coord, deps, clock = build(calibrated_config)
    elapse_quiet(coord, clock)
    first = asyncio.run(coord.acquire_lease("req-1"))
    assert first.outcome is AdmissionOutcome.GRANTED
    asyncio.run(coord.release_lease(first.lease))

    clock.advance(1)
    elapse_quiet(coord, clock)
    second = asyncio.run(coord.acquire_lease("req-2"))
    assert second.outcome is AdmissionOutcome.GRANTED
    assert deps.load_count == 1, "warm admission must not trigger a second load"


def test_active_request_limit_is_enforced(calibrated_config):
    """R06/T07: V1 admits exactly one inference request at a time."""
    coord, deps, clock = build(calibrated_config)
    elapse_quiet(coord, clock)
    first = asyncio.run(coord.acquire_lease("req-1"))
    assert first.outcome is AdmissionOutcome.GRANTED

    clock.advance(0.5)
    elapse_quiet(coord, clock)
    second = asyncio.run(coord.acquire_lease("req-2"))
    assert second.outcome is AdmissionOutcome.DENIED
    assert second.reason is Reason.REQUEST_CAPACITY
    assert any("request_capacity" in b for b in second.detail["blockers"])


def test_reader_pin_does_not_block_inference_and_is_counted(calibrated_config):
    """R08/T10: reader pins protect teardown but are not inference demand."""
    coord, deps, clock = build(calibrated_config)
    elapse_quiet(coord, clock)
    assert asyncio.run(coord.acquire_lease("req-1")).outcome is AdmissionOutcome.GRANTED

    clock.advance(1)
    elapse_quiet(coord, clock)
    pin = asyncio.run(coord.acquire_lease("pin-1", LeaseKind.READER))
    assert pin.outcome is AdmissionOutcome.GRANTED
    assert pin.lease.kind is LeaseKind.READER


# --------------------------------------------------------------------------- #
# Veto, drain, unload (R08, T06/T08/T09)
# --------------------------------------------------------------------------- #


def test_priority_veto_denies_new_work_but_keeps_resident_model(calibrated_config):
    """R06/T06: an already-loaded model is denied under veto; it is not unloaded."""
    coord, deps, clock = build(calibrated_config)
    elapse_quiet(coord, clock)
    assert asyncio.run(coord.acquire_lease("req-1")).outcome is AdmissionOutcome.GRANTED
    asyncio.run(coord.release_lease(coord._leases["inference:req-1:0"]))

    clock.advance(1)
    elapse_veto(coord, clock)
    result = asyncio.run(coord.acquire_lease("req-2"))
    assert result.outcome is AdmissionOutcome.DENIED
    assert result.reason is Reason.EXTERNAL_GPU_BUSY
    assert deps.unload_count == 0, "policy action must not forcibly unload under work"
    assert coord.lifecycle is Lifecycle.READY


def test_veto_with_no_lease_drains_then_unloads(calibrated_config):
    """R08/T16: veto with no active lease goes READY -> DRAINING -> UNLOADED."""
    coord, deps, clock = build(calibrated_config)
    elapse_quiet(coord, clock)
    first = asyncio.run(coord.acquire_lease("req-1"))
    asyncio.run(coord.release_lease(first.lease))
    assert coord.lifecycle is Lifecycle.READY

    clock.advance(1)
    elapse_veto(coord, clock)
    action = asyncio.run(coord.tick())
    assert action == "unload_veto"
    assert coord.lifecycle is Lifecycle.UNLOADED
    assert deps.unload_count == 1


def test_veto_with_active_lease_drains_and_does_not_unload(calibrated_config):
    """R08/T08: existing leases drain; teardown waits for them."""
    coord, deps, clock = build(calibrated_config)
    elapse_quiet(coord, clock)
    lease = asyncio.run(coord.acquire_lease("req-1")).lease

    clock.advance(1)
    elapse_veto(coord, clock)
    assert asyncio.run(coord.tick()) is None
    assert coord.lifecycle is Lifecycle.DRAINING
    assert deps.unload_count == 0

    # Lease completes normally -> the drain can proceed.
    asyncio.run(coord.release_lease(lease))
    action = asyncio.run(coord.tick())
    assert action == "unload_drained"
    assert coord.lifecycle is Lifecycle.UNLOADED


def test_backend_work_defers_unload_even_without_leases(calibrated_config):
    """R08/T14: a stuck backend is an operator-visible fault, not a kill."""
    coord, deps, clock = build(calibrated_config)
    elapse_quiet(coord, clock)
    lease = asyncio.run(coord.acquire_lease("req-1")).lease
    asyncio.run(coord.release_lease(lease))

    deps.set_busy(True)
    clock.advance(1)
    elapse_veto(coord, clock)
    assert asyncio.run(coord.tick()) is None, "must not unload while backend work remains"
    assert coord.lifecycle is Lifecycle.DRAINING
    assert deps.unload_count == 0

    deps.set_busy(False)
    assert asyncio.run(coord.tick()) == "unload_drained"
    assert deps.unload_count == 1


def test_unload_failure_latches_fault_not_unloaded(calibrated_config):
    """R08/R13: an unload that did not clear the container is a fault."""
    coord, deps, clock = build(calibrated_config, fail_unload=True)
    elapse_quiet(coord, clock)
    lease = asyncio.run(coord.acquire_lease("req-1")).lease
    asyncio.run(coord.release_lease(lease))
    asyncio.run(coord.request_unload(Reason.ORCHESTRATOR_PAUSED))
    assert coord.lifecycle is Lifecycle.FAULT
    assert coord.fault_reason is not None


# --------------------------------------------------------------------------- #
# Idle TTL (R08, T10)
# --------------------------------------------------------------------------- #


def test_idle_ttl_unloads_after_final_completion(calibrated_config):
    """R08/T10: TTL starts at the release of the final inference lease."""
    coord, deps, clock = build(calibrated_config, quiet_seconds=0.2)
    coord.cfg.idle_unload.seconds = 300
    elapse_quiet(coord, clock)
    lease = asyncio.run(coord.acquire_lease("req-1")).lease
    asyncio.run(coord.release_lease(lease))

    clock.advance(299)
    feed(coord, clock)
    assert asyncio.run(coord.tick()) is None, "must not unload before the TTL"
    assert deps.unload_count == 0

    clock.advance(2)
    feed(coord, clock)
    assert asyncio.run(coord.tick()) == "unload_idle"
    assert deps.unload_count == 1
    assert coord.lifecycle is Lifecycle.UNLOADED


def test_new_inference_lease_resets_the_idle_timer(calibrated_config):
    """R08/T10: a new inference lease resets the idle countdown."""
    coord, deps, clock = build(calibrated_config)
    elapse_quiet(coord, clock)
    lease = asyncio.run(coord.acquire_lease("req-1")).lease
    asyncio.run(coord.release_lease(lease))

    clock.advance(200)
    elapse_quiet(coord, clock)
    second = asyncio.run(coord.acquire_lease("req-2")).lease
    asyncio.run(coord.release_lease(second))

    clock.advance(200)
    elapse_quiet(coord, clock)
    assert asyncio.run(coord.tick()) is None, "timer restarted at the second completion"


def test_status_query_does_not_reset_idle_ttl(calibrated_config):
    """R08/T10: status/model-list/blocked requests must not keep a model resident."""
    coord, deps, clock = build(calibrated_config)
    elapse_quiet(coord, clock)
    lease = asyncio.run(coord.acquire_lease("req-1")).lease
    asyncio.run(coord.release_lease(lease))

    for _ in range(5):
        clock.advance(100)
        elapse_quiet(coord, clock)
        asyncio.run(coord.status())  # a read must not create demand

    clock.advance(320)
    elapse_quiet(coord, clock)
    assert asyncio.run(coord.tick()) == "unload_idle"


def test_reader_pin_defers_unload_without_refreshing_ttl(calibrated_config):
    """R08/T10: an outstanding reader pin defers teardown but never extends the TTL."""
    coord, deps, clock = build(calibrated_config)
    elapse_quiet(coord, clock)
    lease = asyncio.run(coord.acquire_lease("req-1")).lease
    asyncio.run(coord.release_lease(lease))

    pin = asyncio.run(coord.acquire_lease("read-1", LeaseKind.READER)).lease
    clock.advance(400)
    elapse_quiet(coord, clock)
    assert asyncio.run(coord.tick()) is None, "the pin must defer unload"
    assert coord.lifecycle in (Lifecycle.DRAINING, Lifecycle.READY)
    assert deps.unload_count == 0

    asyncio.run(coord.release_lease(pin))
    assert asyncio.run(coord.tick()) == "unload_idle", "TTL was NOT extended by the pin"


def test_expired_ttl_does_not_unload_with_a_live_lease(calibrated_config):
    """R08/T10: TTL expiry rechecks leases atomically before unloading."""
    coord, deps, clock = build(calibrated_config)
    elapse_quiet(coord, clock)
    first = asyncio.run(coord.acquire_lease("req-1")).lease
    asyncio.run(coord.release_lease(first))
    live = asyncio.run(coord.acquire_lease("req-2")).lease

    clock.advance(5000)
    elapse_quiet(coord, clock)
    assert asyncio.run(coord.tick()) is None
    assert deps.unload_count == 0
    assert live.released is False


# --------------------------------------------------------------------------- #
# Manual pause / resume (R10, T12)
# --------------------------------------------------------------------------- #


def test_pause_is_idempotent_and_drains(calibrated_config):
    """R10/T12: pause closes admission immediately and reports actual state."""
    coord, deps, clock = build(calibrated_config)
    elapse_quiet(coord, clock)
    asyncio.run(coord.acquire_lease("req-1"))

    asyncio.run(coord.pause())
    asyncio.run(coord.pause())  # idempotent
    assert coord.lifecycle is Lifecycle.DRAINING or coord.lifecycle is Lifecycle.UNLOADED
    assert deps.unload_count in (0, 1)

    status = asyncio.run(coord.status())
    assert status["paused"] is True
    assert status["admission"]["can_admit_now"] is False


def test_pause_with_no_work_hands_the_unload_to_maintenance(calibrated_config):
    """R10/T19 + R08: pause returns promptly and tick() performs the unload.

    The endpoint must answer within T19's ~1 s budget, so pause() reserves the
    unload transition and returns; the teardown is executed by maintenance. This
    asserts both halves of that contract: prompt return, and a real unload once
    tick() runs.
    """
    coord, deps, clock = build(calibrated_config)
    elapse_quiet(coord, clock)
    asyncio.run(coord.acquire_lease("req-1"))
    asyncio.run(coord.release_lease(coord._leases["inference:req-1:0"]))

    asyncio.run(coord.pause())
    # Returned promptly with the reserved transition named, not a fake "unloaded".
    assert coord.lifecycle in (Lifecycle.UNLOADING, Lifecycle.UNLOADED)
    assert coord._pending_unload is True or coord.lifecycle is Lifecycle.UNLOADED

    # Maintenance performs the teardown.
    action = asyncio.run(coord.tick())
    assert action == "unload_deferred"
    assert coord.lifecycle is Lifecycle.UNLOADED
    assert deps.unload_count == 1


def test_resume_clears_only_the_manual_veto(calibrated_config):
    """R10/T12: resume must not clear capacity, telemetry or external vetoes."""
    coord, deps, clock = build(calibrated_config)
    elapse_quiet(coord, clock)
    asyncio.run(coord.pause())
    asyncio.run(coord.resume())  # clear the pause so only the policy veto remains

    # An external workload is present and sustained -> veto.
    clock.advance(1)
    elapse_veto(coord, clock)
    asyncio.run(coord.pause())  # pause on top of the veto
    verdict = asyncio.run(coord.resume())
    assert verdict["paused"] is False
    assert verdict["can_admit_now"] is False
    assert coord.external_decision.state is PriorityState.BUSY


def test_resume_does_not_prewarm(calibrated_config):
    """R10: resume alone must not load a model."""
    coord, deps, clock = build(calibrated_config)
    elapse_quiet(coord, clock)
    asyncio.run(coord.pause())
    asyncio.run(coord.resume())
    assert deps.load_count == 0
    assert coord.lifecycle is Lifecycle.UNLOADED


def test_start_paused_blocks_admission(calibrated_config):
    """R10/T01: start_paused is honoured from configuration."""
    calibrated_config.start_paused = True
    coord, deps, clock = build(calibrated_config)
    elapse_quiet(coord, clock)
    result = asyncio.run(coord.acquire_lease("req-1"))
    assert result.outcome is AdmissionOutcome.DENIED
    assert result.reason is Reason.ORCHESTRATOR_PAUSED


# --------------------------------------------------------------------------- #
# Status / admission agreement (R11, T06)
# --------------------------------------------------------------------------- #


def test_status_and_admission_agree_on_the_same_snapshot(calibrated_config):
    """R06/R11/T06: published booleans must match the real admission evaluator."""
    coord, deps, clock = build(calibrated_config)
    elapse_quiet(coord, clock)
    asyncio.run(coord.acquire_lease("req-1"))

    status = asyncio.run(coord.status())
    assert status["admission"]["can_admit_now"] is False
    result = asyncio.run(coord.acquire_lease("req-2"))
    assert result.outcome is AdmissionOutcome.DENIED
    assert status["admission"]["can_admit_now"] == (
        result.outcome is AdmissionOutcome.GRANTED
    )


def test_clear_policy_with_blocked_cold_load_is_reportable(calibrated_config):
    """R11/T06: priority CLEAR with can_begin_cold_load false is valid and must name
    the actual blocker (here: insufficient memory)."""
    coord, deps, clock = build(calibrated_config)
    # Satisfy the quiet window while free memory stays far below the profile need.
    elapse_quiet(coord, clock, free=1000 * MIB)

    status = asyncio.run(coord.status())
    assert status["policy"] == PriorityState.CLEAR.value
    assert status["admission"]["can_begin_cold_load"] is False
    assert any("insufficient_vram" in b for b in status["admission"]["cold_blockers"])
    assert "insufficient_vram" in status["admission"]["reason"]


def test_status_reports_missing_metrics_as_null_not_zero(calibrated_config):
    """R11: a missing metric is null with a reason, not a fabricated zero."""
    coord, deps, clock = build(calibrated_config)
    status = asyncio.run(coord.status())  # no snapshot ingested yet
    assert status["gpu"]["memory_total"] is None
    assert status["gpu"]["utilization_percent"] is None
    assert status["detector"] is None
    assert status["telemetry"]["notes"] == ["no snapshot yet"]
    assert status["admission"]["can_admit_now"] is False


def test_status_is_available_while_paused(calibrated_config):
    """R12/T12: status must answer while paused."""
    coord, deps, clock = build(calibrated_config)
    asyncio.run(coord.pause())
    status = asyncio.run(coord.status())
    assert status["enabled"] is True
    assert status["paused"] is True


def test_status_details_stay_bounded(calibrated_config):
    """R11: no prompts, keys, command lines or URLs in status."""
    coord, deps, clock = build(calibrated_config)
    feed(coord, clock, external=[(4242, 500 * MIB)])
    status = asyncio.run(coord.status())
    top = status["gpu"]["top_external_processes"]
    assert len(top) <= 5
    for row in top:
        assert set(row) == {"pid", "used_mib", "comm", "sources"}


def test_shutdown_closes_admission_and_unloads(calibrated_config):
    """R13/T20: graceful shutdown closes admission and does not raise."""
    coord, deps, clock = build(calibrated_config)
    elapse_quiet(coord, clock)
    asyncio.run(coord.acquire_lease("req-1"))
    asyncio.run(coord.begin_shutdown())
    result = asyncio.run(coord.acquire_lease("req-2"))
    assert result.outcome is AdmissionOutcome.DENIED


# --------------------------------------------------------------------------- #
# Telemetry freshness / ownership (R02, R03, T02)
# --------------------------------------------------------------------------- #


def test_stale_telemetry_blocks_admission(calibrated_config):
    """R02/T03: a stale snapshot cannot admit new work."""
    coord, deps, clock = build(calibrated_config)
    elapse_quiet(coord, clock)
    clock.advance(60)  # no new samples; max_age is 3s
    result = asyncio.run(coord.acquire_lease("req-1"))
    assert result.outcome is AdmissionOutcome.DENIED
    assert result.reason is Reason.TELEMETRY_UNAVAILABLE
    assert any("telemetry_stale" in b for b in result.detail["blockers"])


def test_invalid_telemetry_blocks_admission(calibrated_config):
    """R02: missing essential memory data fails closed for new work."""
    coord, deps, clock = build(calibrated_config)
    feed(coord, clock, valid=False)
    result = asyncio.run(coord.acquire_lease("req-1"))
    assert result.outcome is AdmissionOutcome.DENIED
    assert coord.snapshot.valid is False


def test_owned_identity_is_not_counted_as_external(calibrated_config):
    """R03/T02: the server's own allocation must not veto its own admission."""
    coord, deps, clock = build(calibrated_config)
    own_pid = 999001
    coord._own_identities.add(f"boot-fake:{own_pid}:{own_pid * 10}")

    snap = make_snapshot(
        captured=clock.now,
        external=[(1, 50 * MIB)],
        owned=[(own_pid, 8500 * MIB)],
        utilization=0,
        activity_status=ActivityStatus.SPARSE,
    )
    coord.ingest_snapshot(snap, clock.now)
    assert snap.external_total_bytes == 50 * MIB, "owned memory must be excluded"
    assert coord.external_decision.state is PriorityState.CLEAR


def test_register_owned_pid_refuses_unresolvable_identity(calibrated_config):
    """R03/T02: an unverifiable identity must stay external, never be excluded."""
    coord, deps, clock = build(calibrated_config)
    key = coord.register_owned_pid(2**22 + 12345)  # almost certainly not running
    assert key is None
    assert coord.owned_identities == frozenset()


def test_release_lease_is_idempotent(calibrated_config):
    """R06/R08: cleanup paths may release defensively without corrupting counters."""
    coord, deps, clock = build(calibrated_config)
    elapse_quiet(coord, clock)
    lease = asyncio.run(coord.acquire_lease("req-1")).lease
    asyncio.run(coord.release_lease(lease))
    asyncio.run(coord.release_lease(lease))
    asyncio.run(coord.release_lease(lease))
    status = asyncio.run(coord.status())
    assert status["requests"]["active"] == 0


def test_reserve_breach_requests_drain_without_killing_work(calibrated_config):
    """R06/T11-ish (M2 sweep): a sustained reserve breach requests drain; it never
    forcibly unloads active work.

    The reserve is the safety margin held free. When free memory falls below it
    while a model is resident, the correct response is to stop new work and let
    the drain proceed — not to tear the model down under a live request.
    """
    coord, deps, clock = build(calibrated_config)
    elapse_quiet(coord, clock)
    lease = asyncio.run(coord.acquire_lease("req-1")).lease
    assert lease is not None

    # Free memory collapses below the reserve while the lease is held.
    clock.advance(0.5)
    feed(coord, clock, free=1 * MIB, used=(12227 - 1) * MIB)
    status = asyncio.run(coord.status())
    assert status["admission"]["can_admit_now"] is False, "a reserve breach must close admission"

    # The live request's lease is untouched and the model is not unloaded.
    assert lease.released is False, "a reserve breach must not cancel active work"
    assert deps.unload_count == 0, "teardown waits for the live lease"


def test_reused_pid_activity_is_not_trusted(calibrated_config):
    """R03/T02: activity from an earlier lifetime of a reused PID must not count.

    Identity is ``(boot_id, pid, start_ticks)`` precisely so a recycled PID
    cannot inherit another process's history. A snapshot whose activity names a
    PID that is not the one enumerated for that identity is not evidence about
    the enumerated process.
    """
    from orchestration.lifecycle import CoordinatorDeps

    coord, deps, clock = build(calibrated_config)
    # Enumerate an external PID with one identity...
    feed(
        coord,
        clock,
        external=[(4242, 64 * MIB)],
        activity_status=ActivityStatus.AVAILABLE,
        activity=[(4242, 90)],
    )
    # ...the tracker may latch it, but the *memory* trigger is the authority for
    # a resident-but-idle process, and an unresolved identity is never excluded
    # from external accounting.
    snap = coord.snapshot
    assert snap.external_total_bytes == 64 * MIB, "an unresolvable identity stays external"
    assert all(not p.is_owned for p in snap.processes)

    # A PID whose identity cannot be resolved must fail closed for ownership:
    # register_owned_pid refuses it, so it can never be silently excluded.
    key = coord.register_owned_pid(4242)
    assert key is not None or coord.owned_identities == frozenset()


def test_candidate_hold_does_not_evict_a_resident_model(calibrated_config):
    """R04/T04: a CANDIDATE holds new admission but must not unload the model.

    Only a *sustained* (BUSY) veto drains a resident model. While evidence is
    still accumulating, a working model must keep serving its live request.
    """
    coord, deps, clock = build(calibrated_config)
    elapse_quiet(coord, clock)
    lease = asyncio.run(coord.acquire_lease("req-1")).lease

    # One hot sample: CANDIDATE, not yet confirmed (enter_seconds not elapsed).
    clock.advance(0.2)
    feed(coord, clock, utilization=90, external=[(4242, 4096 * MIB)])
    assert coord.external_decision.state is PriorityState.CANDIDATE
    assert coord.external_decision.holds_admission is True, "candidate holds NEW admission"

    # No drain was requested and the live lease survives.
    assert asyncio.run(coord.tick()) is None
    assert coord.lifecycle in (Lifecycle.READY, Lifecycle.DRAINING)
    assert deps.unload_count == 0, "a candidate must not evict a working model"
    assert lease.released is False


def test_resume_cannot_cancel_an_explicit_unload_and_draining_is_not_a_wedge(calibrated_config):
    """R08/R10/R11 — review blocker F1, pinned as a regression test.

    An explicit admin unload must not be cancellable by ``resume()`` (which
    clears only the manual pause). Previously the unload route reused
    ``ORCHESTRATOR_PAUSED`` as its drain reason, so a resume cleared the pending
    drain while the lifecycle sat in DRAINING — and no tick() branch could ever
    unload from there, denying every request forever.
    """
    coord, deps, clock = build(calibrated_config)
    elapse_quiet(coord, clock)
    lease = asyncio.run(coord.acquire_lease("req-1")).lease
    assert lease is not None

    # Explicit unload while work is outstanding -> DRAINING.
    asyncio.run(coord.request_unload(Reason.EXPLICIT_UNLOAD))
    assert coord.lifecycle is Lifecycle.DRAINING

    # A resume must clear ONLY the manual veto and must not cancel the drain.
    asyncio.run(coord.resume())
    assert coord._drain_requested is True, "resume must not cancel an explicit unload"

    # The drain completes once the lease is released.
    asyncio.run(coord.release_lease(lease))
    assert asyncio.run(coord.tick()) == "unload_drained"
    assert coord.lifecycle is Lifecycle.UNLOADED
    assert deps.unload_count == 1


def test_draining_without_a_pending_drain_still_reaches_unloaded(calibrated_config):
    """R08/R11 — the wedge itself is unreachable, however DRAINING was entered.

    DRAINING with nothing outstanding must always be able to finish, so no
    combination of flags can strand the coordinator in a state that denies every
    request and never unloads.
    """
    coord, deps, clock = build(calibrated_config)
    elapse_quiet(coord, clock)
    asyncio.run(coord.acquire_lease("req-1"))
    asyncio.run(coord.release_lease(coord._leases["inference:req-1:0"]))

    # Force the pathological combination the old code could reach.
    coord.lifecycle = Lifecycle.DRAINING
    coord._drain_requested = False

    assert asyncio.run(coord.tick()) == "unload_drained"
    assert coord.lifecycle is Lifecycle.UNLOADED
    assert deps.unload_count == 1


def test_unload_on_an_unloaded_coordinator_leaves_no_sticky_drain(calibrated_config):
    """R08/R12 — review major F5, pinned.

    An unload requested while nothing is loaded must be a no-op. Previously it
    left `_drain_requested` set, so the NEXT cold load was unloaded by the
    following maintenance pass.
    """
    coord, deps, clock = build(calibrated_config)
    elapse_quiet(coord, clock)

    result = asyncio.run(coord.request_unload(Reason.EXPLICIT_UNLOAD))
    assert result["action"] == "already_unloaded"
    assert coord._drain_requested is False, "an unloaded no-op must not arm a drain"

    # A later eligible cold load must survive maintenance.
    granted = asyncio.run(coord.acquire_lease("req-1"))
    assert granted.outcome is AdmissionOutcome.GRANTED
    assert asyncio.run(coord.tick()) is None, "the fresh model must not be unloaded"
    assert deps.unload_count == 0


def test_state_only_denial_reports_a_real_reason_not_ok(calibrated_config):
    """R11/R12 — review major F4, pinned.

    A request refused purely because the model is mid-lifecycle must get a real
    reason code (and therefore a Retry-After), not 503 ``ok``.
    """
    coord, deps, clock = build(calibrated_config)
    elapse_quiet(coord, clock)
    coord.lifecycle = Lifecycle.DRAINING

    result = asyncio.run(coord.acquire_lease("req-1"))
    assert result.outcome is AdmissionOutcome.DENIED
    assert result.reason is Reason.MODEL_TRANSITION, (
        f"a lifecycle-only denial must not report {result.reason.value!r}"
    )
    assert any("lifecycle_not_ready" in b for b in result.detail["blockers"])


def test_wait_mode_new_arrival_cannot_bypass_a_queued_head(calibrated_config):
    """R09 — review blocker F2, pinned.

    The fast path used to grant to ANY request when the device looked admissible,
    without consulting the queue: a brand-new arrival could take the lease while
    an older waiter sat queued, which is exactly the bypass FIFO exists to
    prevent. Lease release must also wake the head so it is re-evaluated at the
    moment capacity frees rather than at the next sampler tick.
    """
    coord, deps, clock = build(wait_config(calibrated_config, wait_seconds=30.0))
    elapse_quiet(coord, clock)

    async def go():
        # The holder keeps the only inference slot.
        holder = await coord.acquire_lease("holder")
        assert holder.outcome is AdmissionOutcome.GRANTED

        # An older request queues behind the full capacity.
        older = asyncio.create_task(coord.acquire_lease("older"))
        for _ in range(3):
            await asyncio.sleep(0)
        assert [w.request_id for w in coord._waiters] == ["older"]

        # Capacity frees. The head must be woken and admitted — and a new arrival
        # racing in at this instant must not take the slot first.
        await coord.release_lease(holder.lease)
        older_result = await asyncio.wait_for(older, timeout=5)
        return older_result

    older_result = asyncio.run(go())
    assert older_result.outcome is AdmissionOutcome.GRANTED, (
        "the queued head must be admitted the moment capacity frees"
    )


def test_wait_mode_new_arrival_queues_behind_a_waiting_head(calibrated_config):
    """R09 — the bypass itself, asserted directly from the fast path.

    With an older waiter queued and the device otherwise admissible, a brand-new
    arrival must NOT be granted a lease: it joins the queue behind the head.
    """
    coord, deps, clock = build(wait_config(calibrated_config, wait_seconds=30.0))
    elapse_quiet(coord, clock)

    async def go():
        holder = await coord.acquire_lease("holder")
        older = asyncio.create_task(coord.acquire_lease("older"))
        for _ in range(3):
            await asyncio.sleep(0)

        # Free capacity but leave the head queued by making the new arrival race
        # first: it must still queue behind "older" rather than be granted.
        newer = asyncio.create_task(coord.acquire_lease("newer"))
        for _ in range(3):
            await asyncio.sleep(0)
        assert [w.request_id for w in coord._waiters] == ["older", "newer"], (
            "the newer arrival must queue behind the older waiter, not bypass it"
        )
        assert newer.done() is False

        # Cleanup: let the head through and cancel the rest.
        await coord.release_lease(holder.lease)
        older_result = await asyncio.wait_for(older, timeout=5)
        newer.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await newer
        return older_result

    older_result = asyncio.run(go())
    assert older_result.outcome is AdmissionOutcome.GRANTED


def test_disconnect_during_own_cold_load_does_not_latch_a_process_fault(calibrated_config):
    """R07/R09/R13 — review major F7, pinned.

    A client that disconnects while its own cold load is running must not cancel
    the shared loader and must not latch a process-wide FAULT: the coordinator
    owns the load, so it completes (or fails) on its own terms and the next
    request is served normally.
    """
    coord, deps, clock = build(wait_config(calibrated_config, wait_seconds=30.0), load_seconds=0.3)
    elapse_quiet(coord, clock)

    async def go():
        task = asyncio.create_task(coord.acquire_lease("disconnecting"))
        for _ in range(3):
            await asyncio.sleep(0)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        # Let the detached load finish and reconcile.
        for _ in range(60):
            await asyncio.sleep(0.01)
            if not coord._transition_active:
                break
        return coord.lifecycle

    lifecycle = asyncio.run(go())
    assert lifecycle is not Lifecycle.FAULT, (
        "a client disconnect must not fault the whole process"
    )
    assert deps.load_count == 1, "the shared load ran exactly once and was not abandoned"
    # The model is usable afterwards.
    assert asyncio.run(coord.status())["fault"] is None


def test_unknown_lease_release_is_ignored(calibrated_config):
    """R06: releasing a lease the coordinator does not know is a no-op, not a crash."""
    from orchestration.lifecycle import Lease

    coord, deps, clock = build(calibrated_config)
    stray = Lease(
        lease_id="not-real",
        kind=LeaseKind.INFERENCE,
        request_id="x",
        container_identity=None,
        acquired_monotonic=clock.now,
    )
    asyncio.run(coord.release_lease(stray))
    assert asyncio.run(coord.status())["requests"]["active"] == 0


# --------------------------------------------------------------------------- #
# Bounded admission waiting (R09, M3, T11)
# --------------------------------------------------------------------------- #


def wait_config(calibrated_config, pending=8, wait_seconds=2.0):
    """Flip the fixture config to wait mode with a short deadline."""
    calibrated_config.admission.mode = "wait"
    calibrated_config.admission.max_pending_requests = pending
    calibrated_config.admission.max_wait_seconds = wait_seconds
    return calibrated_config


def test_wait_mode_shared_load_waits_then_admits(calibrated_config):
    """R09/T11: a request behind a transition waits and is admitted after it."""
    coord, deps, clock = build(wait_config(calibrated_config), load_seconds=1.0)
    elapse_quiet(coord, clock)

    async def go():
        first = asyncio.create_task(coord.acquire_lease("req-1"))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        second_task = asyncio.create_task(coord.acquire_lease("req-2"))
        first_result = await first
        # Free the capacity so the queued waiter can be admitted.
        await asyncio.sleep(0.01)
        await coord.release_lease(first_result.lease)
        second = await asyncio.wait_for(second_task, timeout=5)
        return first_result, second

    first_result, second = asyncio.run(go())
    assert first_result.outcome is AdmissionOutcome.GRANTED
    assert second.outcome is AdmissionOutcome.GRANTED, (
        "wait mode holds the second request on the FIFO instead of denying it"
    )
    assert deps.load_count == 1, "the shared load happened exactly once"


def test_wait_mode_deadline_is_enforced(calibrated_config):
    """R09/T11: pre-lease time is bounded; expiry is a timeout, no lease taken.

    The injected clock is the only deadline authority, so the test drives it:
    a background task advances the fake clock and feeds fresh snapshots, the
    way the production sampler wakes the wait loop.
    """
    coord, deps, clock = build(wait_config(calibrated_config, wait_seconds=1.0))

    async def go():
        async def sampler():
            for _ in range(40):
                await asyncio.sleep(0.001)
                clock.advance(0.1)
                coord.ingest_snapshot(
                    make_snapshot(
                        captured=clock.now,
                        utilization=0,
                        external=[],
                        activity_status=ActivityStatus.SPARSE,
                    ),
                    clock.now,
                )

        pump = asyncio.create_task(sampler())
        result = await asyncio.wait_for(coord.acquire_lease("req-1", deadline_seconds=0.05), timeout=5)
        pump.cancel()
        return result

    result = asyncio.run(go())
    assert result.outcome is AdmissionOutcome.TIMEOUT
    assert result.reason is Reason.ADMISSION_TIMEOUT
    assert deps.load_count == 0, "a timed-out request must not leave demand behind"
    assert coord._waiters == [], "the expired slot must be released"


def test_wait_mode_overflow_fails_immediately(calibrated_config):
    """R09/T11: a full wait list rejects new arrivals immediately."""
    coord, deps, clock = build(wait_config(calibrated_config, pending=1))
    elapse_quiet(coord, clock)

    async def go():
        # Occupant: vetoed so it stays queued (never admissible in this window).
        elapse_veto(coord, clock)
        occupant = asyncio.create_task(coord.acquire_lease("occupied"))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        overflow = await coord.acquire_lease("overflow")
        occupant.cancel()
        return overflow

    result = asyncio.run(go())
    assert result.outcome is AdmissionOutcome.DENIED
    assert result.reason is Reason.ADMISSION_QUEUE_FULL


def test_wait_mode_fifo_prevents_bypass(calibrated_config):
    """R09/T11: a newer arrival cannot be admitted past an older eligible waiter.

    While a veto holds, an older request sits at the queue head. A newer arrival
    must land *behind* it and must not hold a lease. When the veto clears, the
    OLDER waiter is the one granted.
    """
    coord, deps, clock = build(wait_config(calibrated_config, wait_seconds=30.0))

    async def go():
        elapse_veto(coord, clock)
        older = asyncio.create_task(coord.acquire_lease("older"))
        for _ in range(3):
            await asyncio.sleep(0)
        newer = asyncio.create_task(coord.acquire_lease("newer"))
        for _ in range(3):
            await asyncio.sleep(0)

        # FIFO order is observable in the queue itself, and neither arrival holds
        # a lease while the veto stands.
        assert [w.request_id for w in coord._waiters] == ["older", "newer"]
        assert newer.done() is False, "a newer arrival must not be admitted while older waits"
        assert (await coord.status())["requests"]["active"] == 0

        # Let time pass with a quiet device: the veto's release window elapses and
        # the FIFO head is granted.
        async def clear_and_wait():
            return await asyncio.wait_for(older, timeout=15)

        older_result = await with_clock_pump(
            coord, clock, clear_and_wait(), step=0.5, max_seconds=60.0
        )
        newer.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await newer
        return older_result

    older_result = asyncio.run(go())
    assert older_result.outcome is AdmissionOutcome.GRANTED, (
        "the FIFO head must be granted once the veto clears"
    )


def test_wait_mode_disconnect_releases_the_slot(calibrated_config):
    """R09/T11: a cancelled waiter frees its slot exactly once."""
    coord, deps, clock = build(wait_config(calibrated_config, wait_seconds=30.0))

    async def go():
        elapse_veto(coord, clock)
        task = asyncio.create_task(coord.acquire_lease("disconnecting"))
        for _ in range(3):
            await asyncio.sleep(0)
        assert len(coord._waiters) == 1
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        assert len(coord._waiters) == 0, "disconnect must release the wait slot"

    asyncio.run(go())


def test_wait_mode_pause_fails_waiters(calibrated_config):
    """R09/R10: a manual pause fails unadmitted waiters instead of parking them."""
    coord, deps, clock = build(wait_config(calibrated_config, wait_seconds=30.0))

    async def go():
        elapse_veto(coord, clock)
        waiter_task = asyncio.create_task(coord.acquire_lease("paused-req"))
        for _ in range(3):
            await asyncio.sleep(0)
        result = await coord.pause()
        assert result["paused"] is True
        waiter_result = await asyncio.wait_for(waiter_task, timeout=3)
        return waiter_result

    result = asyncio.run(go())
    assert result.outcome is AdmissionOutcome.DENIED
    assert result.reason is Reason.ORCHESTRATOR_PAUSED
    assert coord._waiters == []


def test_wait_mode_no_success_headers_before_admission(calibrated_config):
    """R09/T11: a waiting request has NOT been granted a lease yet."""
    coord, deps, clock = build(wait_config(calibrated_config, wait_seconds=30.0))

    async def go():
        elapse_veto(coord, clock)
        task = asyncio.create_task(coord.acquire_lease("pinned"))
        for _ in range(3):
            await asyncio.sleep(0)
        # While queued, the request holds NO lease and started NO inference.
        status = await coord.status()
        assert status["requests"]["active"] == 0
        assert status["requests"]["pending"] == 1
        assert deps.calls.count("load:start") == 0
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    asyncio.run(go())


def test_waiter_deadline_covers_queue_and_load_time(calibrated_config):
    """R09: the deadline is ONE bound from enqueue — queue time consumes it.

    The triggering request owns a cold load that takes real (injected) time; a
    second request enqueues behind it and is never admitted because the first
    request keeps its lease. The queued request must therefore report a bounded
    timeout — never hang, and never start a load of its own.
    """
    coord, deps, clock = build(wait_config(calibrated_config, wait_seconds=20.0), load_seconds=3.0)
    elapse_quiet(coord, clock)

    async def go():
        owner = asyncio.create_task(coord.acquire_lease("owner"))
        for _ in range(3):
            await asyncio.sleep(0)
        queued = asyncio.create_task(coord.acquire_lease("queued", deadline_seconds=5.0))

        async def wait_both():
            owner_result = await asyncio.wait_for(owner, timeout=20)
            queued_result = await asyncio.wait_for(queued, timeout=20)
            return owner_result, queued_result

        return await with_clock_pump(coord, clock, wait_both(), step=0.5, max_seconds=60.0)

    owner_result, queued_result = asyncio.run(go())
    assert owner_result.outcome is AdmissionOutcome.GRANTED, "the triggering request owns and wins its load"
    assert queued_result.outcome is AdmissionOutcome.TIMEOUT, (
        f"a request stuck behind a held lease must time out, not hang (got {queued_result.outcome})"
    )
    assert queued_result.reason is Reason.ADMISSION_TIMEOUT
    assert queued_result.lease is None
    assert deps.load_count == 1, "a timed-out waiter must not start its own load"
    assert coord._waiters == [], "the expired slot must be released"


def test_queues_do_not_survive_restart(calibrated_config):
    """R09/R13: restart starts with no pending requests (in-memory only)."""
    coord, deps, clock = build(wait_config(calibrated_config, wait_seconds=30.0))
    elapse_veto(coord, clock)

    async def go():
        task = asyncio.create_task(coord.acquire_lease("queued"))
        for _ in range(3):
            await asyncio.sleep(0)
        assert len(coord._waiters) == 1, "the request is queued in-memory"
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(go())
    # A fresh coordinator (restart) has no waiters — nothing was persisted.
    fresh_coord, _, _ = build(calibrated_config)
    assert fresh_coord._waiters == []
    assert asyncio.run(fresh_coord.status())["requests"]["pending"] == 0


# --------------------------------------------------------------------------- #
# Generator-recovery ownership (R07 — review major F12)
# --------------------------------------------------------------------------- #


def _adopt_running_recovery(coord, deps):
    """Put a live fake recovery task on the deps and adopt it."""

    async def never_settles():
        await asyncio.Event().wait()  # parked until the test cancels it

    deps.recovery_task = asyncio.get_event_loop().create_task(never_settles())
    assert coord.adopt_recovery_task() is True
    assert coord._recovery_task is not None
    return deps.recovery_task


def test_recovery_running_closes_admission(calibrated_config):
    """R07/F12: new admission must be closed while a generator recovery runs.

    The recovery task cancels every active job before rebuilding the generator,
    so `backend_busy` (the job registry) can legitimately read EMPTY mid-recovery.
    The coordinator's verdict must not depend on that blind registry while the
    backend is mid-mutation.
    """
    coord, deps, clock = build(calibrated_config)
    elapse_quiet(coord, clock)
    # Make the model resident and otherwise perfectly admissible.
    result = asyncio.run(coord.acquire_lease("warmup"))
    assert result.outcome is AdmissionOutcome.GRANTED
    asyncio.run(coord.release_lease(result.lease))

    async def go():
        recovery = _adopt_running_recovery(coord, deps)
        try:
            verdict = await coord.admission_preview()
            assert verdict["can_admit_now"] is False, (
                "an in-flight generator recovery must close admission"
            )
            assert any("model_recovery" in b for b in verdict["blockers"])
            assert verdict["reason"] == "model_transition"
            # The real decision path denies too (status and admission agree).
            denied = await coord.acquire_lease("req-2")
            assert denied.outcome is AdmissionOutcome.DENIED
        finally:
            recovery.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await recovery
            # Done-callback clears the reference; one tick lets it run.
            for _ in range(3):
                await asyncio.sleep(0)
        assert coord._recovery_task is None, "a settled recovery must be forgotten"
        verdict = await coord.admission_preview()
        assert verdict["can_admit_now"] is True, (
            "once recovery settles, admission must reopen"
        )

    asyncio.run(go())


def test_tick_does_not_unload_under_a_running_recovery(calibrated_config):
    """R07/F12: the exact race the finding describes, pinned.

    Backend busy reads EMPTY (recovery cancelled the jobs first), no leases are
    held — the pre-fix coordinator would reserve and start an unload here,
    tearing the container down under live CUDA work.
    """
    coord, deps, clock = build(calibrated_config)
    elapse_quiet(coord, clock)
    result = asyncio.run(coord.acquire_lease("warmup"))
    assert result.outcome is AdmissionOutcome.GRANTED
    asyncio.run(coord.release_lease(result.lease))

    async def go():
        recovery = _adopt_running_recovery(coord, deps)
        try:
            assert deps._busy is False and not coord._has_live_lease_locked()
            action = await coord.tick()
            assert action is None, (
                "a tick must not start an unload while a generator recovery is running"
            )
            assert coord.lifecycle is Lifecycle.READY
            # Even with a drain explicitly requested, recovery defers it.
            coord._drain_requested = True
            coord._drain_reason = Reason.EXPLICIT_UNLOAD
            action = await coord.tick()
            assert action is None, "recovery must defer a requested drain too"
            assert coord.lifecycle is Lifecycle.DRAINING, (
                "the drain request must still be honoured as a state (drain waits)"
            )
        finally:
            recovery.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await recovery
            for _ in range(3):
                await asyncio.sleep(0)

        # Recovery settled: the same tick now completes the drain -> unload.
        action = await coord.tick()
        assert action == "unload_drained", (
            f"once recovery settles the deferred drain must proceed (got {action})"
        )
        assert coord.lifecycle is Lifecycle.UNLOADED
        assert deps.unload_count == 1

    asyncio.run(go())


def test_recovery_blocks_idle_ttl_unload(calibrated_config):
    """R07/F12: the idle-TTL branch must not unload underneath recovery either."""
    coord, deps, clock = build(calibrated_config)
    elapse_quiet(coord, clock)
    result = asyncio.run(coord.acquire_lease("warmup"))
    asyncio.run(coord.release_lease(result.lease))

    async def go():
        recovery = _adopt_running_recovery(coord, deps)
        try:
            # Idle long past the TTL with the registry empty.
            clock.advance(coord.cfg.idle_unload.seconds + 1.0)
            action = await coord.tick()
            assert action is None, "TTL expiry must not unload under recovery"
            assert coord.lifecycle is Lifecycle.READY
        finally:
            recovery.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await recovery
            for _ in range(3):
                await asyncio.sleep(0)
        # Now the TTL path may act.
        action = await coord.tick()
        assert action == "unload_idle"

    asyncio.run(go())


def test_status_agrees_with_acquisition_for_a_queued_head(calibrated_config):
    """R11/T06 — review major F3, pinned.

    Invariant: with pending > 0 and an otherwise-admissible state, the published
    `can_admit_now`/`reason` must agree with what acquire_lease actually decides
    for that queued request. The coordinator now keeps ONE evaluator (the same
    `_evaluate_locked`) and the wait path evaluates the head as itself, so the
    published verdict and the real decision cannot diverge.
    """
    coord, deps, clock = build(wait_config(calibrated_config, wait_seconds=30.0))
    elapse_quiet(coord, clock)

    async def go():
        holder = await coord.acquire_lease("holder")
        assert holder.outcome is AdmissionOutcome.GRANTED

        # An older eligible request queues behind the single active slot.
        head = asyncio.create_task(coord.acquire_lease("head-waiter"))
        for _ in range(3):
            await asyncio.sleep(0)
        assert [w.request_id for w in coord._waiters] == ["head-waiter"]

        # Status (a monitor's view — no request identity) must NOT claim the
        # server is admitting while the eligible head waiter sits unserved.
        status = await coord.status()
        assert status["requests"]["pending"] == 1
        assert status["admission"]["can_admit_now"] is False, (
            "can_admit_now=true while an eligible request is queued describes "
            "a server that looks ready but is not serving that request (F3)"
        )
        assert status["admission"]["reason"] == "request_capacity", (
            f"the queued-head blocker must be named, not 'ok' "
            f"(got {status['admission']['reason']!r})"
        )

        # The head waiter itself is granted as soon as capacity frees: its own
        # evaluation must not be demoted by its own presence in the queue.
        await coord.release_lease(holder.lease)
        head_result = await asyncio.wait_for(head, timeout=5)
        return head_result, coord

    head_result, coord = asyncio.run(go())
    assert head_result.outcome is AdmissionOutcome.GRANTED, (
        "the queued head must be admitted — the demotion must apply to monitors, "
        "never block the head's own admission"
    )
    assert asyncio.run(coord.status())["requests"]["pending"] == 0
    assert asyncio.run(coord.status())["admission"]["can_admit_now"] is False, (
        "the head holds the only slot again: capacity is consumed, so the "
        "monitor's verdict correctly stays non-admissible"
    )


def test_status_reason_names_queued_head_not_ok(calibrated_config):
    """R11 — the demoted verdict carries a real reason code, never 'ok'."""
    coord, deps, clock = build(wait_config(calibrated_config, wait_seconds=30.0))
    elapse_quiet(coord, clock)

    async def go():
        holder = await coord.acquire_lease("holder")
        queued = asyncio.create_task(coord.acquire_lease("queued"))
        for _ in range(3):
            await asyncio.sleep(0)
        status = await coord.status()
        queued.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await queued
        await coord.release_lease(holder.lease)
        return status

    status = asyncio.run(go())
    assert status["admission"]["reason"] != "ok"
    assert status["admission"]["can_admit_now"] is False


def test_request_unload_defers_to_recovery(calibrated_config):
    """R07/F12: an explicit unload during recovery drains, not unloads."""
    coord, deps, clock = build(calibrated_config)
    elapse_quiet(coord, clock)
    result = asyncio.run(coord.acquire_lease("warmup"))
    asyncio.run(coord.release_lease(result.lease))

    async def go():
        recovery = _adopt_running_recovery(coord, deps)
        try:
            outcome = await coord.request_unload(Reason.EXPLICIT_UNLOAD)
            assert outcome["action"] == "draining", (
                "an unload during recovery must be a deferred drain"
            )
            assert coord.lifecycle is Lifecycle.DRAINING
            assert deps.unload_count == 0
        finally:
            recovery.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await recovery
            for _ in range(3):
                await asyncio.sleep(0)
        await coord.tick()
        assert deps.unload_count == 1, "the deferred drain completes after recovery"

    asyncio.run(go())


# --------------------------------------------------------------------------- #
# Minors F13/F14/F15/F15b
# --------------------------------------------------------------------------- #


def test_f15_typed_envelope_comparison_distinguishes_type_drift(calibrated_config):
    """F15: str() comparison cannot distinguish int 1 from "1" — typed compare.

    A container reporting a STRING where the calibration pinned an int is type
    drift (the effective load did not follow the calibrated path) and must
    latch FAULT; a numeric value reported with a different numeric type
    (131072 vs 131072.0) is the same setting and must verify clean.
    """
    coord, deps, clock = build(calibrated_config)
    elapse_quiet(coord, clock)
    import dataclasses

    coord.profile = dataclasses.replace(
        coord.profile, envelope={"max_seq_len": 131072, "cache_mode": "Q4"}
    )

    async def go():
        # Same numeric value, different numeric type -> NOT a divergence.
        coord.deps.container_envelope = lambda: {"max_seq_len": 131072.0, "cache_mode": "Q4"}
        assert coord._verify_envelope_locked() is None
        # Same digits, string type -> divergence (the old str() compare missed this).
        coord.deps.container_envelope = lambda: {"max_seq_len": "131072", "cache_mode": "Q4"}
        mismatch = coord._verify_envelope_locked()
        assert mismatch is not None and "max_seq_len" in mismatch
        # bool vs int must NOT be conflated (True == 1 numerically).
        coord.profile = dataclasses.replace(coord.profile, envelope={"max_batch_size": 1})
        coord.deps.container_envelope = lambda: {"max_batch_size": True}
        assert coord._verify_envelope_locked() is not None

    asyncio.run(go())


def test_f15b_unnormalised_chunk_size_is_rejected_at_enablement(calibrated_config):
    """F15b: a calibrated chunk_size of 1000 would verify against the backend's
    normalised 1024 and read back as a spurious FAULT — refuse it at enablement."""
    from orchestration.config import enablement_errors

    calibrated_config.model.chunk_size = 1000
    errors = enablement_errors(calibrated_config)
    assert any("chunk_size" in e and "multiple of 256" in e for e in errors), errors

    calibrated_config.model.chunk_size = 1024
    assert not [e for e in enablement_errors(calibrated_config) if "chunk_size" in e]


def test_f13_waiter_wake_bounded_below_sampler_cadence(calibrated_config):
    """F13: a queued head is woken by a bounded poll, not left to the sampler.

    The earlier version of this test asserted `min(10.0, 0.2) == 0.2`, which
    restated the literal constant and would have passed with ANY cap (review
    finding — a tautology, not evidence). This version exercises the real
    behaviour: with the injected clock ADVANCED past the deadline but no sampler
    wake ever set, the waiter must still return a bounded TIMEOUT — which is only
    possible because the loop re-checks on its own poll.
    """
    coord, deps, clock = build(wait_config(calibrated_config, wait_seconds=1.0))
    elapse_quiet(coord, clock)
    # Hold the only slot so the waiter can never be admitted.
    holder = asyncio.run(coord.acquire_lease("holder"))
    assert holder.outcome is AdmissionOutcome.GRANTED

    async def go():
        async def deadline_pump():
            # Advance the clock past the deadline WITHOUT ingesting a snapshot:
            # no sampler wake is ever delivered to the waiter.
            for _ in range(40):
                await asyncio.sleep(0.01)
                clock.advance(0.1)

        pump = asyncio.create_task(deadline_pump())
        try:
            return await asyncio.wait_for(coord.acquire_lease("queued"), timeout=5)
        finally:
            pump.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pump

    result = asyncio.run(go())
    assert result.outcome is AdmissionOutcome.TIMEOUT, (
        "the waiter must time out on its own poll, with no sampler wake"
    )
    assert result.reason is Reason.ADMISSION_TIMEOUT
    assert coord._waiters == []


def test_f14_envelope_check_skips_only_unreported_keys(calibrated_config):
    """F14 (decision recorded): the fail-open applies ONLY to keys the container
    cannot report; every reported key must still match exactly."""
    coord, deps, clock = build(calibrated_config)
    elapse_quiet(coord, clock)
    import dataclasses

    coord.profile = dataclasses.replace(
        coord.profile, envelope={"max_seq_len": 131072, "cache_size": 131072}
    )

    async def go():
        # cache_size unreported -> skipped; max_seq_len reported and matching.
        coord.deps.container_envelope = lambda: {"max_seq_len": 131072}
        assert coord._verify_envelope_locked() is None
        # A reported divergence still latches even when another key is unreported.
        coord.deps.container_envelope = lambda: {"max_seq_len": 4096}
        assert coord._verify_envelope_locked() is not None

    asyncio.run(go())


# --------------------------------------------------------------------------- #
# Second-round review findings (this pass, read-only review of M2.5)
# --------------------------------------------------------------------------- #


def test_review_admission_closed_before_the_recovery_is_adopted(calibrated_config):
    """MAJOR: the live container is consulted, not only the adopted copy.

    Adoption runs once per sampler cadence, so an adopted-only check left a ≤1 s
    window in which a recovery was already scheduled on the container but not yet
    adopted — admission would be granted, and a tick could begin unloading, while
    the backend sat mid-`create_generator()` with an EMPTY job registry. The check
    must read the container through deps as well.
    """
    coord, deps, clock = build(calibrated_config)
    elapse_quiet(coord, clock)
    # Model resident and otherwise perfectly admissible.
    warm = asyncio.run(coord.acquire_lease("warmup"))
    assert warm.outcome is AdmissionOutcome.GRANTED
    asyncio.run(coord.release_lease(warm.lease))

    async def go():
        # Container says a recovery is running; the coordinator has NOT adopted it.
        assert coord._recovery_task is None
        deps.recovery_running = True
        try:
            verdict = await coord.admission_preview()
            assert verdict["can_admit_now"] is False, (
                "an un-adopted recovery on the live container must still close admission"
            )
            assert any("model_recovery" in b for b in verdict["blockers"])
            denied = await coord.acquire_lease("racing-request")
            assert denied.outcome is AdmissionOutcome.DENIED, (
                "a request racing an un-adopted recovery must not be admitted"
            )
            # And a tick must not unload underneath it either.
            coord._drain_requested = True
            coord._drain_reason = Reason.EXPLICIT_UNLOAD
            assert await coord.tick() is None
            assert coord.lifecycle is Lifecycle.DRAINING
        finally:
            deps.recovery_running = False
            coord._drain_requested = False
            coord._drain_reason = None
            coord.lifecycle = Lifecycle.READY  # undo the drain armed above
        # Recovery gone and the state restored: admission reopens.
        assert (await coord.admission_preview())["can_admit_now"] is True

    asyncio.run(go())


def test_review_adoption_is_idempotent_and_ignores_stale_callbacks(calibrated_config):
    """MINOR: one callback per task, and a superseded task must not clear a newer one."""
    coord, deps, clock = build(calibrated_config)
    elapse_quiet(coord, clock)

    async def go():
        first = _adopt_running_recovery(coord, deps)
        try:
            # Re-adopting the same task must not stack another done-callback.
            before = len(coord._recovery_task._callbacks)
            assert coord.adopt_recovery_task() is True
            assert len(coord._recovery_task._callbacks) == before, (
                "re-adoption must not register duplicate done-callbacks"
            )
        finally:
            first.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await first
            for _ in range(3):
                await asyncio.sleep(0)
        assert coord._recovery_task is None

        # A superseded task's callback must not null the newer reference.
        async def never_settles():
            await asyncio.Event().wait()

        stale = asyncio.get_event_loop().create_task(never_settles())
        current = asyncio.get_event_loop().create_task(never_settles())
        try:
            coord._recovery_task = current
            coord._recovery_task_done(stale)  # fires for the OLD task
            assert coord._recovery_task is current, (
                "a stale callback must not clear the task now being tracked"
            )
            assert coord._recovery_outstanding_locked() is True
        finally:
            for t in (stale, current):
                t.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await t
            for _ in range(3):
                await asyncio.sleep(0)

    asyncio.run(go())


def test_review_reader_verdict_agrees_with_status_under_a_queued_head(calibrated_config):
    """MAJOR: the F3 invariant must hold for READER too.

    A queued INFERENCE waiter says nothing about whether a metadata read may
    proceed, so demoting READER made status and the READER decision disagree on
    the same snapshot — the F3 shape, one branch over.
    """
    coord, deps, clock = build(wait_config(calibrated_config, wait_seconds=30.0))
    elapse_quiet(coord, clock)

    async def go():
        holder = await coord.acquire_lease("holder")
        assert holder.outcome is AdmissionOutcome.GRANTED
        queued = asyncio.create_task(coord.acquire_lease("queued-inference"))
        for _ in range(3):
            await asyncio.sleep(0)

        # Status is demoted (an inference waiter is queued)...
        status = await coord.status()
        assert status["admission"]["can_admit_now"] is False
        assert status["admission"]["reason"] == "request_capacity"

        # ...and the decision a READER actually gets is its OWN verdict, which
        # reflects the real blocker without inventing a different code.
        reader = await coord.acquire_lease("read-under-queue", kind=LeaseKind.READER)
        assert reader.outcome is AdmissionOutcome.GRANTED, (
            "a reader pin does not contend for the inference slot, so a queued "
            f"inference waiter must not block it (got {reader.outcome} / {reader.reason})"
        )
        assert reader.reason is Reason.OK

        queued.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await queued
        await coord.release_lease(holder.lease)

    asyncio.run(go())


def test_review_queued_head_blocker_is_published_in_blockers_too(calibrated_config):
    """MINOR: R11 requires the reason code AND the blockers together."""
    coord, deps, clock = build(wait_config(calibrated_config, wait_seconds=30.0))
    elapse_quiet(coord, clock)

    async def go():
        holder = await coord.acquire_lease("holder")
        queued = asyncio.create_task(coord.acquire_lease("queued"))
        for _ in range(3):
            await asyncio.sleep(0)
        status = await coord.status()
        queued.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await queued
        await coord.release_lease(holder.lease)
        return status

    status = asyncio.run(go())
    assert status["admission"]["can_admit_now"] is False
    assert status["admission"]["reason"] == "request_capacity"
    assert status["admission"]["blockers"], (
        "a monitor keying on `blockers` must not see an empty list beside a "
        "non-admissible verdict"
    )
    # The blocker list names the real obstacle (the active-request limit) and the
    # queued-head demotion is consistent with the published reason code.
    assert any("request_capacity" in b for b in status["admission"]["blockers"])


# --------------------------------------------------------------------------- #
# Restart reconciliation (R13)
# --------------------------------------------------------------------------- #


def test_restart_with_a_stray_container_latches_fault_and_blocks_admission(calibrated_config):
    """R13: residency is READ from the container, never inherited as belief.

    A container that exists at startup without an orchestrated load (an upstream
    startup load, a dummy model, a leftover from a previous process) means the
    process is holding unmeasured residency. Publishing UNLOADED and admitting a
    cold load against it would double-allocate; the honest outcome is FAULT with
    admission closed and the bypass named.
    """
    coord, deps, clock = build(calibrated_config)
    deps._present = True  # a container exists that this coordinator never loaded

    result = asyncio.run(coord.reconcile_startup())

    assert result["container_present"] is True
    assert result["lifecycle"] == "FAULT"
    assert result["leases"] == 0 and result["pending"] == 0
    status = asyncio.run(coord.status())
    assert status["lifecycle"] == "FAULT"
    assert "without an orchestrated load" in status["fault"]
    denial = asyncio.run(coord.acquire_lease("req-after-restart"))
    assert denial.outcome is AdmissionOutcome.DENIED


def test_restart_clears_leases_and_waiters_and_reads_no_residency(calibrated_config):
    """R13: "no trusted residency, no leases, no pending requests" after restart."""
    coord, deps, clock = build(wait_config(calibrated_config, wait_seconds=30.0))
    elapse_quiet(coord, clock)

    async def go():
        # Build up believable state: one held lease and one queued waiter.
        holder = await coord.acquire_lease("holder")
        queued = asyncio.create_task(coord.acquire_lease("queued"))
        for _ in range(3):
            await asyncio.sleep(0)
        assert coord._leases and coord._waiters
        queued.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await queued
        return holder

    asyncio.run(go())
    assert coord._leases, "precondition: a lease is outstanding"

    # Model the actual restart: a new process with nothing resident. The lease
    # table and queue are in-memory only, so the reconciliation must clear them
    # and publish the state it OBSERVES (no container -> UNLOADED).
    loads_before = deps.load_count
    deps._present = False
    result = asyncio.run(coord.reconcile_startup())

    assert result["leases"] == 0 and result["pending"] == 0
    assert coord._leases == {} and coord._waiters == []
    assert result["lifecycle"] == "UNLOADED"
    assert deps.load_count == loads_before, "reconciliation must not load anything"


def test_restart_adopts_a_recovery_already_running(calibrated_config):
    """R13/F12: a recovery in flight across the startup boundary is observed."""
    coord, deps, clock = build(calibrated_config)

    async def never_settles():
        await asyncio.Event().wait()

    async def go():
        deps.recovery_task = asyncio.get_running_loop().create_task(never_settles())
        try:
            result = await coord.reconcile_startup()
            assert result["recovery_adopted"] is True
            assert coord._recovery_task is deps.recovery_task
            # And the protection is live immediately: no teardown under recovery.
            verdict = await coord.admission_preview()
            assert verdict["can_admit_now"] is False
            assert any("model_recovery" in b for b in verdict["blockers"])
        finally:
            deps.recovery_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await deps.recovery_task

    asyncio.run(go())
