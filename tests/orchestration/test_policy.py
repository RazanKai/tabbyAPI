"""Pure-policy tests: hysteresis, sparse activity, capacity, validation.

Mocked tier. Every case injects its own clock and snapshots, so no GPU, no server
and no wall-clock sleeps are involved. Each test names the SPEC requirement and
acceptance case it is evidence for.
"""

from __future__ import annotations

import pytest

from orchestration import policy
from orchestration.config import OrchestratorConfig, enablement_errors
from orchestration.policy import (
    ColdLoadConfig,
    ExternalWorkloadConfig,
    ExternalWorkloadTracker,
    PriorityState,
    QuietWindowTracker,
    Reason,
    VramProfile,
    derive_coverage,
    evaluate_capacity,
)
from orchestration.telemetry import ActivityStatus

from orch_helpers import MIB, make_snapshot


def ew_config(**over) -> ExternalWorkloadConfig:
    base = dict(
        process_vram_enter_bytes=1024 * MIB,
        process_vram_release_bytes=768 * MIB,
        total_vram_enter_bytes=2304 * MIB,
        total_vram_release_bytes=1792 * MIB,
        process_activity_enter_percent=25,
        process_activity_release_percent=10,
        enter_seconds=2.0,
        release_seconds=10.0,
    )
    base.update(over)
    return ExternalWorkloadConfig(**base)


def cold_config(**over) -> ColdLoadConfig:
    base = dict(
        max_device_utilization_percent=15,
        quiet_seconds=10.0,
        max_external_vram_growth_bytes=128 * MIB,
    )
    base.update(over)
    return ColdLoadConfig(**base)


# --------------------------------------------------------------------------- #
# External workload: entry requires sustained evidence (R04, T04)
# --------------------------------------------------------------------------- #


def test_memory_trigger_enters_only_after_enter_seconds():
    """R04/T04: a single hot sample is a CANDIDATE, not yet BUSY."""
    tracker = ExternalWorkloadTracker(ew_config(enter_seconds=2.0))
    hot = make_snapshot(captured=0.0, external=[(4242, 2048 * MIB)])

    d1 = tracker.update(hot, 0.0)
    assert d1.state is PriorityState.CANDIDATE
    assert d1.holds_admission is True, "a candidate must hold NEW admission"

    d2 = tracker.update(hot, 2.5)
    assert d2.state is PriorityState.BUSY
    assert d2.trigger == "memory"
    assert d2.reason is Reason.EXTERNAL_GPU_BUSY


def test_activity_trigger_alone_vetoes_with_low_vram():
    """R04: activity covers compute-heavy workloads that hold little memory."""
    tracker = ExternalWorkloadTracker(ew_config(enter_seconds=0.0))
    snap = make_snapshot(
        captured=0.0,
        external=[(4242, 64 * MIB)],
        activity_status=ActivityStatus.AVAILABLE,
        activity=[(4242, 90)],
    )
    d = tracker.update(snap, 0.0)
    assert d.state is PriorityState.BUSY
    assert d.trigger == "activity"


def test_below_enter_thresholds_stays_clear():
    """R04/T04: recorded idle/browser traces must be allowed through."""
    tracker = ExternalWorkloadTracker(ew_config())
    quiet = make_snapshot(
        captured=0.0,
        external=[(541647, 345 * MIB), (2282, 226 * MIB)],
        activity_status=ActivityStatus.SPARSE,
    )
    d = tracker.update(quiet, 0.0)
    assert d.state is PriorityState.CLEAR
    assert d.holds_admission is False


def test_hysteresis_prevents_flapping_between_thresholds():
    """R04/T04: between enter and release thresholds, the prior latch is preserved."""
    tracker = ExternalWorkloadTracker(ew_config(enter_seconds=0.0, release_seconds=10.0))
    hot = make_snapshot(captured=0.0, external=[(4242, 2048 * MIB)])
    assert tracker.update(hot, 0.0).state is PriorityState.BUSY

    # Drop into the hysteresis band: above release, below enter.
    band = make_snapshot(captured=1.0, external=[(4242, 900 * MIB)])
    d = tracker.update(band, 1.0)
    assert d.state is PriorityState.BUSY, "must stay vetoed inside the band"
    assert d.holds_admission is True
    assert "releasing" in d.blockers or "external_memory_still_high" in d.blockers


def test_release_requires_full_release_window():
    """R04/T04: release needs all signals below release thresholds for release_seconds."""
    tracker = ExternalWorkloadTracker(ew_config(enter_seconds=0.0, release_seconds=10.0))
    hot = make_snapshot(captured=0.0, external=[(4242, 2048 * MIB)])
    tracker.update(hot, 0.0)

    cool = make_snapshot(captured=1.0, external=[(4242, 100 * MIB)])
    assert tracker.update(cool, 1.0).state is PriorityState.BUSY
    assert tracker.update(cool, 5.0).state is PriorityState.BUSY
    d = tracker.update(cool, 11.5)
    assert d.state is PriorityState.CLEAR
    assert d.reason is Reason.OK


def test_release_clock_resets_when_signals_return():
    """R04: an unconfirmed clear must not accumulate across a fresh hot sample."""
    tracker = ExternalWorkloadTracker(ew_config(enter_seconds=0.0, release_seconds=10.0))
    tracker.update(make_snapshot(captured=0.0, external=[(4242, 2048 * MIB)]), 0.0)
    cool = make_snapshot(captured=1.0, external=[(4242, 100 * MIB)])
    tracker.update(cool, 1.0)
    tracker.update(cool, 6.0)  # 5s into the window
    tracker.update(make_snapshot(captured=7.0, external=[(4242, 2048 * MIB)]), 7.0)
    d = tracker.update(cool, 16.0)  # only 0s of the new window
    assert d.state is PriorityState.BUSY
    assert "releasing" in d.blockers


# --------------------------------------------------------------------------- #
# Sparse activity semantics (R04, T03)
# --------------------------------------------------------------------------- #


def test_sparse_activity_does_not_manufacture_a_trigger():
    """R04/T03: missing samples are not evidence of activity."""
    tracker = ExternalWorkloadTracker(ew_config(enter_seconds=0.0))
    snap = make_snapshot(
        captured=0.0,
        external=[(4242, 64 * MIB)],
        activity_status=ActivityStatus.SPARSE,
        utilization=0,
    )
    d = tracker.update(snap, 0.0)
    assert d.state is PriorityState.CLEAR
    assert d.detail["activity_status"] == "sparse"


def test_unsupported_activity_is_distinct_from_sparse_and_error():
    """R04/T03: unsupported / sparse / error must not collapse into one behaviour."""
    for status, expected in (
        (ActivityStatus.SPARSE, "sparse"),
        (ActivityStatus.UNSUPPORTED, "unsupported"),
        (ActivityStatus.ERROR, "error"),
    ):
        tracker = ExternalWorkloadTracker(ew_config(enter_seconds=0.0))
        snap = make_snapshot(
            captured=0.0, external=[(4242, 64 * MIB)], activity_status=status
        )
        d = tracker.update(snap, 0.0)
        assert d.state is PriorityState.CLEAR, f"{status} must not assert a veto"
        assert d.detail["activity_status"] == expected


def test_latched_activity_survives_sparse_samples_until_quiescence():
    """R04: a latched activity veto is not cleared by samples disappearing."""
    tracker = ExternalWorkloadTracker(ew_config(enter_seconds=0.0, release_seconds=1.0))
    busy = make_snapshot(
        captured=0.0,
        external=[(4242, 64 * MIB)],
        activity_status=ActivityStatus.AVAILABLE,
        activity=[(4242, 90)],
    )
    assert tracker.update(busy, 0.0).state is PriorityState.BUSY

    # Memory is clear and samples vanish entirely. The latch must hold.
    sparse = make_snapshot(
        captured=1.0,
        external=[(4242, 16 * MIB)],
        activity_status=ActivityStatus.SPARSE,
    )
    for t in (1.0, 5.0, 50.0):
        d = tracker.update(sparse, t)
        assert d.state is PriorityState.BUSY, "sparse samples must not clear an activity latch"
        assert "activity_latch_unresolved" in d.blockers

    # Quiescent path releases the *latch*. The release window itself still has to
    # elapse with the signals proven below their lower thresholds (R04).
    tracker.clear_activity_latch_after_quiescence()
    d = tracker.update(sparse, 60.0)
    assert d.state is PriorityState.BUSY, "the release window has not elapsed yet"
    assert "releasing" in d.blockers
    d = tracker.update(sparse, 61.5)
    assert d.state is PriorityState.CLEAR


def test_unknown_sample_does_not_advance_clear_timer():
    """R04: an unknown sample must not be treated as a quiet one."""
    tracker = ExternalWorkloadTracker(ew_config(enter_seconds=0.0, release_seconds=10.0))
    tracker.update(make_snapshot(captured=0.0, external=[(4242, 2048 * MIB)]), 0.0)
    unknown = make_snapshot(captured=1.0, external=[(4242, 100 * MIB)], valid=False)
    tracker.update(unknown, 1.0)
    tracker.update(unknown, 30.0)
    d = tracker.update(unknown, 40.0)
    assert d.state is PriorityState.BUSY
    assert d.reason is Reason.TELEMETRY_UNAVAILABLE


def test_invalid_snapshot_never_clears_and_holds_admission():
    """R02/R04: essential telemetry missing fails closed for new work."""
    tracker = ExternalWorkloadTracker(ew_config())
    d = tracker.update(make_snapshot(captured=0.0, valid=False), 0.0)
    assert d.holds_admission is True
    assert d.reason is Reason.TELEMETRY_UNAVAILABLE


def test_manual_pause_reports_its_own_state():
    """R11: MANUAL_PAUSE is a distinct policy state."""
    tracker = ExternalWorkloadTracker(ew_config())
    tracker.manual_pause = True
    d = tracker.update(make_snapshot(captured=0.0), 0.0)
    assert d.state is PriorityState.MANUAL_PAUSE
    assert d.reason is Reason.ORCHESTRATOR_PAUSED


# --------------------------------------------------------------------------- #
# Cold-load quietness (R05, T04)
# --------------------------------------------------------------------------- #


def test_quiet_window_requires_continuous_duration():
    """R05: a cold load needs the full quiet window, not one quiet sample."""
    tracker = QuietWindowTracker(cold_config(quiet_seconds=10.0))
    snap = make_snapshot(captured=0.0, utilization=1, external=[(541647, 345 * MIB)])
    assert tracker.update(snap, 0.0).quiet is False
    assert tracker.update(snap, 9.0).quiet is False
    assert tracker.update(snap, 10.5).quiet is True


def test_external_vram_growth_resets_the_quiet_window():
    """R05/T04: growth across the window resets it — catches pre-load staging."""
    tracker = QuietWindowTracker(
        cold_config(quiet_seconds=10.0, max_external_vram_growth_bytes=128 * MIB)
    )
    quiet = make_snapshot(captured=0.0, utilization=0, external=[(1, 100 * MIB)])
    tracker.update(quiet, 0.0)
    tracker.update(quiet, 9.0)

    grew = make_snapshot(captured=10.0, utilization=0, external=[(1, 600 * MIB)])
    d = tracker.update(grew, 10.0)
    assert d.quiet is False
    assert d.blocker is Reason.GPU_NOT_QUIET
    assert any("external_vram_growth" in b for b in d.blockers)


def test_device_utilization_above_threshold_blocks_quietness():
    """R05: device activity must stay below the cold threshold throughout."""
    tracker = QuietWindowTracker(cold_config(quiet_seconds=1.0))
    busy = make_snapshot(captured=0.0, utilization=80, external=[])
    d = tracker.update(busy, 0.0)
    assert d.quiet is False
    assert any("device_utilization" in b for b in d.blockers)


def test_quiet_tracker_read_is_side_effect_free():
    """R08/T10: a query must not extend the quiet window."""
    tracker = QuietWindowTracker(cold_config(quiet_seconds=10.0))
    snap = make_snapshot(captured=0.0, utilization=0, external=[])
    tracker.update(snap, 0.0)
    # Repeated reads at a late timestamp must not change the recorded window.
    assert tracker.satisfied_at(100.0) is True
    assert tracker.satisfied_at(1.0) is False
    assert tracker.satisfied_seconds() == 0.0


# --------------------------------------------------------------------------- #
# Capacity (R05, R06, T05)
# --------------------------------------------------------------------------- #


def test_cold_need_uses_max_of_load_peak_and_resident_plus_request():
    """R05: cold_need = max(load_peak, resident + request_peak) + reserve."""
    profile = VramProfile(
        resident_delta_bytes=8500 * MIB,
        load_peak_delta_bytes=9000 * MIB,
        request_peak_extra_bytes=700 * MIB,
        reserve_bytes=512 * MIB,
        calibration_id="cal-1",
    )
    # resident + request = 9200 > load_peak 9000, so the sum wins.
    assert profile.cold_need_bytes() == 9200 * MIB + 512 * MIB
    assert profile.warm_need_bytes() == 700 * MIB + 512 * MIB
    assert profile.complete is True


def test_uncalibrated_profile_is_not_zero():
    """R05/T05: an uncalibrated profile must block, not silently become 0."""
    profile = VramProfile(None, None, None, None, None)
    assert profile.complete is False
    assert profile.cold_need_bytes() is None
    snap = make_snapshot(captured=0.0)
    decision = evaluate_capacity(snap, profile.cold_need_bytes(), context="cold")
    assert decision.ok is False
    assert decision.reason is Reason.UNSUPPORTED_PROFILE


def test_capacity_uses_actual_free_bytes_and_reports_shortfall():
    """R05/T05: a denied budget reports the shortfall, not just 'waiting'."""
    snap = make_snapshot(captured=0.0, free=5000 * MIB)
    decision = evaluate_capacity(snap, 9000 * MIB, context="cold")
    assert decision.ok is False
    assert decision.reason is Reason.INSUFFICIENT_VRAM
    assert decision.margin_bytes == -4000 * MIB
    assert "short by" in decision.blockers[0]


def test_capacity_ignores_total_minus_used():
    """R05: free bytes are the authority; total-used is never substituted."""
    snap = make_snapshot(captured=0.0, total=12227 * MIB, used=1000 * MIB, free=10513 * MIB)
    # need 10500: fits under `free` (10513), and total-used would also fit — but if the
    # implementation used total-used while free were smaller, it would wrongly admit.
    assert evaluate_capacity(snap, 10500 * MIB, context="cold").ok is True

    tight = make_snapshot(captured=0.0, total=12227 * MIB, used=1000 * MIB, free=8000 * MIB)
    assert evaluate_capacity(tight, 10500 * MIB, context="cold").ok is False


# --------------------------------------------------------------------------- #
# Coverage honesty (R04, R11)
# --------------------------------------------------------------------------- #


def test_coverage_does_not_claim_activity_without_samples():
    """R04/T03: memory-only capability must not be reported as full coverage."""
    snap = make_snapshot(captured=0.0, activity_status=ActivityStatus.SPARSE)
    cov = derive_coverage(snap, ever_sampled_activity=False)
    assert cov.full_activity_coverage is False
    assert cov.as_dict()["claims_full_activity_coverage"] is False


def test_coverage_claims_activity_only_after_real_samples():
    snap = make_snapshot(
        captured=0.0,
        activity_status=ActivityStatus.AVAILABLE,
        activity=[(1, 50)],
    )
    cov = derive_coverage(snap, ever_sampled_activity=True)
    assert cov.full_activity_coverage is True


# --------------------------------------------------------------------------- #
# Config validation (R14, T01)
# --------------------------------------------------------------------------- #


def test_release_below_enter_is_enforced_by_config():
    """R14/T01: release < entry is a configuration error, not a runtime surprise."""
    with pytest.raises(Exception):
        OrchestratorConfig(external_workload={"process_vram_enter_mib": 100, "process_vram_release_mib": 200})


def test_enablement_requires_calibration_and_reserve():
    """R14/T01: enabling without a calibration or reserve is refused."""
    cfg = OrchestratorConfig(enabled=True, device_uuid="GPU-x")
    problems = enablement_errors(cfg)
    assert any("resident_delta_mib" in p for p in problems)
    assert any("reserve_mib" in p for p in problems)
    assert any("calibration_id" in p for p in problems)


def test_enablement_requires_the_calibrated_envelope():
    """R14/R05: an unbound envelope cannot be enabled — the calibration would be
    silently invalidated by model.use_as_default or the model folder's overrides."""
    base = dict(
        enabled=True,
        device_uuid="GPU-x",
        model={
            "name": "m",
            "resident_delta_mib": 10,
            "load_peak_delta_mib": 10,
            "request_peak_extra_mib": 1,
            "calibration_id": "c",
        },
        vram={"reserve_mib": 1},
    )
    problems = enablement_errors(OrchestratorConfig(**base))
    for key in ("max_seq_len", "cache_size", "cache_mode", "chunk_size", "max_batch_size"):
        assert any(f"orchestrator.model.{key}" in p for p in problems), key

    # Completing the envelope clears exactly those problems.
    base["model"] = {**base["model"], **{
        "max_seq_len": 4096, "cache_size": 4096, "cache_mode": "FP16",
        "chunk_size": 2048, "max_batch_size": 1,
    }}
    remaining = enablement_errors(OrchestratorConfig(**base))
    assert not any("envelope" in p or "orchestrator.model.max_seq_len" in p for p in remaining)


def test_disabled_mode_has_no_enablement_errors():
    """R01/T01: disabled mode must never be blocked by orchestration validation."""
    assert enablement_errors(OrchestratorConfig(enabled=False)) == []


def test_wait_mode_is_now_a_valid_configuration():
    """R09: 'wait' mode became a supported, validated admission mode in M3.

    The M1 gate ("refused until M3") is retired: M3 implements the FIFO wait
    list, so the config now validates and enablement no longer rejects it. The
    *behavioural* wait-mode gates (FIFO, deadline, overflow, cleanup) live in
    test_lifecycle.py's wait-list section.
    """
    cfg = OrchestratorConfig(
        enabled=True,
        device_uuid="GPU-x",
        model={
            "name": "m",
            "resident_delta_mib": 10,
            "load_peak_delta_mib": 10,
            "request_peak_extra_mib": 1,
            "calibration_id": "c",
            "max_seq_len": 4096,
            "cache_size": 4096,
            "cache_mode": "FP16",
            "chunk_size": 2048,
            "max_batch_size": 1,
        },
        vram={"reserve_mib": 1},
        admission={"mode": "wait"},
    )
    assert enablement_errors(cfg) == [], "wait mode is a fully supported mode now"


def test_policy_config_validation_flags_ordering():
    """R14: the policy-level config validator mirrors the model validator."""
    errors = ew_config(process_vram_release_bytes=9999 * MIB).validate()
    assert any("process_vram_release_mib" in e for e in errors)
    assert cold_config(max_device_utilization_percent=101).validate()


def test_shipped_defaults_match_the_committed_r04_calibration():
    """M4.2 / R04: the shipped threshold defaults ARE the committed calibration.

    The M4.13 review found the calibration numbers living only in the workspace
    artefact and the WORKLOG prose: the code still carried the pre-calibration
    candidates, so every "calibrated" acceptance run was measured against
    thresholds nobody had validated, and nothing in the suite could notice.

    This binds the two authorities. It reads the calibration artefact from the
    development workspace (``<workspace>/evidence/m4-calibration/
    r04-threshold-aggregate-v2.json``, i.e. two levels above this checkout) rather
    than embedding host measurements in the published tree, and skips with a stated
    reason when the workspace is absent — a skip here means "no authority to
    compare against", never "the numbers are fine".
    """

    import json
    import pathlib

    # <workspace>/fork/tabbyAPI/tests/orchestration/test_policy.py
    workspace = pathlib.Path(__file__).resolve().parents[4]
    artefact = workspace / "evidence" / "m4-calibration" / "r04-threshold-aggregate-v2.json"
    if not artefact.is_file():
        pytest.skip(f"no calibration artefact at {artefact}; nothing to bind against")

    recommended = json.loads(artefact.read_text())["recommended_thresholds"]
    shipped = OrchestratorConfig().external_workload

    pairs = {
        "process_vram_enter_mib": shipped.process_vram_enter_mib,
        "process_vram_release_mib": shipped.process_vram_release_mib,
        "total_vram_enter_mib": shipped.total_vram_enter_mib,
        "total_vram_release_mib": shipped.total_vram_release_mib,
        "process_activity_enter_percent": shipped.process_activity_enter_percent,
        "process_activity_release_percent": shipped.process_activity_release_percent,
    }
    mismatched = {
        key: (value, recommended[key]) for key, value in pairs.items() if value != recommended[key]
    }
    assert not mismatched, (
        "shipped defaults disagree with the committed R04 calibration "
        f"(shipped, committed): {mismatched}"
    )
