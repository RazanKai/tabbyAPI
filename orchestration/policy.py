"""Pure decision policy over telemetry snapshots.

Everything here is a pure function of the inputs: no I/O, no clock reads, no locks,
no mutation of process state. That is what makes the hysteresis windows and the
sparse-activity rules testable with injected clocks and recorded traces (SPEC
section 10) instead of only on hardware.

The policy answers four distinct questions, and keeping them distinct is the point
(SPEC R04/R05/R11):

1. **External priority** — is a heavy external workload running, so a resident model
   should drain? (``evaluate_external_workload``)
2. **Cold-load quietness** — is the device quiet enough to start a *new* load?
   (``evaluate_cold_load_quietness``) — a deliberately *different* test. A moderate
   device load may block a new cold load without forcing a resident model to drain.
3. **Capacity** — is there actually enough free memory for the profile?
   (``evaluate_capacity`` — raw NVML free bytes only)
4. **Activity coverage** — which signals can be trusted at all? (``coverage``)

Unknown never becomes zero. A latched veto survives sparse samples; a missing
sample never manufactures a trigger; and no percentage arithmetic is ever used to
estimate foreign utilization (concurrent GPU activity is not an additive ledger).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from .telemetry import ActivityStatus, Snapshot

MIB = 1024 * 1024


class PriorityState(str, Enum):
    """External-priority policy state (SPEC R11).

    ``CLEAR`` means "no external-priority veto" — it is *not* permission to load or
    generate, which is why admission consults several independent booleans.
    """

    UNKNOWN = "UNKNOWN"
    CLEAR = "CLEAR"
    CANDIDATE = "CANDIDATE"
    BUSY = "BUSY"
    MANUAL_PAUSE = "MANUAL_PAUSE"


class Reason(str, Enum):
    """Stable machine-readable blocker/reason codes (SPEC R12)."""

    OK = "ok"
    EXTERNAL_GPU_BUSY = "external_gpu_busy"
    GPU_NOT_QUIET = "gpu_not_quiet"
    INSUFFICIENT_VRAM = "insufficient_vram"
    TELEMETRY_UNAVAILABLE = "telemetry_unavailable"
    ORCHESTRATOR_PAUSED = "orchestrator_paused"
    ORCHESTRATOR_FAULT = "orchestrator_fault"
    MODEL_TRANSITION = "model_transition"
    REQUEST_CAPACITY = "request_capacity"
    ADMISSION_TIMEOUT = "admission_timeout"
    MODEL_NOT_CONFIGURED = "model_not_configured"
    UNSUPPORTED_PROFILE = "unsupported_profile"
    ADMISSION_QUEUE_FULL = "admission_queue_full"
    #: Internal drain reason for an operator-requested unload. Deliberately NOT
    #: ORCHESTRATOR_PAUSED: ``resume()`` clears only the manual pause, so reusing
    #: the pause code would let a resume cancel the admin's unload demand and
    #: leave the coordinator wedged in DRAINING with no pending drain.
    EXPLICIT_UNLOAD = "explicit_unload"


@dataclass(frozen=True)
class ExternalWorkloadConfig:
    """Thresholds and windows for the external-workload priority policy.

    These are configuration, not measurements. SPEC section 7 makes calibration a
    release gate: the values below are the *candidate* settings the spec records,
    and they are only meaningful once M0 has produced a same-metric desktop trace
    to set the release side against.
    """

    process_vram_enter_bytes: int
    process_vram_release_bytes: int
    total_vram_enter_bytes: int
    total_vram_release_bytes: int
    process_activity_enter_percent: int
    process_activity_release_percent: int
    enter_seconds: float
    release_seconds: float

    def validate(self) -> list[str]:
        """Structural validation. Release must be strictly below entry (R14)."""
        errors: list[str] = []
        if self.process_vram_release_bytes >= self.process_vram_enter_bytes:
            errors.append("external_workload.process_vram_release_mib must be < process_vram_enter_mib")
        if self.total_vram_release_bytes >= self.total_vram_enter_bytes:
            errors.append("external_workload.total_vram_release_mib must be < total_vram_enter_mib")
        if self.process_activity_release_percent >= self.process_activity_enter_percent:
            errors.append(
                "external_workload.process_activity_release_percent must be < process_activity_enter_percent"
            )
        if self.enter_seconds < 0 or self.release_seconds < 0:
            errors.append("external_workload windows must be non-negative")
        return errors


@dataclass(frozen=True)
class ColdLoadConfig:
    max_device_utilization_percent: int
    quiet_seconds: float
    max_external_vram_growth_bytes: int

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not 0 <= self.max_device_utilization_percent <= 100:
            errors.append("cold_load.max_device_utilization_percent must be within 0..100")
        if self.quiet_seconds < 0:
            errors.append("cold_load.quiet_seconds must be non-negative")
        if self.max_external_vram_growth_bytes < 0:
            errors.append("cold_load.max_external_vram_growth_mib must be non-negative")
        return errors


@dataclass(frozen=True)
class VramProfile:
    """Calibrated incremental GPU footprint of one model/profile (SPEC R05).

    Values are *incremental above the same running-but-unloaded server baseline*,
    in bytes. ``None`` means "not yet calibrated", which must block enablement
    rather than being treated as zero.
    """

    resident_delta_bytes: Optional[int]
    load_peak_delta_bytes: Optional[int]
    request_peak_extra_bytes: Optional[int]
    reserve_bytes: Optional[int]
    calibration_id: Optional[str]
    #: Calibrated load envelope (R14): the load is bound to these keys and the
    #: real container is verified against them after every load.
    envelope: Optional[dict[str, Any]] = None

    @property
    def complete(self) -> bool:
        return (
            self.resident_delta_bytes is not None
            and self.load_peak_delta_bytes is not None
            and self.request_peak_extra_bytes is not None
            and self.reserve_bytes is not None
            and bool(self.calibration_id)
        )

    def cold_need_bytes(self) -> Optional[int]:
        """``max(load_peak, resident + request_peak_extra) + reserve`` (SPEC R05)."""
        if not self.complete:
            return None
        load_peak = self.load_peak_delta_bytes
        resident = self.resident_delta_bytes
        request_extra = self.request_peak_extra_bytes
        reserve = self.reserve_bytes
        assert load_peak is not None and resident is not None
        assert request_extra is not None and reserve is not None
        return max(load_peak, resident + request_extra) + reserve

    def warm_need_bytes(self) -> Optional[int]:
        """Warm admission never charges the full cold footprint again (R06)."""
        if not self.complete:
            return None
        request_extra = self.request_peak_extra_bytes
        reserve = self.reserve_bytes
        assert request_extra is not None and reserve is not None
        return request_extra + reserve


@dataclass(frozen=True)
class DetectorCoverage:
    """What the detector can honestly claim (R04)."""

    activity_status: ActivityStatus
    memory_enumeration_ok: bool
    graphics_rows_observed: bool
    activity_samples_observed: bool

    @property
    def memory_only(self) -> bool:
        return self.activity_status in (ActivityStatus.UNSUPPORTED, ActivityStatus.ERROR)

    @property
    def full_activity_coverage(self) -> bool:
        """Only a real sample stream may claim activity detection."""
        return (
            self.activity_status is ActivityStatus.AVAILABLE
            and self.activity_samples_observed
        )

    def as_dict(self) -> dict:
        return {
            "activity_status": self.activity_status.value,
            "memory_enumeration_ok": self.memory_enumeration_ok,
            "graphics_rows_observed": self.graphics_rows_observed,
            "activity_samples_observed": self.activity_samples_observed,
            "claims_full_activity_coverage": self.full_activity_coverage,
            "memory_only_mode": self.memory_only,
        }


@dataclass(frozen=True)
class ExternalWorkloadDecision:
    state: PriorityState
    reason: Reason
    trigger: Optional[str] = None
    #: True while a candidate is building but has not yet met enter_seconds. The
    #: caller holds *new admission* but must not unload a resident model yet (R04).
    holds_admission: bool = False
    blockers: tuple[str, ...] = ()
    detail: dict = field(default_factory=dict)


@dataclass(frozen=True)
class QuietnessDecision:
    quiet: bool
    quiet_seconds_observed: float
    blocker: Optional[Reason]
    blockers: tuple[str, ...] = ()
    detail: dict = field(default_factory=dict)


@dataclass(frozen=True)
class CapacityDecision:
    ok: bool
    need_bytes: Optional[int]
    free_bytes: Optional[int]
    margin_bytes: Optional[int]
    reason: Reason
    blockers: tuple[str, ...] = ()


class ExternalWorkloadTracker:
    """Hysteresis state machine for the external-workload veto.

    Held by the coordinator and fed one snapshot at a time, always under its mutex.
    Kept separate from the pure evaluation function so that the *windows* are state
    while the *decisions* stay pure.

    Rules implemented (SPEC R04):
    * OR policy across triggers — memory thresholds cover large-but-idle workloads,
      activity thresholds cover smaller compute-heavy ones.
    * Entry needs continuous qualifying evidence for ``enter_seconds``.
    * Release needs ALL configured signals below their lower thresholds for
      ``release_seconds``.
    * An unknown sample does not advance the clear timer.
    * Sparse activity does not manufacture a trigger and does not by itself clear an
      activity latch; clearing an activity latch requires the quiescent path.
    """

    def __init__(self, config: ExternalWorkloadConfig):
        self.config = config
        self.state = PriorityState.UNKNOWN
        self._enter_since: Optional[float] = None
        self._release_since: Optional[float] = None
        self._latched_activity: bool = False
        #: External PIDs whose samples are currently holding the activity latch.
        #: Tracked per process so a single exit clears only that contribution (R04).
        self._activity_pids: set[int] = set()
        self._trigger: Optional[str] = None
        self._detail: dict = {}
        #: A manual pause is a separate veto; the tracker reports it but does not own it.
        self.manual_pause = False

    def reset(self) -> None:
        self.state = PriorityState.UNKNOWN
        self._enter_since = None
        self._release_since = None
        self._latched_activity = False
        self._activity_pids = set()
        self._trigger = None
        self._detail = {}

    # -- helpers ------------------------------------------------------------ #

    def _memory_qualifies(self, snap: Snapshot) -> tuple[bool, dict]:
        enter = self.config.process_vram_enter_bytes
        total_enter = self.config.total_vram_enter_bytes
        largest_ok = snap.external_largest_bytes >= enter
        total_ok = snap.external_total_bytes >= total_enter
        return (
            largest_ok or total_ok,
            {
                "external_largest_bytes": snap.external_largest_bytes,
                "external_total_bytes": snap.external_total_bytes,
                "process_vram_enter_bytes": enter,
                "total_vram_enter_bytes": total_enter,
                "largest_over": largest_ok,
                "total_over": total_ok,
            },
        )

    def _activity_qualifies(self, snap: Snapshot) -> tuple[bool, dict, bool, set[int]]:
        """Return ``(qualifies, detail, unknown, hot_pids)``.

        ``unknown`` is True when the activity signal could not be read at all, which
        must not be interpreted either way. ``hot_pids`` names the external PIDs that
        supplied the qualifying samples, so the latch can be held *per contributing
        process* and one process exiting does not clear another's veto (R04).
        """
        detail = {
            "activity_status": snap.activity_status.value,
            "process_activity_enter_percent": self.config.process_activity_enter_percent,
            "max_external_sm_percent": None,
        }
        if snap.activity_status is ActivityStatus.AVAILABLE:
            external_pids = {p.pid for p in snap.processes if not p.is_owned}
            best = None
            hot_pids: set[int] = set()
            for act in snap.activity:
                if act.pid in external_pids and act.sm_percent is not None:
                    if act.sm_percent >= self.config.process_activity_enter_percent:
                        hot_pids.add(act.pid)
                    best = act.sm_percent if best is None else max(best, act.sm_percent)
            detail["max_external_sm_percent"] = best
            if best is None:
                # Samples exist but none belong to an external process. That is not
                # evidence of external activity, and not evidence against it either.
                return False, detail, True, set()
            return best >= self.config.process_activity_enter_percent, detail, False, hot_pids
        if snap.activity_status is ActivityStatus.SPARSE or snap.activity_status is ActivityStatus.UNSUPPORTED:
            # Never manufacture a trigger from absence (R04).
            return False, detail, True, set()
        return False, detail, True, set()

    def _memory_clear(self, snap: Snapshot) -> bool:
        return (
            snap.external_largest_bytes < self.config.process_vram_release_bytes
            and snap.external_total_bytes < self.config.total_vram_release_bytes
        )

    def _activity_clear(self, snap: Snapshot) -> bool:
        if snap.activity_status is not ActivityStatus.AVAILABLE:
            return False  # unknown samples do not advance a clear timer
        external_pids = {p.pid for p in snap.processes if not p.is_owned}
        values = [
            a.sm_percent
            for a in snap.activity
            if a.pid in external_pids and a.sm_percent is not None
        ]
        if not values:
            return False
        return max(values) < self.config.process_activity_release_percent

    # -- the state machine -------------------------------------------------- #

    def update(self, snap: Snapshot, now_monotonic: float) -> ExternalWorkloadDecision:
        if self.manual_pause:
            return ExternalWorkloadDecision(
                state=PriorityState.MANUAL_PAUSE,
                reason=Reason.ORCHESTRATOR_PAUSED,
                trigger="manual_pause",
            )

        if not snap.valid:
            # Essential telemetry is missing: keep whatever we believed, hold new
            # admission, and say why. Never fall through to CLEAR on bad data.
            return ExternalWorkloadDecision(
                state=self.state if self.state is not PriorityState.UNKNOWN else PriorityState.UNKNOWN,
                reason=Reason.TELEMETRY_UNAVAILABLE,
                trigger=self._trigger,
                holds_admission=True,
                blockers=("telemetry_invalid",),
                detail={"valid": False, "ownership_error": snap.ownership_error},
            )

        mem_hot, mem_detail = self._memory_qualifies(snap)
        act_hot, act_detail, act_unknown, hot_pids = self._activity_qualifies(snap)
        detail = {**mem_detail, **act_detail, "activity_unknown": act_unknown}

        qualifying = mem_hot or act_hot

        if qualifying:
            self._release_since = None
            if self._enter_since is None:
                self._enter_since = now_monotonic
            held = now_monotonic - self._enter_since
            # An activity sighting is latched even before it is confirmed, so a
            # burst that then goes quiet still requires the quiescent path to clear.
            # The contributing PIDs are recorded so the latch survives until *all*
            # of them are gone (one exit must clear only that PID's contribution).
            if act_hot:
                self._latched_activity = True
                self._activity_pids |= hot_pids
            if held >= self.config.enter_seconds:
                self.state = PriorityState.BUSY
                self._trigger = "activity" if act_hot else "memory"
                self._detail = detail
                return ExternalWorkloadDecision(
                    state=PriorityState.BUSY,
                    reason=Reason.EXTERNAL_GPU_BUSY,
                    trigger=self._trigger,
                    holds_admission=True,
                    blockers=("external_workload_sustained",),
                    detail=detail,
                )
            self.state = PriorityState.CANDIDATE
            self._trigger = "activity" if act_hot else "memory"
            self._detail = detail
            return ExternalWorkloadDecision(
                state=PriorityState.CANDIDATE,
                reason=Reason.EXTERNAL_GPU_BUSY,
                trigger=self._trigger,
                # Candidate holds new admission but does NOT unload a resident model.
                holds_admission=True,
                blockers=("external_workload_candidate",),
                detail=detail,
            )

        # Not qualifying this instant. Can we clear?
        self._enter_since = None

        # A release *window* only applies to a veto that was actually asserted or
        # building. If nothing was ever asserted there is nothing to release, and
        # requiring a release window here would make the very first clear snapshot
        # report BUSY — blocking all admission on a quiet device.
        prior_asserted = self.state in (PriorityState.BUSY, PriorityState.CANDIDATE)
        if not prior_asserted:
            self.state = PriorityState.CLEAR
            self._release_since = None
            self._trigger = None
            self._detail = detail
            return ExternalWorkloadDecision(
                state=PriorityState.CLEAR,
                reason=Reason.OK,
                detail=detail,
            )

        # Memory must clear independently of activity, and an unsettled activity
        # latch cannot be cleared by absent samples.
        memory_clear = self._memory_clear(snap)
        activity_clear = self._activity_clear(snap)
        activity_resolved = activity_clear or (not self._latched_activity)
        if memory_clear and activity_resolved:
            if self._release_since is None:
                self._release_since = now_monotonic
            held = now_monotonic - self._release_since
            if held >= self.config.release_seconds:
                self.state = PriorityState.CLEAR
                self._latched_activity = False
                self._trigger = None
                self._detail = detail
                return ExternalWorkloadDecision(
                    state=PriorityState.CLEAR,
                    reason=Reason.OK,
                    detail={**detail, "release_seconds_observed": held},
                )
            # Still latched through the release window: preserve the prior state.
            prior = self.state if self.state in (PriorityState.BUSY, PriorityState.CANDIDATE) else PriorityState.BUSY
            self.state = prior
            self._detail = detail
            return ExternalWorkloadDecision(
                state=prior,
                reason=Reason.EXTERNAL_GPU_BUSY if prior is PriorityState.BUSY else Reason.EXTERNAL_GPU_BUSY,
                trigger=self._trigger,
                holds_admission=True,
                blockers=("releasing",),
                detail={**detail, "release_seconds_observed": held},
            )

        # Something is still hot, or an activity latch is unresolved.
        self._release_since = None
        if not memory_clear and self.state in (PriorityState.BUSY, PriorityState.CANDIDATE):
            self.state = PriorityState.BUSY
            self._detail = detail
            return ExternalWorkloadDecision(
                state=PriorityState.BUSY,
                reason=Reason.EXTERNAL_GPU_BUSY,
                trigger=self._trigger or "memory",
                holds_admission=True,
                blockers=("external_memory_still_high",),
                detail=detail,
            )
        if self._latched_activity and not activity_clear:
            self.state = PriorityState.BUSY
            self._detail = detail
            return ExternalWorkloadDecision(
                state=PriorityState.BUSY,
                reason=Reason.EXTERNAL_GPU_BUSY,
                trigger=self._trigger or "activity",
                holds_admission=True,
                blockers=("activity_latch_unresolved",),
                detail=detail,
            )
        # Nothing was hot and nothing was latched: clear.
        self.state = PriorityState.CLEAR
        self._trigger = None
        self._detail = detail
        return ExternalWorkloadDecision(
            state=PriorityState.CLEAR,
            reason=Reason.OK,
            detail=detail,
        )

    def note_external_pid_exit(self, pid: int) -> None:
        """A blocking PID exited — drops only that PID's contribution (R04).

        The activity latch is held per contributing process (``_activity_pids``), so
        one process exiting must not clear another's veto. The next ``update``
        re-derives reality from the snapshot; memory triggers still clear on their
        own, which is why this cannot clear the whole veto.
        """
        self._activity_pids.discard(pid)
        if not self._activity_pids:
            self._latched_activity = False

    def held_by_pids(self, live_pids: set[int]) -> set[int]:
        """Contributing PIDs that are still live — the latch's remaining holders.

        Reported in decision detail so an operator can see *why* an activity veto is
        still asserted while its samples are absent.
        """
        return {pid for pid in self._activity_pids if pid in live_pids}

    def clear_activity_latch_after_quiescence(self) -> None:
        """Called by the coordinator after a verified device-quiet window.

        This is the normal resolution path for a latched activity veto whose
        samples disappeared (R04): drain/unload -> no managed GPU work -> observed
        device quietness for the release window -> latch release. ``update`` alone
        can never take this path (absent samples must not clear a latch), so without
        this call a sustained activity veto with no further samples would hold
        admission forever.
        """
        self._latched_activity = False
        self._activity_pids = set()

    @property
    def unresolved_activity_latch(self) -> bool:
        """Whether an activity veto is latched but not yet proven resolved (R04).

        The coordinator consults this to decide whether a quiescent window may
        release the latch; ``update`` itself never clears it from absent samples.
        """
        return self._latched_activity

    @property
    def detail(self) -> dict:
        return dict(self._detail)


# --------------------------------------------------------------------------- #
# Cold-load quietness: a separate question from external priority
# --------------------------------------------------------------------------- #


class QuietWindowTracker:
    """Tracks a continuous quiet/stable window for cold-load admission (R05).

    Quietness requires *all* of:
    * device activity below the cold threshold, throughout;
    * external process memory not rising by more than the tolerance, across the
      window (growth resets the clock);
    * valid essential telemetry.

    A growth reset is intentional: it is what catches a launcher staging assets or a
    game allocating memory *before* it shows up as a heavy consumer.

    Only :meth:`update` advances the window, and it is called once per sample by the
    sampler. :meth:`satisfied_at` is a pure read so that a status poll can never
    extend or reset the window (R08/T10) — a query must not create model demand.
    """

    def __init__(self, config: ColdLoadConfig):
        self.config = config
        self._since: Optional[float] = None
        self._baseline_external_bytes: Optional[int] = None
        self._reset_reason: Optional[str] = None
        self._last_observed_seconds: float = 0.0

    def reset(self, reason: Optional[str] = None) -> None:
        self._since = None
        self._baseline_external_bytes = None
        self._reset_reason = reason
        self._last_observed_seconds = 0.0

    def satisfied_at(self, now_monotonic: float) -> bool:
        """Whether the last observed quiet window already meets the requirement."""
        if self._since is None:
            return False
        return (now_monotonic - self._since) >= self.config.quiet_seconds

    def satisfied_seconds(self) -> float:
        """Length of the last observed quiet window as measured by ``update``."""
        return self._last_observed_seconds

    def update(self, snap: Snapshot, now_monotonic: float) -> QuietnessDecision:
        blockers: list[str] = []

        if not snap.valid:
            self.reset("telemetry_invalid")
            return QuietnessDecision(
                quiet=False,
                quiet_seconds_observed=0.0,
                blocker=Reason.TELEMETRY_UNAVAILABLE,
                blockers=("telemetry_invalid",),
                detail={"valid": False},
            )

        utilization_ok = (
            snap.device_utilization_percent is not None
            and snap.device_utilization_percent <= self.config.max_device_utilization_percent
        )
        if not utilization_ok:
            blockers.append(
                f"device_utilization {snap.device_utilization_percent}"
                f" > {self.config.max_device_utilization_percent}"
            )

        growth = 0
        if self._baseline_external_bytes is None:
            self._baseline_external_bytes = snap.external_total_bytes
        growth = snap.external_total_bytes - self._baseline_external_bytes
        growth_ok = growth <= self.config.max_external_vram_growth_bytes
        if not growth_ok:
            blockers.append(
                f"external_vram_growth {growth} > {self.config.max_external_vram_growth_bytes}"
            )

        if not utilization_ok or not growth_ok:
            self.reset("; ".join(blockers))
            return QuietnessDecision(
                quiet=False,
                quiet_seconds_observed=0.0,
                blocker=Reason.GPU_NOT_QUIET,
                blockers=tuple(blockers),
                detail={
                    "device_utilization_percent": snap.device_utilization_percent,
                    "external_growth_bytes": growth,
                },
            )

        if self._since is None:
            self._since = now_monotonic
        held = now_monotonic - self._since
        self._last_observed_seconds = held
        detail = {
            "device_utilization_percent": snap.device_utilization_percent,
            "external_growth_bytes": growth,
            "quiet_seconds_observed": held,
            "required_seconds": self.config.quiet_seconds,
        }
        if held >= self.config.quiet_seconds:
            return QuietnessDecision(
                quiet=True,
                quiet_seconds_observed=held,
                blocker=None,
                detail=detail,
            )
        return QuietnessDecision(
            quiet=False,
            quiet_seconds_observed=held,
            blocker=Reason.GPU_NOT_QUIET,
            blockers=(f"quiet_window {held:.1f}s < {self.config.quiet_seconds}s",),
            detail=detail,
        )

    @property
    def last_reset_reason(self) -> Optional[str]:
        return self._reset_reason


# --------------------------------------------------------------------------- #
# Capacity: raw NVML free bytes only
# --------------------------------------------------------------------------- #


def evaluate_capacity(
    snap: Snapshot, need_bytes: Optional[int], *, context: str
) -> CapacityDecision:
    """Decide whether ``need_bytes`` fits in *actual* free device memory.

    Deliberately uses ``snap.mem_free`` and nothing derived from it: no
    ``total - used``, and no presumed-reclaimable additions (SPEC 3.3/R05).

    Blocker strings are prefixed with the stable reason code so that an operator
    reading status sees both the machine-readable cause and the measured shortfall
    (R11: "status must name the memory threshold and current value").
    """
    if need_bytes is None:
        return CapacityDecision(
            ok=False,
            need_bytes=None,
            free_bytes=snap.mem_free,
            margin_bytes=None,
            reason=Reason.UNSUPPORTED_PROFILE,
            blockers=(
                f"{Reason.UNSUPPORTED_PROFILE.value}: {context} profile is not "
                "calibrated (footprint unknown)",
            ),
        )
    if snap.mem_free is None:
        return CapacityDecision(
            ok=False,
            need_bytes=need_bytes,
            free_bytes=None,
            margin_bytes=None,
            reason=Reason.TELEMETRY_UNAVAILABLE,
            blockers=(
                f"{Reason.TELEMETRY_UNAVAILABLE.value}: {context} device free "
                "memory unavailable",
            ),
        )
    margin = snap.mem_free - need_bytes
    if margin < 0:
        return CapacityDecision(
            ok=False,
            need_bytes=need_bytes,
            free_bytes=snap.mem_free,
            margin_bytes=margin,
            reason=Reason.INSUFFICIENT_VRAM,
            blockers=(
                f"{Reason.INSUFFICIENT_VRAM.value}: {context} short by "
                f"{(-margin) / MIB:.0f} MiB "
                f"(free {snap.mem_free / MIB:.0f} MiB, need {need_bytes / MIB:.0f} MiB)",
            ),
        )
    return CapacityDecision(
        ok=True,
        need_bytes=need_bytes,
        free_bytes=snap.mem_free,
        margin_bytes=margin,
        reason=Reason.OK,
    )


def derive_coverage(snap: Snapshot, *, ever_sampled_activity: bool) -> DetectorCoverage:
    """Describe honestly what the detector can claim from this snapshot."""
    graphics = any("graphics" in p.sources for p in snap.processes)
    return DetectorCoverage(
        activity_status=snap.activity_status,
        memory_enumeration_ok=(
            snap.mem_used is not None and snap.external_total_bytes >= 0 and snap.valid
        ),
        graphics_rows_observed=graphics,
        activity_samples_observed=ever_sampled_activity,
    )


__all__ = [
    "CapacityDecision",
    "ColdLoadConfig",
    "DetectorCoverage",
    "ExternalWorkloadConfig",
    "ExternalWorkloadDecision",
    "ExternalWorkloadTracker",
    "PriorityState",
    "QuietWindowTracker",
    "QuietnessDecision",
    "Reason",
    "VramProfile",
    "derive_coverage",
    "evaluate_capacity",
]
