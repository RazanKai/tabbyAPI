"""The single lifecycle coordinator: leases, transitions, drain and idle unload.

One instance per process. It is the only component that decides whether a request
may touch the model, and the only thing that starts an unload (SPEC section 4,
"one transition authority").

Design rules this file is held to:

* **Short mutex, long work outside it.** The coordinator's mutex guards state
  inspection and mutation only. Loading, unloading, telemetry reads and any waiting
  happen with the mutex released, then the result is reconciled back under it. A
  long operation never freezes status or pause responsiveness (T14/T19).
* **Reserve a transition, then execute.** A transition token prevents a second
  transition from starting while the first runs unlocked.
* **Leases are held until backend work really stops.** For streaming responses the
  HTTP generator closing is *not* completion (integration map section 2), so the
  lease is released only after the generation tasks are observed finished.
* **Never load without demand; never unload under active work.**
* Idle time starts at the release of the final *inference* lease. Reader pins defer
  teardown but never refresh the inference TTL (R08/T10).
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Optional

from . import policy
from .config import OrchestratorConfig
from .policy import (
    ExternalWorkloadDecision,
    ExternalWorkloadTracker,
    PriorityState,
    QuietWindowTracker,
    Reason,
    VramProfile,
    evaluate_capacity,
)
from .telemetry import ActivityStatus, Snapshot, TelemetryAdapter

MIB = 1024 * 1024


class Lifecycle(str, Enum):
    """Model lifecycle. Orthogonal to the external-priority policy state (R11)."""

    UNLOADED = "UNLOADED"
    LOADING = "LOADING"
    READY = "READY"
    DRAINING = "DRAINING"
    UNLOADING = "UNLOADING"
    FAULT = "FAULT"


class LeaseKind(str, Enum):
    """What a lease protects.

    ``INFERENCE`` is real demand and drives the idle TTL. ``READER`` protects a
    short metadata/tokenization read against teardown but explicitly does *not*
    reset the idle timer (R08).
    """

    INFERENCE = "inference"
    READER = "reader"


class AdmissionOutcome(str, Enum):
    GRANTED = "granted"
    DENIED = "denied"
    TIMEOUT = "timeout"


@dataclass
class AdmissionResult:
    outcome: AdmissionOutcome
    reason: Reason
    lease: "Lease | None" = None
    detail: dict = field(default_factory=dict)


@dataclass
class Lease:
    """An admission ticket that keeps the model resident while held."""

    lease_id: str
    kind: LeaseKind
    request_id: str
    container_identity: Optional[str]
    acquired_monotonic: float
    released: bool = False
    release_monotonic: Optional[float] = None

    @property
    def age_seconds(self) -> float:
        end = self.release_monotonic
        if end is None:
            end = time.monotonic()
        return end - self.acquired_monotonic


class LifecycleError(Exception):
    """Raised for coordinator misuse (e.g. releasing an unknown lease)."""


@dataclass
class Waiter:
    """One request on the bounded FIFO admission wait list (R09, M3).

    ``entered_monotonic`` starts the pre-lease deadline at ENQUEUE time, so all
    pre-lease time — quiet-window wait, a shared load, queue time — is bounded
    by one deadline. ``admitted`` is set exactly once; ``released`` marks the
    slot as freed (deadline, disconnect, cancellation, shutdown, or admission).
    Expiry is computed against the coordinator's injected clock, never wall
    time, so tests stay deterministic.
    """

    request_id: str
    entered_monotonic: float
    deadline_seconds: float
    event: "asyncio.Event"
    clock: Callable[[], float]
    result: Optional["AdmissionResult"] = None
    admitted: bool = False
    released: bool = False

    def expired_at(self, now: Optional[float] = None) -> bool:
        return (now if now is not None else self.clock()) - self.entered_monotonic >= self.deadline_seconds

    def remaining_seconds(self, now: Optional[float] = None) -> float:
        current = now if now is not None else self.clock()
        return max(0.0, self.deadline_seconds - (current - self.entered_monotonic))


@dataclass
class CoordinatorDeps:
    """Callbacks the coordinator needs from TabbyAPI.

    Injected rather than imported so the coordinator carries no dependency on the
    server's module graph and stays unit-testable with fakes. Production wiring
    lives in ``orchestration/install.py``.
    """

    #: Perform the model load. Must raise on failure; must not swallow cancellation.
    load_model: Callable[[], Awaitable[Any]]
    #: Perform the model unload. Must be safe to call when nothing is loaded.
    unload_model: Callable[[], Awaitable[None]]
    #: True when a usable model container exists right now.
    container_present: Callable[[], bool]
    #: True when the container still has outstanding backend work (jobs/generators).
    backend_busy: Callable[[], bool]
    #: Describe the current container for status/identity pinning.
    container_identity: Callable[[], Optional[str]]
    #: Read the container's *effective* load envelope (max_seq_len, cache_size,
    #: cache_mode, chunk_size, max_batch_size) for post-load verification against
    #: the calibration. None values mean "cannot be read here", which fails open
    #: only for that key (a fake in tests returns everything).
    container_envelope: Callable[[], dict[str, Any]]
    #: True while the backend's generator-recovery task is in flight (R07/T14).
    #: Recovery recreates the generator via ``ensure_future`` — an untracked task
    #: that cancels active jobs first, so ``backend_busy`` (the job registry) can
    #: read EMPTY while live CUDA teardown/rebuild work is under way.
    recovery_in_progress: Callable[[], bool]
    #: Adopt the backend's in-flight recovery task so the coordinator owns it
    #: through completion (R07: "retain and observe"). None/absent task is fine;
    #: the callable must never raise for a missing task.
    adopt_recovery_task: Callable[[], Optional["asyncio.Task"]]


class LifecycleCoordinator:
    """Owns load/unload/lease transitions for exactly one model."""

    def __init__(
        self,
        cfg: OrchestratorConfig,
        deps: CoordinatorDeps,
        telemetry: TelemetryAdapter,
        *,
        clock: Callable[[], float] = time.monotonic,
        log: Optional[Callable[[str], None]] = None,
    ):
        self.cfg = cfg
        self.deps = deps
        self.telemetry = telemetry
        self._clock = clock
        self._log = log or (lambda _msg: None)

        self._mutex = asyncio.Lock()
        self.lifecycle = Lifecycle.UNLOADED
        self.fault_reason: Optional[str] = None
        self.last_error: Optional[str] = None

        self._leases: dict[str, Lease] = {}
        self._transition_active = False
        self._transition_started: Optional[float] = None
        self._transition_kind: Optional[str] = None
        #: Strong reference to an in-flight load task, so a load whose requesting
        #: client disappeared is neither abandoned nor garbage-collected (R07).
        self._load_task: Optional[asyncio.Task] = None
        #: The backend's in-flight generator-recovery task, once adopted (F12/R07).
        #: Recovery is created upstream via ``asyncio.ensure_future`` and is
        #: otherwise untracked: it cancels active jobs before rebuilding the
        #: generator, so the job registry can read EMPTY mid-recovery while real
        #: backend work is outstanding. Holding a strong reference here keeps the
        #: task observable until it finishes; ``recovery_outstanding`` closes
        #: admission and defers unload while it runs.
        self._recovery_task: Optional[asyncio.Task] = None

        self._idle_since: Optional[float] = None
        self._last_completion: Optional[float] = None
        self._manual_pause = False
        self._shutting_down = False

        #: Set when a drain was requested but could not start immediately.
        self._drain_requested = False
        self._drain_reason: Optional[Reason] = None
        #: Set when an unload transition has been RESERVED but deliberately not
        #: executed inline (pause must return within T19's budget). tick() is the
        #: executor. Distinct from `_transition_active` so maintenance never
        #: double-executes a transition that a caller is already awaiting.
        self._pending_unload = False

        self._external = ExternalWorkloadTracker(
            policy.ExternalWorkloadConfig(
                process_vram_enter_bytes=cfg.external_workload.process_vram_enter_mib * MIB,
                process_vram_release_bytes=cfg.external_workload.process_vram_release_mib * MIB,
                total_vram_enter_bytes=cfg.external_workload.total_vram_enter_mib * MIB,
                total_vram_release_bytes=cfg.external_workload.total_vram_release_mib * MIB,
                process_activity_enter_percent=cfg.external_workload.process_activity_enter_percent,
                process_activity_release_percent=cfg.external_workload.process_activity_release_percent,
                enter_seconds=cfg.external_workload.enter_seconds,
                release_seconds=cfg.external_workload.release_seconds,
            )
        )
        self._quiet = QuietWindowTracker(
            policy.ColdLoadConfig(
                max_device_utilization_percent=cfg.cold_load.max_device_utilization_percent,
                quiet_seconds=cfg.cold_load.quiet_seconds,
                max_external_vram_growth_bytes=cfg.cold_load.max_external_vram_growth_mib * MIB,
            )
        )
        self.profile = VramProfile(
            resident_delta_bytes=(
                None if cfg.model.resident_delta_mib is None else cfg.model.resident_delta_mib * MIB
            ),
            load_peak_delta_bytes=(
                None if cfg.model.load_peak_delta_mib is None else cfg.model.load_peak_delta_mib * MIB
            ),
            request_peak_extra_bytes=(
                None
                if cfg.model.request_peak_extra_mib is None
                else cfg.model.request_peak_extra_mib * MIB
            ),
            reserve_bytes=(None if cfg.vram.reserve_mib is None else cfg.vram.reserve_mib * MIB),
            calibration_id=cfg.model.calibration_id,
            envelope={
                key: getattr(cfg.model, key)
                for key in ("max_seq_len", "cache_size", "cache_mode", "chunk_size", "max_batch_size")
                if getattr(cfg.model, key) is not None
            },
        )

        #: Latest snapshot and the external-workload verdict derived from it.
        self.snapshot: Optional[Snapshot] = None
        self.external_decision: ExternalWorkloadDecision = ExternalWorkloadDecision(
            state=PriorityState.UNKNOWN, reason=Reason.TELEMETRY_UNAVAILABLE
        )
        self._snapshot_taken_at: Optional[float] = None
        self._ever_sampled_activity = False
        self._own_identities: set[str] = set()

        #: Bounded FIFO admission wait list (R09, M3). Only used in ``wait`` mode;
        #: reject mode never enqueues. Lives in memory only — queues do not
        #: survive restart.
        self._waiters: list[Waiter] = []

        if cfg.start_paused:
            self._manual_pause = True
            self._external.manual_pause = True

    # ------------------------------------------------------------------ #
    # Owned identity (R03)
    # ------------------------------------------------------------------ #

    def register_owned_pid(self, pid: int) -> Optional[str]:
        """Register a backend worker this server started, by resolved identity.

        Identity is ``(boot_id, pid, start_ticks)``, never a bare PID: a reused PID
        must not inherit ownership (R03/T02).
        """
        from .telemetry import identity_key_for

        key = identity_key_for(pid, self.telemetry.boot_id)
        if key is None:
            # Cannot verify the identity: do NOT register. It stays external, so a
            # failed lookup cannot inflate available capacity.
            self._log(f"could not resolve identity for pid {pid}; treating as external")
            return None
        self._own_identities.add(key)
        return key

    def forget_owned_pid(self, pid: int) -> None:
        """Drop a worker's identity. Keyed by pid field, not by string suffix."""
        from .telemetry import read_boot_id

        boot = self.telemetry.boot_id or read_boot_id()
        self._own_identities = {
            key
            for key in self._own_identities
            if not (boot is not None and key.startswith(f"{boot}:{pid}:"))
        }

    @property
    def owned_identities(self) -> frozenset[str]:
        return frozenset(self._own_identities)

    # ------------------------------------------------------------------ #
    # Snapshot ingestion
    # ------------------------------------------------------------------ #

    def ingest_snapshot(self, snap: Snapshot, now_monotonic: Optional[float] = None) -> None:
        """Record a new snapshot and re-evaluate policy. Called by the sampler task."""
        now = self._clock() if now_monotonic is None else now_monotonic
        previous = self.snapshot
        self.snapshot = snap
        self._snapshot_taken_at = now
        if snap.activity_status is ActivityStatus.AVAILABLE:
            self._ever_sampled_activity = True
        self._note_external_exits(previous, snap)
        decision = self._external.update(snap, now)
        self.external_decision = decision
        self._quiet.update(snap, now)
        self._resolve_activity_latch(decision, snap, now)
        # Resource updates wake the wait list so waiters re-evaluate promptly (R09).
        if self._waiters:
            self._wake_waiters()

    def _note_external_exits(self, previous: Optional[Snapshot], current: Snapshot) -> None:
        """Tell the tracker which external PIDs exited between two snapshots.

        R04: exit of a blocking PID clears only that PID's contribution. A PID that
        is still enumerated in the new snapshot has not exited, so only the
        disappeared ones are reported; activity for a vanished PID must not keep the
        latch alive on its own.
        """
        if previous is None:
            return
        live = {p.pid for p in current.processes}
        for sample in previous.processes:
            if sample.is_owned or sample.pid in live:
                continue
            self._external.note_external_pid_exit(sample.pid)

    def _resolve_activity_latch(
        self, decision: ExternalWorkloadDecision, snap: Snapshot, now: float
    ) -> None:
        """Release a latched activity veto once a real quiescent boundary is observed.

        R04's normal resolution path: drain/unload -> no managed GPU work -> observed
        device quietness for the release window -> latch release. This is the *only*
        way an activity latch whose samples have vanished can clear, so without it a
        sustained veto would hold admission indefinitely (measured: 600 s of sparse
        samples with external memory at 1 MiB still reported BUSY).

        This is deliberately a *device-level* observation, which is what R04 specifies:
        the window must be genuinely quiet, sustained for ``release_seconds``, and no
        managed GPU work may be running or the uncertainty is preserved instead. It is
        not "one quiet sample erases a live veto" — that is exactly what R04 forbids.
        A workload that is resident-but-idle (a paused game holding VRAM, a menu) is
        covered by the independent *memory* trigger, which this path never touches;
        activity is the compute-heavy signal, and a device that has measured quiet for
        the whole release window is not compute-heavy.

        Requirements, all of them:
        * the tracker is holding an unresolved activity latch;
        * the activity signal is *absent* (SPARSE), not merely low. A low numeric
          sample is handled by ``_activity_clear`` and never needs this path;
        * no managed GPU work: no lifecycle transition, no live lease, no backend job;
        * device quietness sustained for ``release_seconds``.
        """
        if not self._external.unresolved_activity_latch:
            return
        if snap.activity_status is not ActivityStatus.SPARSE:
            return
        if self._transition_active or self._has_live_lease_locked():
            return
        if _safe_busy(self.deps):
            return
        quiet_for = self._quiet.satisfied_seconds()
        if not self._quiet.satisfied_at(now) or quiet_for < self.cfg.external_workload.release_seconds:
            return
        self._external.clear_activity_latch_after_quiescence()
        self._log(
            f"activity latch released after a verified quiescent window ({quiet_for:.1f}s quiet)"
        )

    def snapshot_age_seconds(self, now_monotonic: Optional[float] = None) -> Optional[float]:
        if self._snapshot_taken_at is None:
            return None
        now = self._clock() if now_monotonic is None else now_monotonic
        return now - self._snapshot_taken_at

    def telemetry_fresh(self, now_monotonic: Optional[float] = None) -> bool:
        age = self.snapshot_age_seconds(now_monotonic)
        if age is None:
            return False
        return age <= self.cfg.telemetry.max_age_seconds

    # ------------------------------------------------------------------ #
    # Admission
    # ------------------------------------------------------------------ #

    async def _evaluate_locked(self, kind: LeaseKind, request_id: Optional[str] = None) -> dict:
        """Compute the admission verdict for the *current* state.

        Returned as a dict so the same function feeds both admission and status,
        which is what makes the published booleans provably agree with the real
        admission decision (R11/T06). ``request_id`` is the caller's identity when
        admission (not status) is asking: the FIFO head waiter is evaluated as
        itself, everything else sees the queued-head demotion below.
        """
        blockers: list[str] = []
        now = self._clock()

        paused = self._manual_pause or self._shutting_down
        faulted = self.lifecycle is Lifecycle.FAULT

        fresh = self.telemetry_fresh(now)
        snap = self.snapshot
        valid = snap is not None and snap.valid

        external_holds = self.external_decision.holds_admission
        transition = self._transition_active
        # F12/R07: a generator recovery mutates the container's generator and
        # has cancelled every active job by the time it is observable, so new
        # admission must be closed while it runs — this is the same reason a
        # model transition closes admission.
        recovery_holds = self._recovery_outstanding_locked()

        inference_count = sum(
            1 for lease in self._leases.values() if not lease.released and lease.kind is LeaseKind.INFERENCE
        )
        at_capacity = inference_count >= self.cfg.admission.max_active_requests

        if paused:
            blockers.append("orchestrator_paused" if self._manual_pause else "shutting_down")
        if faulted:
            blockers.append(f"orchestrator_fault: {self.fault_reason}")
        if not valid:
            blockers.append("telemetry_invalid" if snap is not None else "telemetry_missing")
        elif not fresh:
            blockers.append(f"telemetry_stale ({self.snapshot_age_seconds(now):.1f}s)")
        if external_holds:
            blockers.append(f"external_workload:{self.external_decision.state.value}")
        if transition:
            blockers.append(f"model_transition:{self._transition_kind}")
        if recovery_holds:
            blockers.append("model_recovery:generator")

        # ---- what a *warm* request needs (model already resident) ------------- #
        warm_capacity = None
        if self.lifecycle is Lifecycle.READY and valid and snap is not None:
            warm_capacity = evaluate_capacity(snap, self.profile.warm_need_bytes(), context="warm")
            if not warm_capacity.ok:
                blockers.extend(warm_capacity.blockers)
        # A resident but inadmissible state must still be reported. A reader pin is
        # not inference demand, so the active-request limit applies only to
        # inference leases (R06/R08).
        if at_capacity and kind is LeaseKind.INFERENCE:
            blockers.append(f"request_capacity (max {self.cfg.admission.max_active_requests})")

        # can_admit_now must answer the SAME question acquire_lease answers (T06:
        # "status and admission agree"). For a resident model that is "no whole-state
        # blockers"; for an UNLOADED model the only admission route is the cold load,
        # so it agrees with can_begin_cold (computed below); for any in-flight
        # lifecycle (LOADING/DRAINING/UNLOADING) acquire denies outright.
        can_admit_now = not blockers

        # ---- what starting a *cold* load additionally needs ------------------- #
        # NOTE: this only *reads* the quiet tracker. Advancing it is the sampler's
        # job, so that a status poll can never push the window forward (R08/T10).
        cold_blockers: list[str] = []
        # Carry the whole-state blockers first so the reported reason is the *real*
        # first obstacle, not an artefact of which list happened to be checked.
        cold_blockers.extend(blockers)
        if self.lifecycle is not Lifecycle.UNLOADED:
            cold_blockers.append(f"lifecycle_not_unloaded:{self.lifecycle.value}")
        if external_holds:
            cold_blockers.append(f"external_workload:{self.external_decision.state.value}")
        quiet_ready = self._quiet_window_satisfied()
        if not quiet_ready:
            cold_blockers.append("quiet_window_incomplete")
        cold_capacity = None
        if self.lifecycle is Lifecycle.UNLOADED and valid and snap is not None:
            cold_capacity = evaluate_capacity(snap, self.profile.cold_need_bytes(), context="cold")
            if not cold_capacity.ok:
                cold_blockers.extend(cold_capacity.blockers)
        if self.lifecycle is not Lifecycle.UNLOADED:
            can_begin_cold = False
        else:
            can_begin_cold = (
                not cold_blockers and not paused and not faulted and fresh and valid and not transition
            )

        if self.lifecycle is Lifecycle.UNLOADED:
            can_admit_now = can_begin_cold
        elif self.lifecycle is not Lifecycle.READY:
            can_admit_now = False

        # F3 (R11/T06): a queued FIFO head is unserved INFERENCE demand. When a
        # waiter sits at the head, "can admit now" is a statement about THAT
        # request — a monitor reading `can_admit_now: true` while the eligible
        # head waiter is still queued is describing a server that looks ready but
        # is not serving. Demote the verdict for every caller except the head
        # waiter itself (which acquire_lease evaluates as itself, so the invariant
        # "published booleans agree with what acquire_lease decides for that
        # queued request" holds without demoting its own evaluation).
        #
        # Only INFERENCE is demoted: a READER pin does not contend for the
        # inference slot, so a queued inference waiter says nothing about whether
        # a metadata read may proceed (review major — demoting READER made status
        # and the READER decision disagree on the same snapshot, which is the
        # very F3 shape this rule exists to remove).
        queued_head = self._waiters[0] if self._waiters else None
        queued_ahead_holds = (
            kind is LeaseKind.INFERENCE
            and queued_head is not None
            and queued_head.request_id != request_id
        )
        if can_admit_now and queued_ahead_holds:
            can_admit_now = False

        capacity = warm_capacity if warm_capacity is not None else cold_capacity
        # Report the *actual* first obstacle. With an unloaded model the whole-state
        # blockers are typically empty while the cold path is blocked (e.g. by the
        # memory budget), and R11 requires status to name that blocker rather than
        # an unqualified "ok".
        reported_blockers = blockers if blockers else cold_blockers
        if can_admit_now is False and not reported_blockers:
            # The demotion came from the queued-head rule (F3), not from a
            # whole-state or cold blocker: name it, or the published reason
            # would say "ok" for a request that is demonstrably not being
            # served (the F1-wedge reporting shape, forbidden by R11).
            reported_blockers = ["request_capacity: queued ahead"]
            # R11 requires the reason code AND the blockers together (review
            # minor): a monitor keying on `blockers` must not see an empty list
            # beside `can_admit_now: false`. The demotion blocker is published in
            # both, so `reason` and `blockers` cannot contradict each other.
            blockers = list(reported_blockers)
        return {
            "reason": _first_reason(reported_blockers).value,
            "blockers": blockers,
            "cold_blockers": cold_blockers,
            "can_admit_now": can_admit_now,
            "can_begin_cold_load": can_begin_cold,
            "inference_leases": inference_count,
            "reader_pins": sum(
                1 for lease in self._leases.values() if not lease.released and lease.kind is LeaseKind.READER
            ),
            "paused": paused,
            "telemetry_fresh": fresh,
            "telemetry_valid": valid,
            "capacity": capacity,
            "need_bytes": self.profile.warm_need_bytes()
            if self.lifecycle is Lifecycle.READY
            else self.profile.cold_need_bytes(),
            "free_bytes": snap.mem_free if snap is not None else None,
            "quiet_window_satisfied": quiet_ready,
            "quiet_seconds_observed": self._quiet.satisfied_seconds(),
        }

    def _quiet_window_satisfied(self) -> bool:
        """Last quietness verdict, advancing the window only on new snapshots.

        The tracker records the quiet-start timestamp; this reports whether the
        *last observed* window is already long enough, using the snapshot's own
        capture time so that repeated reads cannot extend it.
        """
        snap = self.snapshot
        if snap is None or not snap.valid:
            return False
        return self._quiet.satisfied_at(snap.captured_monotonic)

    async def admission_preview(self, kind: LeaseKind = LeaseKind.INFERENCE) -> dict:
        """Read-only verdict, for status. Takes the mutex briefly."""
        async with self._mutex:
            return await self._evaluate_locked(kind)

    async def acquire_lease(
        self,
        request_id: str,
        kind: LeaseKind = LeaseKind.INFERENCE,
        *,
        deadline_seconds: Optional[float] = None,
    ) -> AdmissionResult:
        """Acquire a lease, performing a cold load if this request triggers one.

        ``mode: reject`` behaviour: if the model is resident, admit immediately or
        deny. If it is not resident, the triggering request may wait for its own
        bounded cold load, then must re-check policy before being granted anything,
        because priority may have changed while loading (R07).

        ``mode: wait`` behaviour (R09, M3): a request that cannot be admitted
        immediately is placed on the bounded FIFO wait list with the pre-lease
        deadline, and FIFO order is respected when re-evaluating — an older
        eligible waiter is admitted before a newer arrival. The deadline covers
        ALL pre-lease time and expiry is a TIMEOUT with no inference started.
        """
        deadline = deadline_seconds if deadline_seconds is not None else self.cfg.admission.max_wait_seconds

        if self.cfg.admission.mode == "wait" and kind is LeaseKind.INFERENCE:
            return await self._acquire_lease_wait(request_id, deadline)
        return await self._acquire_lease_reject(request_id, kind, deadline)

    async def _acquire_lease_reject(
        self, request_id: str, kind: LeaseKind, deadline: float
    ) -> AdmissionResult:
        """The M1/M2 reject-mode path: admit, deny, or own a cold load.

        The pre-lease deadline applies here too (R09: "all pre-lease time is
        bounded"). Reject mode has no queue, but the triggering request still
        waits for its own cold load, so that wait is wrapped in the deadline:
        expiry is ``admission_timeout`` with no inference started for it. The
        load itself is NOT cancelled — the coordinator owns it (R07).
        """
        async with self._mutex:
            verdict = await self._evaluate_locked(kind, request_id)
            if verdict["can_admit_now"] and self.lifecycle is Lifecycle.READY:
                lease = self._grant_locked(request_id, kind)
                return AdmissionResult(AdmissionOutcome.GRANTED, Reason.OK, lease, verdict)
            if kind is LeaseKind.READER:
                # F10/R12: a reader pin protects a short read of an ALREADY-
                # resident model. It must never wait for, or trigger, a cold
                # load — the model being absent is exactly the condition the
                # read is refused for. (Inference demand owns cold loads.)
                #
                # The reason is the verdict's own (review major): the evaluator
                # is now kind-aware, so the READER decision and the published
                # status agree on the same snapshot. Naming a different code here
                # re-created the F3 disagreement one branch over.
                blockers = verdict["blockers"] or [
                    f"lifecycle_not_ready:{self.lifecycle.value}"
                ]
                return AdmissionResult(
                    AdmissionOutcome.DENIED,
                    _first_reason(blockers),
                    None,
                    {**verdict, "blockers": blockers},
                )
            if self.lifecycle is not Lifecycle.UNLOADED:
                # Resident but not admissible -> this request cannot be served.
                # The lifecycle state itself is a blocker: without naming it, a
                # request refused *only* because the model is DRAINING/UNLOADING/
                # LOADING would be reported as `ok` with no Retry-After, which
                # tells an inference-only client nothing (R11/R12).
                blockers = verdict["blockers"] or [
                    f"lifecycle_not_ready:{self.lifecycle.value}"
                ]
                if self.lifecycle is Lifecycle.LOADING:
                    blockers = [f"model_transition:{self._transition_kind}"] + blockers
                return AdmissionResult(
                    AdmissionOutcome.DENIED,
                    _first_reason(blockers),
                    None,
                    {**verdict, "blockers": blockers},
                )
            if not verdict["can_begin_cold_load"]:
                return AdmissionResult(
                    AdmissionOutcome.DENIED, _first_reason(verdict["cold_blockers"]), None, verdict
                )
            # This request owns the cold load.
            self._reserve_transition_locked("cold_load", self._clock())

        try:
            ok, error = await asyncio.wait_for(
                self._execute_load_detached(), timeout=max(0.0, deadline)
            )
        except asyncio.TimeoutError:
            # Deadlines are enforced, but the shared load is not abandoned: it
            # keeps running under the coordinator and reconciles itself (R07).
            self._log(f"request {request_id} timed out during its cold load; load continues")
            return AdmissionResult(
                AdmissionOutcome.TIMEOUT,
                Reason.ADMISSION_TIMEOUT,
                None,
                {"timeout_seconds": deadline},
            )
        except asyncio.CancelledError:
            raise

        async with self._mutex:
            # The load ran outside the mutex, so ownership of the transition must be
            # reconciled here. `_reconcile_after_load` is the single place that clears
            # the token and publishes residency; calling it unconditionally keeps one
            # authority for the LOADING -> READY/UNLOADED/FAULT decision instead of
            # duplicating it on the success path.
            self._reconcile_after_load_locked(
                None if ok else RuntimeError(error or "load failed")
            )
            if not ok or self.lifecycle is not Lifecycle.READY:
                return AdmissionResult(
                    AdmissionOutcome.DENIED,
                    Reason.ORCHESTRATOR_FAULT if not ok else Reason.TELEMETRY_UNAVAILABLE,
                    None,
                    {"load_error": error, "lifecycle": self.lifecycle.value},
                )
            # Re-evaluate: priority may have appeared during the load (R07).
            # Evaluated as THIS request: in reject mode there is no queue, but
            # passing the identity keeps the evaluator's contract uniform.
            after = await self._evaluate_locked(kind, request_id)
            if not after["can_admit_now"]:
                self._drain_requested = True
                self._drain_reason = _first_reason(after["blockers"])
                return AdmissionResult(
                    AdmissionOutcome.DENIED,
                    _first_reason(after["blockers"]),
                    None,
                    after,
                )
            lease = self._grant_locked(request_id, kind)
            return AdmissionResult(AdmissionOutcome.GRANTED, Reason.OK, lease, after)

    # ------------------------------------------------------------------ #
    # Bounded admission waiting (R09, M3)
    # ------------------------------------------------------------------ #

    async def _acquire_lease_wait(self, request_id: str, deadline: float) -> AdmissionResult:
        """Wait-mode admission: one bounded in-memory FIFO (R09).

        * FIFO eligibility: an older waiter is always evaluated before a newer
          arrival, so a new request can never bypass it.
        * One deadline from enqueue covers every pre-lease second.
        * Overflow fails immediately with 429-shaped REQUEST_CAPACITY... mapped
          by the caller to ADMISSION_QUEUE_FULL via the queue-full code.
        * A slot is released exactly once: admission, deadline, disconnect
          (caller cancellation), or shutdown.
        """
        # Fast path: immediately admissible requests never touch the queue — EXCEPT
        # when an older request is already waiting. R09's "FIFO ordering prevents
        # new arrivals bypassing older eligible waiters" is a property of the
        # admission decision, not of the queue, so the fast path must check the
        # queue head before granting. Without this a new arrival slips past a
        # stranded head waiter, which is exactly the bypass the queue exists to
        # prevent.
        async with self._mutex:
            verdict = await self._evaluate_locked(LeaseKind.INFERENCE, request_id)
            head_is_other = bool(self._waiters) and self._waiters[0].request_id != request_id
            if verdict["can_admit_now"] and self.lifecycle is Lifecycle.READY and not head_is_other:
                lease = self._grant_locked(request_id, LeaseKind.INFERENCE)
                return AdmissionResult(AdmissionOutcome.GRANTED, Reason.OK, lease, verdict)

            if len(self._waiters) >= self.cfg.admission.max_pending_requests:
                return AdmissionResult(
                    AdmissionOutcome.DENIED,
                    Reason.ADMISSION_QUEUE_FULL,
                    None,
                    verdict,
                )

            waiter = Waiter(
                request_id=request_id,
                entered_monotonic=self._clock(),
                deadline_seconds=deadline,
                event=asyncio.Event(),
                clock=self._clock,
            )
            self._waiters.append(waiter)

        try:
            while True:
                # One decision pass under the mutex. The ONLY deadline authority
                # is the coordinator's clock (waiter.expired_at) — asyncio
                # timeouts here are transport, not policy, so tests with
                # injected clocks stay deterministic.
                grant: Optional[str] = None
                deny_reason: Optional[Reason] = None
                result_verdict: dict = {}
                async with self._mutex:
                    now = self._clock()
                    if waiter.expired_at(now):
                        deny_reason = Reason.ADMISSION_TIMEOUT
                    elif self._shutting_down or self._manual_pause:
                        deny_reason = Reason.ORCHESTRATOR_PAUSED
                    elif self._drain_requested and self._drain_reason is not None and self._drain_reason is not Reason.OK:
                        deny_reason = self._drain_reason
                    else:
                        grant = await self._try_admit_fifo_locked(request_id)
                        if grant is None and self._waiters and self._waiters[0] is waiter:
                            # Head of the line and READY is blocked by cold-load
                            # conditions: own the load, then re-check before
                            # granting (R07). A non-head waiter NEVER starts a
                            # load — that would bypass FIFO eligibility.
                            verdict = await self._evaluate_locked(LeaseKind.INFERENCE, request_id)
                            result_verdict = verdict
                            if verdict["can_begin_cold_load"]:
                                self._reserve_transition_locked("cold_load", self._clock())
                                grant = "load"

                if grant == "load":
                    try:
                        ok, error = await self._execute_load_detached()
                    except asyncio.CancelledError:
                        raise
                    async with self._mutex:
                        self._reconcile_after_load_locked(
                            None if ok else RuntimeError(error or "load failed")
                        )
                    continue  # re-loop: admitted or denied on the next evaluation

                if deny_reason is not None or grant == "granted":
                    async with self._mutex:
                        self._release_waiter_locked(waiter)
                    if deny_reason is not None:
                        return AdmissionResult(
                            AdmissionOutcome.TIMEOUT
                            if deny_reason is Reason.ADMISSION_TIMEOUT
                            else AdmissionOutcome.DENIED,
                            deny_reason,
                            None,
                            {"waited_seconds": round(self._clock() - waiter.entered_monotonic, 2)},
                        )
                    async with self._mutex:
                        lease = self._grant_locked(request_id, LeaseKind.INFERENCE)
                        return AdmissionResult(AdmissionOutcome.GRANTED, Reason.OK, lease, result_verdict)

                # Still waiting. Sleep bounded by the REMAINING injected-clock
                # budget so the caller can never hang past its deadline even if
                # no sampler wake ever arrives. The 200 ms cap is the wake bound
                # (F13): a wake that races the event (released, then cleared
                # before the waiter re-checked) is recovered by the next
                # deadline-bounded pass instead of parking the queued head up to
                # the full sampler cadence.
                remaining = waiter.remaining_seconds()
                if remaining <= 0:
                    continue  # deadline crossed between passes; re-loop catches it
                try:
                    await asyncio.wait_for(waiter.event.wait(), timeout=min(remaining, 0.2))
                except asyncio.TimeoutError:
                    pass  # re-loop re-checks the injected-clock deadline
                waiter.event.clear()
        except asyncio.CancelledError:
            # Client disconnect / cancellation: free the slot exactly once (R09).
            await self._release_waiter(request_id)
            raise

    async def _try_admit_fifo_locked(self, request_id: str):
        """Admit this waiter only if it is the FIFO head and admissible (R09).

        Returns ``"granted"``, ``"load"`` (caller must run the shared load),
        or ``None`` when the waiter must keep waiting. An older waiter that is
        merely *blocked* (not timed out) holds its place — a newer arrival
        cannot bypass it.
        """
        if not self._waiters or self._waiters[0].request_id != request_id:
            return None  # someone older is waiting; FIFO holds
        head = self._waiters[0]
        if head.admitted:
            return None
        verdict = await self._evaluate_locked(LeaseKind.INFERENCE, request_id)
        if verdict["can_admit_now"] and self.lifecycle is Lifecycle.READY:
            head.admitted = True
            return "granted"
        return None

    def _release_waiter_locked(self, waiter: Waiter) -> None:
        """Free a wait slot exactly once (R09)."""
        waiter.released = True
        with contextlib.suppress(ValueError):
            self._waiters.remove(waiter)
        waiter.event.set()

    async def _release_waiter(self, request_id: str) -> None:
        async with self._mutex:
            for waiter in list(self._waiters):
                if waiter.request_id == request_id and not waiter.released:
                    self._release_waiter_locked(waiter)
                    return

    def _wake_waiters(self) -> None:
        """Sampler updates and transition completions wake the waiters (R09)."""
        for waiter in self._waiters:
            waiter.event.set()

    def _grant_locked(self, request_id: str, kind: LeaseKind) -> Lease:
        lease_id = f"{kind.value}:{request_id}:{len(self._leases)}"
        lease = Lease(
            lease_id=lease_id,
            kind=kind,
            request_id=request_id,
            container_identity=self.deps.container_identity(),
            acquired_monotonic=self._clock(),
        )
        self._leases[lease_id] = lease
        if kind is LeaseKind.INFERENCE:
            # New inference demand cancels an idle countdown (R08).
            self._idle_since = None
        return lease

    async def release_lease(self, lease: Lease) -> None:
        """Release a lease. Idempotent, so cleanup paths can be defensive.

        Before releasing an inference lease the pinned container identity is
        re-verified (review major: ``Lease.container_identity`` was captured but
        never re-checked). If the container was swapped underneath the request —
        a load replacing the model while a lease was still held — that is a
        lifecycle-integrity violation: it latches FAULT rather than silently
        returning to READY, because the completed request did not run against
        the container this lifecycle believed was resident.
        """
        async with self._mutex:
            stored = self._leases.get(lease.lease_id)
            if stored is None or stored.released:
                return
            if stored.kind is LeaseKind.INFERENCE and stored.container_identity is not None:
                current = None
                with contextlib.suppress(Exception):
                    current = self.deps.container_identity()
                if current is not None and current != stored.container_identity:
                    self.fault_reason = (
                        f"lease {stored.lease_id} was pinned to container "
                        f"{stored.container_identity!r} but the live container is "
                        f"{current!r}; the model changed underneath an active request"
                    )
                    self.lifecycle = Lifecycle.FAULT
                    self.last_error = self.fault_reason
            stored.released = True
            stored.release_monotonic = self._clock()
            if stored.kind is LeaseKind.INFERENCE:
                self._last_completion = stored.release_monotonic
                if not self._has_live_lease_locked():
                    self._idle_since = stored.release_monotonic
                    self._log("idle timer started after final inference lease")
            # Capacity just freed: wake the wait list so the FIFO head is
            # re-evaluated at the moment it becomes eligible, rather than at the
            # next sampler tick (R09: "resource release ... wake waiters").
            if self._waiters:
                self._wake_waiters()

    def _has_live_lease_locked(self) -> bool:
        return any(not lease.released for lease in self._leases.values())

    # `release()` must be called from `finally:` blocks, which cannot await freely
    # inside a cancelled scope. This sync wrapper schedules the async release.
    def release_lease_soon(self, lease: Optional[Lease]) -> None:
        if lease is None:
            return
        with contextlib.suppress(RuntimeError):
            asyncio.get_running_loop().create_task(self.release_lease(lease))

    # ------------------------------------------------------------------ #
    # Transitions (load / unload) — executed with the mutex released
    # ------------------------------------------------------------------ #

    def _reserve_transition_locked(self, kind: str, now: float) -> None:
        if self._transition_active:
            raise LifecycleError(f"transition already active: {self._transition_kind}")
        self._transition_active = True
        self._transition_kind = kind
        self._transition_started = now
        self.lifecycle = Lifecycle.LOADING if kind == "cold_load" else Lifecycle.UNLOADING

    def _release_transition_locked(self) -> None:
        self._transition_active = False
        self._transition_kind = None
        self._transition_started = None
        self._pending_unload = False
        # A transition ending changes what is admissible, so the wait list must
        # re-evaluate now rather than waiting for the next sampler tick (R09).
        if self._waiters:
            self._wake_waiters()

    async def _execute_load(self) -> tuple[bool, Optional[str]]:
        try:
            await self.deps.load_model()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - load failure is a normal outcome
            return False, f"{type(exc).__name__}: {exc}"
        return True, None

    async def _execute_load_detached(self) -> tuple[bool, Optional[str]]:
        """Run the load in a task that request cancellation cannot abandon.

        R07/R09: the loader is owned by the coordinator, not by the requesting
        client. If the client's task is cancelled mid-load, the load itself must
        still run to completion (or be cancelled deliberately), so the work is
        wrapped in its own task and awaited under ``asyncio.shield``. A cancelled
        request therefore leaves a *live* loader that the next reconciliation
        observes, instead of tearing down shared work and latching a process-wide
        FAULT because "container state is uncertain".

        A strong reference is kept so the task cannot be garbage-collected while
        the request that owned it disappears.
        """
        task = asyncio.get_running_loop().create_task(self._execute_load())
        self._load_task = task
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # Deliberately do NOT cancel `task`: the coordinator still owns it.
            # Reconcile the load's real outcome in the background so state is
            # never left at LOADING.
            def _deferred(t: asyncio.Task) -> None:
                try:
                    ok, error = t.result()
                except asyncio.CancelledError:
                    ok, error = False, "load task cancelled"
                except Exception as exc:  # noqa: BLE001
                    ok, error = False, f"{type(exc).__name__}: {exc}"
                # Reconcile the real outcome: state must never be left at LOADING.
                asyncio.get_running_loop().create_task(
                    self._reconcile_after_load(
                        None if ok else RuntimeError(error or "load failed")
                    )
                )

            task.add_done_callback(_deferred)
            raise

    # ------------------------------------------------------------------ #
    # Generator-recovery ownership (R07, F12)
    # ------------------------------------------------------------------ #

    def adopt_recovery_task(self) -> bool:
        """Adopt the backend's in-flight generator-recovery task, if any.

        Called by the sampler each tick *before* maintenance decisions. The
        upstream recovery path schedules ``create_generator()`` via
        ``asyncio.ensure_future`` and discards the handle, so nothing else
        observes it. Adoption keeps a strong reference (the task cannot be
        garbage-collected mid-recovery) and arms the done-callback that drops
        the reference when recovery settles. Returns True when a live task is
        (or was just) adopted.

        Idempotent per task: re-adopting the SAME task on the next cadence does
        not register a second done-callback (review minor — callbacks used to
        accumulate one per cadence on a long recovery).
        """
        with contextlib.suppress(Exception):
            task = self.deps.adopt_recovery_task()
            if task is not None and not task.done():
                if task is self._recovery_task:
                    return True  # already adopted; do not re-register
                self._recovery_task = task
                task.add_done_callback(self._recovery_task_done)
                return True
        return False

    def _recovery_task_done(self, task: asyncio.Task) -> None:
        """Drop the adopted reference and wake waiters once recovery settles.

        Runs in the event loop directly (done-callback); it only clears a field
        and sets asyncio events, never awaits. Waiters must re-evaluate because
        a finished recovery changes what admission would decide (F12/F13).

        Guards against clearing a NEWER adopted task: a callback belonging to a
        task that has already been superseded must not null the reference the
        coordinator is currently tracking (review minor).
        """
        if task is not self._recovery_task:
            return
        self._recovery_task = None
        self._wake_waiters()

    def _recovery_outstanding_locked(self) -> bool:
        """True while a generator recovery is running on the live container.

        This is the signal ``backend_busy`` cannot provide: recovery cancels the
        container's jobs *first* (emptying the registry the busy signal reads)
        and only then rebuilds the generator, so during that window the job
        registry says "idle" while the backend is mid-mutation. Treat an
        exception in the check as outstanding — the same fail-closed direction
        as ``_safe_busy`` — because unloading under an uncertain recovery is
        exactly the race R07 forbids.

        The live container is consulted **first** (review major: adoption happens
        once per sampler cadence, so an adopted-only check left a ≤1 s window in
        which a recovery was already scheduled but not yet adopted — admission
        would be granted and a tick could begin unloading underneath it). The
        deps binding reads the container's own ``recovery_task``, so this is a
        live property of the backend rather than a polled copy; the adopted
        reference is the fallback for a deps object that cannot see the
        container (fakes in tests).
        """
        with contextlib.suppress(Exception):
            if self.deps.recovery_in_progress():
                return True
        task = self._recovery_task
        if task is None:
            return False
        with contextlib.suppress(Exception):
            return not task.done()
        return True

    async def _reconcile_after_load(
        self, error: Optional[Exception], *, cancelled: bool = False
    ) -> None:
        """Reconcile believed vs actual state after a load attempt.

        READY is published only when a real container exists — never from an
        intention or a task-start event (R07). An uncertain failure latches FAULT
        rather than guessing.

        Acquires the mutex; callers must NOT already hold it.
        """
        async with self._mutex:
            self._reconcile_after_load_locked(error, cancelled=cancelled)

    def _reconcile_after_load_locked(
        self, error: Optional[Exception], *, cancelled: bool = False
    ) -> None:
        self._release_transition_locked()
        present = False
        with contextlib.suppress(Exception):
            present = self.deps.container_present()
        if cancelled:
            self.fault_reason = "load task cancelled; container state uncertain"
            self.lifecycle = Lifecycle.FAULT
            self.last_error = self.fault_reason
            return
        if error is not None:
            self.last_error = str(error)
            if present:
                # Load reported failure but a container exists: do not publish
                # READY on an ambiguous outcome.
                self.fault_reason = f"load failed but a container is present: {error}"
                self.lifecycle = Lifecycle.FAULT
            else:
                self.lifecycle = Lifecycle.UNLOADED
                self.fault_reason = None
            return
        if not present:
            self.fault_reason = "load reported success but no container exists"
            self.lifecycle = Lifecycle.FAULT
            self.last_error = self.fault_reason
            return
        # The container exists: verify its effective envelope against the
        # calibration before publishing READY (review major: the effective load
        # envelope must be bound to the calibration, not to YAML use_as_default).
        mismatch = self._verify_envelope_locked()
        if mismatch is not None:
            self.fault_reason = f"loaded envelope violates the calibrated profile: {mismatch}"
            self.lifecycle = Lifecycle.FAULT
            self.last_error = self.fault_reason
            return
        self.lifecycle = Lifecycle.READY
        self.fault_reason = None
        self._idle_since = self._clock()

    def _verify_envelope_locked(self) -> Optional[str]:
        """Compare the real container's envelope with the calibration (R14/R05).

        Returns a human-readable mismatch description, or None when everything
        matches.

        F14 decision (recorded): a key the container cannot report (None) is
        SKIPPED rather than failing the load — a deliberate, bounded fail-open.
        Reasoning: the check's purpose is to detect a *divergence* from the
        measured footprint; "cannot read this key" is not evidence of
        divergence, and failing the whole load on a reporting gap would trade a
        real, working model for a latched FAULT on every backend that does not
        surface a field. The fail-open is bounded two ways: (a) only the keys
        that ARE reported must match exactly, so any observed divergence still
        latches FAULT; (b) telemetry fail-closed behaviour is untouched — this
        applies only to the post-load envelope read.

        Correction to an earlier overstatement (review nit): the explicit load
        kwargs do NOT fully pin the config→container path. `common/model.py`
        merges the model folder's `tabby_config.yml` **above** request kwargs
        (`inline_overrides` wins in `deep_merge_dicts`), so a folder-level
        override can still change the effective footprint without any request
        field showing it. That is precisely why (a) carries the weight: this
        verifier is the backstop that catches such a divergence when the
        container reports the value. The remaining hole — a folder override on a
        key the container then cannot report — is accepted rather than closed,
        because the alternative (failing every load on an unreported key) breaks
        working backends.
        """
        envelope = self.profile.envelope
        if not envelope:
            return None  # nothing calibrated to bind (disabled-mode path in tests)
        actual: dict[str, Any] = {}
        with contextlib.suppress(Exception):
            actual = dict(self.deps.container_envelope() or {})
        mismatches = []
        for key, expected in envelope.items():
            if expected is None:
                continue
            observed = actual.get(key)
            if observed is None:
                continue  # F14: unreported keys are skipped (see decision above)
            if _envelope_equal(observed, expected):
                continue
            mismatches.append(f"{key}: calibrated {expected!r}, container reports {observed!r}")
        if mismatches:
            return "; ".join(mismatches)
        return None

    async def request_unload(self, reason: Reason, *, force: bool = False) -> dict:
        """Request an unload. Drains first if model use is outstanding.

        Returns a prompt status describing what actually happened — it does not
        claim VRAM is free (R10).

        An already-UNLOADED coordinator returns immediately and sets NO sticky
        drain: leaving `_drain_requested` set would make the next eligible cold
        load unload itself on the following maintenance pass (R08/R12).
        """
        async with self._mutex:
            if self.lifecycle is Lifecycle.UNLOADED and not self._has_live_lease_locked():
                return {"state": self.lifecycle.value, "action": "already_unloaded"}
            self._drain_requested = True
            self._drain_reason = reason
            live = self._has_live_lease_locked()
            busy = False
            with contextlib.suppress(Exception):
                busy = self.deps.backend_busy()
            if live or busy or self._recovery_outstanding_locked() or self._transition_active:
                self.lifecycle = (
                    Lifecycle.DRAINING if self.lifecycle is not Lifecycle.LOADING else self.lifecycle
                )
                return {
                    "state": self.lifecycle.value,
                    "action": "draining",
                    "outstanding_leases": self._live_lease_summary_locked(),
                }
            self._reserve_transition_locked("unload", self._clock())

        await self._perform_unload()
        async with self._mutex:
            return {"state": self.lifecycle.value, "action": "unloaded"}

    async def _perform_unload(self) -> None:
        error: Optional[str] = None
        try:
            await self.deps.unload_model()
        except asyncio.CancelledError:
            # Never leave the transition token set on a cancellation: a leaked
            # UNLOADING + active-token state has no recovery path (no tick branch,
            # reserve_explicit_load refuses, requests deny forever). Reconcile
            # what is true, then re-raise so the caller's cancellation still
            # propagates (R11/R13).
            await self._reconcile_after_unload(None, cancelled=True)
            raise
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
        await self._reconcile_after_unload(error)

    async def _reconcile_after_unload(
        self, error: Optional[str], *, cancelled: bool = False
    ) -> None:
        """Verify container absence plus return toward the unloaded baseline (R08).

        Device *total* used memory is not required to reach zero: CUDA contexts and
        driver allocations legitimately remain.

        ``cancelled`` marks an unload whose await was interrupted (client
        disconnect / shutdown). The transition token is released either way —
        a leaked token has no recovery path — and the resulting state is
        reconciled from the actual container rather than assumed.
        """
        async with self._mutex:
            self._release_transition_locked()
            present = False
            with contextlib.suppress(Exception):
                present = self.deps.container_present()
            if error is not None or present:
                self.fault_reason = f"unload did not clear the container: {error or 'still present'}"
                self.lifecycle = Lifecycle.FAULT
                self.last_error = self.fault_reason
                return
            self.lifecycle = Lifecycle.UNLOADED
            self.fault_reason = None
            self._idle_since = None
            self._drain_requested = False
            self._drain_reason = None
            self._leases = {lease_id: l for lease_id, l in self._leases.items() if not l.released}

    def _live_lease_summary_locked(self) -> list[dict]:
        now = self._clock()
        return [
            {
                "lease_id": lease.lease_id,
                "kind": lease.kind.value,
                "age_seconds": round(now - lease.acquired_monotonic, 2),
            }
            for lease in self._leases.values()
            if not lease.released
        ]

    # ------------------------------------------------------------------ #
    # Explicit admin load (R12: explicit loads use the same machinery)
    # ------------------------------------------------------------------ #

    def reserve_explicit_load(self) -> None:
        """Reserve the cold-load transition for an explicit admin load.

        The same admission boundary an inference-triggered cold load passes:
        no pause/fault/veto/candidate, fresh telemetry, a continuous quiet
        window, sufficient free memory, and no transition already active.
        The admin route turns a refusal into the R12-shaped HTTP error.

        Called synchronously from the route handler, so the check-and-reserve
        runs as one non-async section between awaits: the event loop
        serializes it against the coordinator's own async sections, which is
        the same discipline ``acquire_lease``'s fast path relies on.
        """
        if self._transition_active:
            raise LifecycleError(f"transition already active: {self._transition_kind}")
        blockers = []
        now = self._clock()
        fresh = self.telemetry_fresh(now)
        snap = self.snapshot
        valid = snap is not None and snap.valid
        paused = self._manual_pause or self._shutting_down
        faulted = self.lifecycle is Lifecycle.FAULT
        external_holds = self.external_decision.holds_admission
        quiet_ready = self._quiet_window_satisfied()
        cold_capacity = None
        if self.lifecycle is Lifecycle.UNLOADED and valid and snap is not None:
            cold_capacity = evaluate_capacity(snap, self.profile.cold_need_bytes(), context="cold")
        if paused:
            blockers.append("orchestrator_paused" if self._manual_pause else "shutting_down")
        if faulted:
            blockers.append(f"orchestrator_fault: {self.fault_reason}")
        if not valid:
            blockers.append("telemetry_invalid" if snap is not None else "telemetry_missing")
        elif not fresh:
            blockers.append(f"telemetry_stale ({self.snapshot_age_seconds(now):.1f}s)")
        if external_holds:
            blockers.append(f"external_workload:{self.external_decision.state.value}")
        if self.lifecycle is not Lifecycle.UNLOADED:
            blockers.append(f"lifecycle_not_unloaded:{self.lifecycle.value}")
        if not quiet_ready:
            blockers.append("quiet_window_incomplete")
        if self.lifecycle is Lifecycle.UNLOADED and cold_capacity is not None and not cold_capacity.ok:
            blockers.extend(cold_capacity.blockers)
        if blockers:
            raise LifecycleError(_first_reason(blockers).value)
        self._reserve_transition_locked("cold_load", now)

    async def execute_explicit_load(self) -> None:
        """Run the reserved load through the production deps and reconcile it.

        Owns the transition through completion exactly like the
        inference-triggered cold load: one load owner, failure reconciled, and
        READY published only from a real, envelope-verified container (R07).
        """
        try:
            ok, error = await self._execute_load()
        except asyncio.CancelledError:
            await self._reconcile_after_load(Exception("explicit load task cancelled"), cancelled=True)
            raise
        await self._reconcile_after_load(None if ok else RuntimeError(error or "load failed"))

    # ------------------------------------------------------------------ #
    # Periodic maintenance (idle TTL, drain progression)
    # ------------------------------------------------------------------ #

    async def tick(self) -> Optional[str]:
        """One maintenance pass. Returns an action name when something happened."""
        action: Optional[str] = None
        async with self._mutex:
            if self._shutting_down:
                return None
            now = self._clock()

            # -1) Execute an unload transition that a caller reserved but
            #     deliberately did not await (pause's T19 budget). This must be
            #     checked first: the transition is already reserved, so none of
            #     the branches below can act on it.
            if self._pending_unload:
                self._pending_unload = False
                action = "unload_deferred"
                self._log("maintenance: unload_deferred (reserved by pause)")

            # 0) A sustained policy veto closes admission and drains a resident model
            #    (R08). A CANDIDATE deliberately does *not*: it holds new admission
            #    while confirmation is pending but must not evict a working model.
            elif (
                self.external_decision.state is PriorityState.BUSY
                and self.lifecycle is Lifecycle.READY
                and not self._drain_requested
            ):
                self._drain_requested = True
                self._drain_reason = Reason.EXTERNAL_GPU_BUSY

            # 1) Drain progression: veto/pause/unload with no model use outstanding.
            if self._drain_requested and self.lifecycle is Lifecycle.READY:
                live = self._has_live_lease_locked()
                busy = _safe_busy(self.deps)
                recovery = self._recovery_outstanding_locked()
                if live or busy or recovery:
                    self.lifecycle = Lifecycle.DRAINING
                else:
                    self._reserve_transition_locked("unload", now)
                    action = "unload_veto"
            # 2) Drain completion -> unload once pins/backend work finish.
            #
            # Deliberately NOT gated on `_drain_requested`: a state that is
            # ALREADY DRAINING with nothing outstanding must still reach
            # UNLOADED. Gating it there created an unrecoverable wedge — if the
            # pending-drain flag was ever cleared while the lifecycle sat in
            # DRAINING (a resume cancelling a pause-reasoned drain, say), no
            # branch could ever unload and every request stayed denied forever.
            elif self.lifecycle is Lifecycle.DRAINING:
                if (
                    not self._has_live_lease_locked()
                    and not _safe_busy(self.deps)
                    and not self._recovery_outstanding_locked()
                ):
                    self._reserve_transition_locked("unload", now)
                    action = "unload_drained"
            # 3) Idle TTL expiry: recheck leases and transition state atomically.
            elif (
                self.lifecycle is Lifecycle.READY
                and not self._drain_requested
                and self._idle_since is not None
                and not self._has_live_lease_locked()
            ):
                idle_for = now - self._idle_since
                if (
                    idle_for >= self.cfg.idle_unload.seconds
                    and not _safe_busy(self.deps)
                    and not self._recovery_outstanding_locked()
                ):
                    self._drain_requested = True
                    self._drain_reason = Reason.OK
                    self._reserve_transition_locked("unload", now)
                    action = "unload_idle"

        if action is not None:
            self._log(f"maintenance: {action}")
            await self._perform_unload()
            return action
        return None

    # ------------------------------------------------------------------ #
    # Manual pause / resume (R10)
    # ------------------------------------------------------------------ #

    async def pause(self) -> dict:
        """Idempotent pause: close admission, request the drain, return promptly.

        R10/T19: the route must answer promptly with the *actual* state, and the
        unload genuinely must not be awaited here. This method reserves the
        unload transition (so the returned snapshot names it) and returns; the
        teardown itself is executed by maintenance, ``tick()``, on the sampler
        cadence. Callers that need the terminal state read status or wait on the
        lifecycle externally. Awaiting the unload inline measured 2.0 s with a
        2 s fake — over T19's 1 s budget.

        R09/R10: unadmitted waiters are failed immediately rather than
        accumulating work behind the pause.
        """
        async with self._mutex:
            self._manual_pause = True
            self._external.manual_pause = True
            self._drain_requested = True
            self._drain_reason = Reason.ORCHESTRATOR_PAUSED
            for waiter in list(self._waiters):
                if not waiter.released:
                    waiter.result = AdmissionResult(
                        AdmissionOutcome.DENIED, Reason.ORCHESTRATOR_PAUSED, None, {}
                    )
                    self._release_waiter_locked(waiter)
            if self.lifecycle is Lifecycle.READY:
                live = self._has_live_lease_locked()
                busy = _safe_busy(self.deps)
                if not live and not busy and not self._transition_active and not self._recovery_outstanding_locked():
                    # Nothing outstanding: reserve the unload transition so the
                    # snapshot names it, and hand execution to tick().
                    self._reserve_transition_locked("unload", self._clock())
                    self._pending_unload = True
                else:
                    self.lifecycle = Lifecycle.DRAINING

        return await self.admission_preview()

    async def resume(self) -> dict:
        """Idempotent resume: clears ONLY the manual veto (R10)."""
        async with self._mutex:
            self._manual_pause = False
            self._external.manual_pause = False
            if self._drain_reason is Reason.ORCHESTRATOR_PAUSED:
                self._drain_requested = False
                self._drain_reason = None
            # Waiters resume evaluation with the pause cleared (R09: resume with
            # no live demand must not load — the waiters re-evaluate on their
            # own cadence and still need a real quiet window).
            for waiter in self._waiters:
                waiter.event.set()
        return await self.admission_preview()

    async def begin_shutdown(self) -> dict:
        """Close admission and wait for owned work using upstream semantics (R13)."""
        async with self._mutex:
            self._shutting_down = True
            self._manual_pause = True
            self._external.manual_pause = True
            self._drain_requested = True
            # R09/R13: graceful shutdown fails pending waiters.
            for waiter in list(self._waiters):
                if not waiter.released:
                    waiter.result = AdmissionResult(
                        AdmissionOutcome.DENIED, Reason.ORCHESTRATOR_PAUSED, None, {}
                    )
                    self._release_waiter_locked(waiter)
        try:
            await self.request_unload(Reason.ORCHESTRATOR_PAUSED)
        except Exception:  # noqa: BLE001 - shutdown must not raise
            pass
        async with self._mutex:
            return {"state": self.lifecycle.value, "outstanding": self._live_lease_summary_locked()}

    # ------------------------------------------------------------------ #
    # Status (R11)
    # ------------------------------------------------------------------ #

    async def status(self) -> dict:
        """Bounded, secret-free status. Missing metrics are null with a reason."""
        async with self._mutex:
            verdict = await self._evaluate_locked(LeaseKind.INFERENCE)
            snap = self.snapshot
            now = self._clock()
            coverage = (
                policy.derive_coverage(snap, ever_sampled_activity=self._ever_sampled_activity)
                if snap is not None
                else None
            )
            transition_age = (
                None if self._transition_started is None else round(now - self._transition_started, 2)
            )
            idle_for = None if self._idle_since is None else round(now - self._idle_since, 2)

            top_external = []
            if snap is not None:
                rows = [p for p in snap.processes if not p.is_owned]
                rows.sort(key=lambda p: -p.used_bytes)
                top_external = [
                    {
                        "pid": p.pid,
                        "used_mib": round(p.used_mib, 1),
                        "comm": p.comm,
                        "sources": list(p.sources),
                    }
                    for p in rows[:5]
                ]

            return {
                "enabled": True,
                "model": {
                    "configured": self.cfg.model.name,
                    "container_identity": _safe_call(self.deps.container_identity),
                    "calibration_id": self.cfg.model.calibration_id,
                },
                "lifecycle": self.lifecycle.value,
                "policy": self.external_decision.state.value,
                "paused": verdict["paused"],
                "pause_is_process_local": True,
                "pause_persistence": "runtime pause resets to orchestrator.start_paused on restart",
                "detector": coverage.as_dict() if coverage is not None else None,
                "gpu": {
                    "device_uuid": snap.device_uuid if snap is not None else self.cfg.device_uuid,
                    "memory_total": snap.mem_total if snap is not None else None,
                    "memory_used": snap.mem_used if snap is not None else None,
                    "memory_free": snap.mem_free if snap is not None else None,
                    "attribution_gap": snap.mem_reserved_derived if snap is not None else None,
                    "utilization_percent": snap.device_utilization_percent if snap is not None else None,
                    "owned_memory": _owned_bytes(snap),
                    "external_memory": snap.external_total_bytes if snap is not None else None,
                    "external_processes": snap.external_count if snap is not None else None,
                    "top_external_processes": top_external,
                },
                "activity": {
                    "status": snap.activity_status.value if snap is not None else None,
                    "error": snap.activity_error if snap is not None else None,
                    "max_external_sm_percent": _max_external_sm(snap),
                },
                "telemetry": {
                    "valid": verdict["telemetry_valid"],
                    "fresh": verdict["telemetry_fresh"],
                    "age_seconds": _round_or_none(self.snapshot_age_seconds(now)),
                    "max_age_seconds": self.cfg.telemetry.max_age_seconds,
                    "notes": list(snap.notes) if snap is not None else ["no snapshot yet"],
                },
                "requests": {
                    "active": verdict["inference_leases"],
                    "reader_pins": verdict["reader_pins"],
                    "max_active": self.cfg.admission.max_active_requests,
                    "pending": len(self._waiters),
                    "max_pending": self.cfg.admission.max_pending_requests,
                    "mode": self.cfg.admission.mode,
                },
                "idle": {"seconds_since_last_completion": idle_for, "ttl_seconds": self.cfg.idle_unload.seconds},
                "transition": {
                    "active": self._transition_active,
                    "kind": self._transition_kind,
                    "age_seconds": transition_age,
                },
                "admission": {
                    "can_admit_now": verdict["can_admit_now"],
                    "can_begin_cold_load": verdict["can_begin_cold_load"],
                    "reason": verdict["reason"],
                    "blockers": verdict["blockers"],
                    "cold_blockers": verdict["cold_blockers"],
                },
                "capacity": {
                    "free_bytes": verdict["free_bytes"],
                    "required_bytes": verdict["need_bytes"],
                    "margin_bytes": (
                        None
                        if verdict["capacity"] is None
                        else verdict["capacity"].margin_bytes
                    ),
                    "reserve_bytes": self.profile.reserve_bytes,
                    "profile_complete": self.profile.complete,
                },
                "fault": self.fault_reason,
                "last_error": self.last_error,
            }


def _envelope_equal(observed: Any, expected: Any) -> bool:
    """Typed comparison for calibrated-envelope values (F15).

    ``str(observed) == str(expected)`` cannot distinguish ``int 1`` from
    ``"1"`` — a container that reports a string where the calibration pinned an
    int (or vice versa) is a *type drift* that says the effective load did not
    follow the calibrated path, and must be reported rather than silently
    accepted. Numbers compare across int/float (131072 vs 131072.0 is the same
    setting, not a divergence); everything else compares by type and value.
    """

    if isinstance(expected, bool) or isinstance(observed, bool):
        return type(observed) is type(expected) and observed == expected
    if isinstance(expected, (int, float)) and isinstance(observed, (int, float)):
        return float(observed) == float(expected)
    if type(observed) is not type(expected):
        return False
    return observed == expected


def _first_reason(blockers: list[str]) -> Reason:
    """Map a blocker string back to a stable reason code."""
    if not blockers:
        return Reason.OK
    head = blockers[0]
    for reason in Reason:
        if head.startswith(reason.value):
            return reason
    if head.startswith("external_workload"):
        return Reason.EXTERNAL_GPU_BUSY
    if head.startswith("model_recovery"):
        # A generator recovery is a backend transition: the same "retry after
        # the mutation settles" family as model_transition (R07/F12).
        return Reason.MODEL_TRANSITION
    if head.startswith("quiet_window"):
        return Reason.GPU_NOT_QUIET
    if head.startswith("telemetry"):
        return Reason.TELEMETRY_UNAVAILABLE
    if head.startswith("request_capacity"):
        return Reason.REQUEST_CAPACITY
    if head.startswith("model_transition"):
        return Reason.MODEL_TRANSITION
    if head.startswith("lifecycle_not_ready") or head.startswith("lifecycle_not_unloaded"):
        # A lifecycle-state-only denial is a transition condition: the client
        # should retry once the model settles, and must not read `ok` (R12).
        return Reason.MODEL_TRANSITION
    return Reason.OK


def _safe_busy(deps: CoordinatorDeps) -> bool:
    with contextlib.suppress(Exception):
        return bool(deps.backend_busy())
    return True  # unknown means "assume busy" so we never unload under work


def _safe_call(fn: Callable[[], Any]) -> Any:
    with contextlib.suppress(Exception):
        return fn()
    return None


def _round_or_none(value: Optional[float], digits: int = 2) -> Optional[float]:
    """Round for display, preserving 'unknown' as null rather than 0."""
    if value is None:
        return None
    return round(value, digits)


def _owned_bytes(snap: Optional[Snapshot]) -> Optional[int]:
    if snap is None:
        return None
    return sum(p.used_bytes for p in snap.processes if p.is_owned)


def _max_external_sm(snap: Optional[Snapshot]) -> Optional[int]:
    if snap is None or snap.activity_status is not ActivityStatus.AVAILABLE:
        return None
    external = {p.pid for p in snap.processes if not p.is_owned}
    values = [a.sm_percent for a in snap.activity if a.pid in external and a.sm_percent is not None]
    return max(values) if values else None


__all__ = [
    "AdmissionOutcome",
    "AdmissionResult",
    "CoordinatorDeps",
    "Lease",
    "LeaseKind",
    "Lifecycle",
    "LifecycleCoordinator",
    "LifecycleError",
    "Waiter",
]
