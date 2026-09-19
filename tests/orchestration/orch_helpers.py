"""Shared helpers and fixtures for orchestration tests.

These tests are **mocked** tier (SPEC section 10): injected clocks, injected
telemetry snapshots, fake load/unload callbacks. They prove control logic. They are
*not* GPU acceptance and must never be reported as such.

Import convention follows upstream's ``tests/_common.py``: a plain module imported
by name, because the test directory is not a package.

The fakes are deliberately faithful in the ways that matter:

* ``FakeDeps`` records call ordering, so "one load for N requests" is asserted from
  observed behaviour rather than from a return value.
* ``FakeDeps.backend_busy`` is controllable, so lease-vs-teardown races can be
  exercised deterministically.
* ``make_snapshot`` mirrors the real sampler's field contract, including resolved
  ownership — a snapshot fake that omitted ownership would let an exclusion test
  pass while proving nothing.
"""

from __future__ import annotations

import asyncio
import pathlib
import sys

import pytest

# Make the pinned TabbyAPI checkout importable for the orchestration package.
_SRC = pathlib.Path(__file__).resolve().parent.parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from orchestration.config import (  # noqa: E402
    OrchestratorConfig,
    OrchestratorModelConfig,
    OrchestratorVramConfig,
)
from orchestration.telemetry import (  # noqa: E402
    ActivityStatus,
    ProcessActivity,
    ProcessIdentity,
    ProcessSample,
    Snapshot,
)

MIB = 1024 * 1024


class FakeClock:
    """A monotonic clock the test drives explicitly."""

    def __init__(self, start: float = 1000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeDeps:
    """Fake TabbyAPI lifecycle callbacks with observable ordering."""

    def __init__(
        self,
        *,
        load_seconds: float = 0.0,
        fail_load: bool = False,
        fail_unload: bool = False,
        fail_load_but_container: bool = False,
        clock: FakeClock | None = None,
    ):
        self.calls: list[str] = []
        self.load_seconds = load_seconds
        self.fail_load = fail_load
        self.fail_unload = fail_unload
        self.fail_load_but_container = fail_load_but_container
        self._clock = clock or FakeClock()
        self._present = False
        self._busy = False
        self.identity = "fake-container-1"
        self.load_count = 0
        self.unload_count = 0
        #: Optional fake generator-recovery task (F12). A test sets this to a
        #: real asyncio.Task to model the backend's untracked recovery; None
        #: (the default) means no recovery is running.
        self.recovery_task: "asyncio.Task | None" = None
        #: Models a recovery the coordinator has NOT adopted yet (review major):
        #: the container reports it via `recovery_in_progress()` while
        #: `adopt_recovery_task()` has not been called.
        self.recovery_running: bool = False
        #: Optional hook invoked after the fake clock advances during a load, so a
        #: test can keep the telemetry snapshot fresh the way a real sampler would.
        self.on_clock_advance = None

    def as_deps(self):
        from orchestration.lifecycle import CoordinatorDeps

        return CoordinatorDeps(
            load_model=self.load_model,
            unload_model=self.unload_model,
            container_present=lambda: self._present,
            backend_busy=lambda: self._busy,
            container_identity=lambda: self.identity if self._present else None,
            container_envelope=lambda: {
                "max_seq_len": 4096,
                "cache_size": 4096,
                "cache_mode": "FP16",
                "chunk_size": 2048,
                "max_batch_size": 1,
            }
            if self._present
            else {},
            # F12: an optional fake recovery task the test controls directly by
            # assigning `deps.recovery_task`; None means "no recovery running",
            # the state every pre-existing test assumes.
            recovery_in_progress=lambda: self.recovery_running
            or (self.recovery_task is not None and not self.recovery_task.done()),
            adopt_recovery_task=lambda: self.recovery_task
            if self.recovery_task is not None and not self.recovery_task.done()
            else None,
        )

    async def load_model(self):
        self.calls.append("load:start")
        self.load_count += 1
        if self.load_seconds:
            # Advance the fake clock instead of really sleeping, so tests stay fast
            # and deterministic. In production the background sampler keeps taking
            # snapshots throughout a load, so the hook below models that: without it
            # a long load would leave the coordinator holding a *stale* snapshot and
            # every post-load decision would (correctly, but uselessly) fail closed.
            self._clock.advance(self.load_seconds)
            if self.on_clock_advance is not None:
                self.on_clock_advance()
            await _yield_a_few_times()
        if self.fail_load:
            self.calls.append("load:fail")
            if self.fail_load_but_container:
                self._present = True
            raise RuntimeError("fake load failure")
        self._present = True
        self.calls.append("load:done")

    async def unload_model(self):
        self.calls.append("unload:start")
        self.unload_count += 1
        if self.fail_unload:
            self.calls.append("unload:fail")
            raise RuntimeError("fake unload failure")
        self._present = False
        self.calls.append("unload:done")

    def set_busy(self, busy: bool) -> None:
        self._busy = busy


async def _yield_a_few_times() -> None:
    """Give other tasks a chance to observe the in-flight transition."""
    import asyncio

    for _ in range(3):
        await asyncio.sleep(0)


class FakeTelemetry:
    """Stand-in for TelemetryAdapter that returns scripted snapshots."""

    def __init__(self, boot_id: str = "boot-fake"):
        self.boot_id = boot_id
        self.capabilities = {
            "memory_info": True,
            "process_utilization": True,
            "graphics_processes": True,
        }
        self.init_error = None

    def start(self):  # pragma: no cover - not used by unit tests
        pass

    def stop(self):  # pragma: no cover
        pass


def make_snapshot(
    *,
    captured: float,
    used: int = 2000 * MIB,
    free: int = 10000 * MIB,
    total: int = 12227 * MIB,
    utilization: int = 0,
    external: list[tuple[int, int]] | None = None,
    owned: list[tuple[int, int]] | None = None,
    activity_status: ActivityStatus = ActivityStatus.SPARSE,
    activity: list[tuple[int, int]] | None = None,
    valid: bool = True,
    ownership_error: str | None = None,
) -> Snapshot:
    """Build a snapshot shaped exactly like the sampler's output."""
    processes = []
    ext_total = 0
    ext_largest = 0
    for pid, used_bytes in external or []:
        ident = ProcessIdentity(boot_id="boot-fake", pid=pid, start_ticks=pid * 10)
        processes.append(
            ProcessSample(
                pid=pid,
                used_bytes=used_bytes,
                sources=("graphics",),
                identity=ident,
                comm=f"ext{pid}",
                state="S",
                is_owned=False,
            )
        )
        ext_total += used_bytes
        ext_largest = max(ext_largest, used_bytes)
    for pid, used_bytes in owned or []:
        ident = ProcessIdentity(boot_id="boot-fake", pid=pid, start_ticks=pid * 10)
        processes.append(
            ProcessSample(
                pid=pid,
                used_bytes=used_bytes,
                sources=("compute",),
                identity=ident,
                comm="tabby",
                state="S",
                is_owned=True,
            )
        )

    acts = tuple(
        ProcessActivity(
            pid=pid,
            timestamp_us=int(captured * 1_000_000),
            sm_percent=sm,
            mem_percent=sm,
            enc_percent=0,
            dec_percent=0,
        )
        for pid, sm in (activity or [])
    )

    snap = Snapshot(
        captured_monotonic=captured,
        captured_epoch=1_700_000_000.0 + captured,
        device_uuid="GPU-fake",
        gpu_uuid_ok=True,
        mem_total=total,
        mem_used=used,
        mem_free=free,
        mem_reserved_derived=used - ext_total,
        device_utilization_percent=utilization,
        processes=tuple(processes),
        external_largest_bytes=ext_largest,
        external_total_bytes=ext_total,
        external_count=len(external or []),
        activity_status=activity_status,
        activity_error="fake error" if activity_status is ActivityStatus.ERROR else None,
        activity=acts,
        ownership_error=ownership_error,
        notes=(),
    )
    if not valid:
        # Force invalidity the way the real sampler would: drop an essential metric.
        object.__setattr__(snap, "mem_free", None)
    return snap


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def calibrated_config() -> OrchestratorConfig:
    """An enabled, calibrated config — the state M0's calibration would produce."""
    return OrchestratorConfig(
        enabled=True,
        device_uuid="GPU-fake",
        model=OrchestratorModelConfig(
            name="Qwen3.8-27B-exl3-1.8",
            resident_delta_mib=8500,
            load_peak_delta_mib=9000,
            request_peak_extra_mib=700,
            calibration_id="test-calibration-v1",
        ),
        vram=OrchestratorVramConfig(reserve_mib=512),
    )


@pytest.fixture
def deps(clock) -> FakeDeps:
    return FakeDeps(clock=clock)
