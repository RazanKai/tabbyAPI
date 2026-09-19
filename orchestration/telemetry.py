"""GPU telemetry: NVML sampling and process attribution.

Scope and rules (SPEC R02/R03):

* One adapter over the maintained NVIDIA binding (``nvidia-ml-py``, imported as
  ``pynvml``). The human-readable ``nvidia-smi`` table is a diagnostic cross-check
  only and is never parsed as the production API.
* Samples are immutable and carry monotonic capture time, the GPU UUID, raw byte
  values, device activity, graphics/compute process rows, per-process identity and
  validity. ``None`` always means "not known", never 0.
* Missing essential data fails *closed*: a caller deciding admission must treat an
  invalid or stale snapshot as "do not admit", not as "nothing is happening".
* ``NVML_ERROR_NOT_FOUND`` means "no samples for this call right now" — a normal,
  expected idle state — and is reported as ``SPARSE``. It is emphatically not
  ``NOT_SUPPORTED`` and must never be converted to a numeric zero (R04, T03).

Owned-identity note (R03): NVML does not expose process start time, so identity is
``(boot_id, pid, start_ticks)`` read from ``/proc``. This module only *resolves*
identities; it does not decide what is owned — that belongs to the lifecycle
coordinator, which is the only component that knows which PIDs it started.
"""

from __future__ import annotations

import os
import pathlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

BOOT_ID_PATH = pathlib.Path("/proc/sys/kernel/random/boot_id")

MIB = 1024 * 1024


class ActivityStatus(str, Enum):
    """Outcome of a per-process activity query (R04).

    The distinction between these is load-bearing: only ``AVAILABLE`` may be used
    to assert or clear activity, while ``SPARSE`` must not manufacture a trigger.
    """

    AVAILABLE = "available"
    SPARSE = "sparse"
    UNSUPPORTED = "unsupported"
    ERROR = "error"


@dataclass(frozen=True)
class ProcessIdentity:
    """Stable identity for a process across PID reuse (R03)."""

    boot_id: str
    pid: int
    start_ticks: int

    @property
    def key(self) -> str:
        return f"{self.boot_id}:{self.pid}:{self.start_ticks}"


@dataclass(frozen=True)
class ProcessSample:
    """One GPU process as reported by the driver at a single instant.

    ``is_owned`` is resolved by the sampler against the coordinator's verified
    identity set. It is false for an unresolved identity: an unknown process is
    external pressure, never silently owned (R03).
    """

    pid: int
    used_bytes: int
    sources: tuple[str, ...]  # which driver lists reported this pid
    identity: Optional[ProcessIdentity] = None
    comm: Optional[str] = None
    state: Optional[str] = None
    is_owned: bool = False

    @property
    def used_mib(self) -> float:
        return self.used_bytes / MIB


@dataclass(frozen=True)
class ProcessActivity:
    """Per-process activity at a single instant."""

    pid: int
    timestamp_us: Optional[int]
    sm_percent: Optional[int]
    mem_percent: Optional[int]
    enc_percent: Optional[int]
    dec_percent: Optional[int]


@dataclass(frozen=True)
class Snapshot:
    """An immutable observation of the GPU.

    ``valid`` is the single gate a policy caller must consult: when it is false the
    snapshot must not be used to *permit* anything.
    """

    captured_monotonic: float
    captured_epoch: float
    device_uuid: str
    gpu_uuid_ok: bool

    # Raw memory, in bytes, straight from NVML. `free` is the capacity authority.
    mem_total: Optional[int]
    mem_used: Optional[int]
    mem_free: Optional[int]
    # NVML v1 has no `reserved` field; derived as used - enumerated, informational only.
    mem_reserved_derived: Optional[int]

    device_utilization_percent: Optional[int]

    processes: tuple[ProcessSample, ...] = ()
    external_largest_bytes: int = 0
    external_total_bytes: int = 0
    external_count: int = 0

    activity_status: ActivityStatus = ActivityStatus.SPARSE
    activity_error: Optional[str] = None
    activity: tuple[ProcessActivity, ...] = ()

    ownership_error: Optional[str] = None
    notes: tuple[str, ...] = field(default=())

    @property
    def valid(self) -> bool:
        """Essential telemetry present and self-consistent."""
        return (
            self.gpu_uuid_ok
            and self.mem_total is not None
            and self.mem_used is not None
            and self.mem_free is not None
            and self.mem_total > 0
            and self.mem_free >= 0
            and self.mem_used >= 0
            and self.ownership_error is None
        )

    def age_seconds(self, now_monotonic: Optional[float] = None) -> float:
        now = time.monotonic() if now_monotonic is None else now_monotonic
        return now - self.captured_monotonic

    @property
    def fresh_diagnostics(self) -> dict:
        """Bounded, secret-free diagnostic payload."""
        return {
            "device_uuid": self.device_uuid,
            "mem_total": self.mem_total,
            "mem_used": self.mem_used,
            "mem_free": self.mem_free,
            "mem_reserved_derived": self.mem_reserved_derived,
            "device_utilization_percent": self.device_utilization_percent,
            "external_largest_bytes": self.external_largest_bytes,
            "external_total_bytes": self.external_total_bytes,
            "external_count": self.external_count,
            "activity_status": self.activity_status.value,
            "activity_error": self.activity_error,
            "valid": self.valid,
            "ownership_error": self.ownership_error,
        }


# --------------------------------------------------------------------------- #
# Process identity from /proc (NVML cannot supply start time)
# --------------------------------------------------------------------------- #


def read_boot_id(path: pathlib.Path = BOOT_ID_PATH) -> Optional[str]:
    """Read the host boot id; ``None`` when unavailable (never guessed)."""
    try:
        return path.read_text(encoding="utf8").strip() or None
    except OSError:
        return None


def read_process_identity(
    pid: int, boot_id: Optional[str] = None
) -> Optional[ProcessIdentity]:
    """Resolve ``(boot_id, pid, start_ticks)`` for a live process.

    ``comm`` may contain spaces and parentheses (and, for a zombie, anything at all),
    so the command field is located by splitting on the LAST ``)`` rather than by
    tokenising from the front.
    """
    boot = boot_id or read_boot_id()
    if boot is None:
        return None
    try:
        raw = pathlib.Path(f"/proc/{pid}/stat").read_text(encoding="utf8")
    except OSError:
        return None
    close = raw.rfind(")")
    if close < 0:
        return None
    fields = raw[close + 2 :].split()
    # After the parenthesised comm, field 3 is 'state'; starttime is field 22.
    if len(fields) < 20:
        return None
    try:
        start_ticks = int(fields[19])
    except ValueError:
        return None
    return ProcessIdentity(boot_id=boot, pid=pid, start_ticks=start_ticks)


def read_process_comm(pid: int) -> Optional[str]:
    """Short name from /proc, as an explanation only — never an ownership test."""
    try:
        return pathlib.Path(f"/proc/{pid}/stat").read_text(encoding="utf8").split("(")[1].split(")")[0]
    except (OSError, IndexError):
        return None


def process_state(pid: int) -> Optional[str]:
    try:
        raw = pathlib.Path(f"/proc/{pid}/stat").read_text(encoding="utf8")
    except OSError:
        return None
    close = raw.rfind(")")
    if close < 0:
        return None
    fields = raw[close + 2 :].split()
    return fields[0] if fields else None


# --------------------------------------------------------------------------- #
# The NVML adapter
# --------------------------------------------------------------------------- #


class NvmlError(Exception):
    """Raised by the adapter for unrecoverable NVML problems."""


class TelemetryAdapter:
    """Thin, read-only wrapper over ``pynvml``.

    Every call is defensive: a driver error becomes ``None`` plus a recorded reason
    rather than an exception escaping into the request path. The adapter never
    raises into admission decisions; callers consult ``Snapshot.valid``.
    """

    def __init__(self, device_uuid: str):
        self.device_uuid = device_uuid
        self._pynvml = None
        self._handle = None
        self._boot_id: Optional[str] = None
        self._initialized = False
        self._init_error: Optional[str] = None
        # Capability flags are probed once; they describe the binding, not a sample.
        self._capabilities: dict[str, bool] = {}

    # -- lifecycle ---------------------------------------------------------- #

    def start(self) -> None:
        try:
            import pynvml  # imported here so importing this module never requires a GPU
        except ImportError as exc:  # pragma: no cover - environment dependent
            self._init_error = f"nvidia-ml-py not importable: {exc}"
            raise NvmlError(self._init_error) from exc

        self._pynvml = pynvml
        try:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByUUID(self.device_uuid.encode())
        except Exception as exc:  # noqa: BLE001 - report, do not die
            self._init_error = f"NVML init failed: {type(exc).__name__}: {exc}"
            raise NvmlError(self._init_error) from exc

        self._boot_id = read_boot_id()
        self._initialized = True
        self._capabilities = {
            "memory_info": hasattr(pynvml, "nvmlDeviceGetMemoryInfo"),
            "memory_info_v2": hasattr(pynvml, "nvmlDeviceGetMemoryInfo_v2"),
            "utilization": hasattr(pynvml, "nvmlDeviceGetUtilizationRates"),
            "compute_processes": hasattr(pynvml, "nvmlDeviceGetComputeRunningProcesses"),
            "graphics_processes": hasattr(pynvml, "nvmlDeviceGetGraphicsRunningProcesses"),
            "process_utilization": hasattr(pynvml, "nvmlDeviceGetProcessUtilization"),
            "encoder_utilization": hasattr(pynvml, "nvmlDeviceGetEncoderUtilization"),
            "decoder_utilization": hasattr(pynvml, "nvmlDeviceGetDecoderUtilization"),
        }

    def stop(self) -> None:
        if self._pynvml is not None and self._initialized:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:  # noqa: BLE001 - shutdown must not raise
                pass
        self._initialized = False

    @property
    def boot_id(self) -> Optional[str]:
        return self._boot_id

    @property
    def capabilities(self) -> dict[str, bool]:
        return dict(self._capabilities)

    @property
    def init_error(self) -> Optional[str]:
        return self._init_error

    # -- sampling ----------------------------------------------------------- #

    def _error_value(self, exc: Exception) -> Optional[int]:
        value = getattr(exc, "value", None)
        return int(value) if isinstance(value, int) else None

    def _classify_activity_error(self, exc: Exception) -> ActivityStatus:
        pynvml = self._pynvml
        value = self._error_value(exc)
        if value is None:
            return ActivityStatus.ERROR
        if value == getattr(pynvml, "NVML_ERROR_NOT_SUPPORTED", 3):
            return ActivityStatus.UNSUPPORTED
        if value == getattr(pynvml, "NVML_ERROR_NOT_FOUND", 6):
            # "no samples for this call" — normal while nothing is doing work.
            return ActivityStatus.SPARSE
        return ActivityStatus.ERROR

    def _memory(self, notes: list[str]) -> tuple[Optional[int], Optional[int], Optional[int]]:
        pynvml = self._pynvml
        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"memory query failed: {type(exc).__name__}")
            return None, None, None
        return mem.total, mem.used, mem.free

    def _utilization(self, notes: list[str]) -> Optional[int]:
        try:
            return int(self._pynvml.nvmlDeviceGetUtilizationRates(self._handle).gpu)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"utilization query failed: {type(exc).__name__}")
            return None

    def _rows(self, fn_name: str, label: str, notes: list[str]) -> list[tuple[int, int]]:
        fn = getattr(self._pynvml, fn_name, None)
        if fn is None:
            notes.append(f"{label} process query unavailable in binding")
            return []
        try:
            entries = fn(self._handle)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"{label} process query failed: {type(exc).__name__}")
            return []
        out = []
        for entry in entries:
            pid = getattr(entry, "pid", None)
            used = getattr(entry, "usedGpuMemory", None)
            # The driver reports 0xFFFFFFFF for "no GPU instance" on non-MIG setups;
            # instance ids are not needed for attribution, only memory is.
            if pid is None:
                continue
            out.append((int(pid), int(used or 0)))
        return out

    def _activity(
        self, notes: list[str]
    ) -> tuple[ActivityStatus, tuple["ProcessActivity", ...], Optional[str]]:
        fn = getattr(self._pynvml, "nvmlDeviceGetProcessUtilization", None)
        if fn is None:
            return ActivityStatus.UNSUPPORTED, (), "nvmlDeviceGetProcessUtilization absent"
        try:
            rows = fn(self._handle, 0)
        except Exception as exc:  # noqa: BLE001
            status = self._classify_activity_error(exc)
            return status, (), f"{type(exc).__name__}: {exc}"
        if not rows:
            return ActivityStatus.SPARSE, (), "empty result (no samples)"
        acts = tuple(
            ProcessActivity(
                pid=int(getattr(r, "pid", -1)),
                timestamp_us=getattr(r, "timeStamp", None),
                sm_percent=getattr(r, "smUtil", None),
                mem_percent=getattr(r, "memUtil", None),
                enc_percent=getattr(r, "encUtil", None),
                dec_percent=getattr(r, "decUtil", None),
            )
            for r in rows
            if getattr(r, "pid", None) is not None
        )
        return ActivityStatus.AVAILABLE, acts, None

    def sample(self, excluded_identities: frozenset[str] = frozenset()) -> Snapshot:
        """Take one snapshot.

        ``excluded_identities`` are identity keys the coordinator has verified as
        its own. Anything not in that set is external. An unresolved identity is
        never excluded (R03) — it stays counted as external pressure so that a
        failed lookup cannot silently inflate capacity.
        """
        notes: list[str] = []
        now_mono = time.monotonic()
        now_epoch = time.time()

        gpu_uuid = self.device_uuid
        gpu_uuid_ok = True
        try:
            reported = self._pynvml.nvmlDeviceGetUUID(self._handle)
            if isinstance(reported, bytes):
                reported = reported.decode()
            gpu_uuid = reported
        except Exception as exc:  # noqa: BLE001
            gpu_uuid_ok = False
            notes.append(f"device UUID query failed: {type(exc).__name__}")

        total, used, free = self._memory(notes)
        utilization = self._utilization(notes)

        # Graphics and compute are separate driver lists and MAY describe the same
        # process (renders and computes). Deduplicate by pid, keeping the largest
        # reported allocation; do not add the two lists together (R02).
        raw_rows: list[tuple[int, int, str]] = []
        for pid, used_bytes in self._rows(
            "nvmlDeviceGetComputeRunningProcesses", "compute", notes
        ):
            raw_rows.append((pid, used_bytes, "compute"))
        for pid, used_bytes in self._rows(
            "nvmlDeviceGetGraphicsRunningProcesses", "graphics", notes
        ):
            raw_rows.append((pid, used_bytes, "graphics"))

        merged: dict[int, dict] = {}
        for pid, used_bytes, source in raw_rows:
            slot = merged.setdefault(pid, {"used": 0, "sources": set()})
            slot["used"] = max(slot["used"], used_bytes)
            slot["sources"].add(source)

        ownership_error: Optional[str] = None
        processes: list[ProcessSample] = []
        external_largest = 0
        external_total = 0
        external_count = 0
        owned_total = 0

        for pid, slot in sorted(merged.items()):
            identity = read_process_identity(pid, self._boot_id)
            is_owned = identity is not None and identity.key in excluded_identities
            sample = ProcessSample(
                pid=pid,
                used_bytes=slot["used"],
                sources=tuple(sorted(slot["sources"])),
                identity=identity,
                comm=read_process_comm(pid),
                state=process_state(pid),
                is_owned=is_owned,
            )
            processes.append(sample)

            if is_owned:
                owned_total += slot["used"]
                continue
            # An identity that could not be resolved is NOT owned-by-default.
            external_count += 1
            external_total += slot["used"]
            external_largest = max(external_largest, slot["used"])

        activity_status, activity, activity_error = self._activity(notes)

        # Informational only, and deliberately named as a *gap*: it is device used
        # memory minus everything the driver attributed to a process. On the measured
        # host this includes the driver's own reservation. It is NEVER an admission
        # operand — device free memory is the capacity authority (R03).
        attribution_gap = None
        if used is not None:
            attribution_gap = used - external_total - owned_total

        return Snapshot(
            captured_monotonic=now_mono,
            captured_epoch=now_epoch,
            device_uuid=gpu_uuid,
            gpu_uuid_ok=gpu_uuid_ok,
            mem_total=total,
            mem_used=used,
            mem_free=free,
            mem_reserved_derived=attribution_gap,
            device_utilization_percent=utilization,
            processes=tuple(processes),
            external_largest_bytes=external_largest,
            external_total_bytes=external_total,
            external_count=external_count,
            activity_status=activity_status,
            activity_error=activity_error,
            activity=activity,
            ownership_error=ownership_error,
            notes=tuple(notes),
        )


def identity_key_for(pid: int, boot_id: Optional[str] = None) -> Optional[str]:
    """Convenience for the coordinator registering its own workers."""
    identity = read_process_identity(pid, boot_id)
    return identity.key if identity else None


__all__ = [
    "ActivityStatus",
    "NvmlError",
    "ProcessActivity",
    "ProcessIdentity",
    "ProcessSample",
    "Snapshot",
    "TelemetryAdapter",
    "identity_key_for",
    "read_boot_id",
    "read_process_comm",
    "read_process_identity",
]
