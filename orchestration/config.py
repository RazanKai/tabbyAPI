"""Typed configuration for the orchestrator, plugged into TabbyAPI's own system.

Behavior belongs in TabbyAPI YAML, not in new environment variables (SPEC R14).
This module defines a single top-level ``orchestrator:`` section plus its nested
models, mirroring ``common/config_models.py`` conventions: one model per section, a
docstring upstream reuses as the YAML comment, and concrete defaults on every field.

Registration happens once, in ``orchestration/install.py``, which appends the field
to ``TabbyConfigModel``. Because upstream derives its CLI flags, environment
overrides and sample-config emission by iterating ``TabbyConfigModel.model_fields``,
registering the field is sufficient — nothing else needs to know about it.

Defaults are small and documented, and the default state is *disabled and
uncalibrated*: null footprint values deliberately prevent enablement until a
calibration exists (SPEC section 8).
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, model_validator

MIB = 1024 * 1024


def _base_config_model():
    """Return upstream's ``BaseConfigModel`` when available.

    Resolved lazily so this module stays importable in a standalone test session
    where the server's dependencies are absent. Tests exercise the same field
    declarations either way; only config-file emission differs.
    """
    try:
        from common.config_models import BaseConfigModel as upstream_base

        return upstream_base
    except Exception:  # noqa: BLE001 - standalone test context
        return BaseModel


_Base = _base_config_model()


class OrchestratorModelConfig(_Base):  # type: ignore[misc,valid-type]
    """The single configured model, its calibrated footprint, and its load envelope.

    ``resident_delta_mib``, ``load_peak_delta_mib``, ``request_peak_extra_mib`` and
    ``calibration_id`` are null until measured. A null footprint is not zero: it
    makes the profile incomplete, which blocks enablement.

    The envelope keys (``max_seq_len``, ``cache_size``, ``cache_mode``,
    ``chunk_size``, ``max_batch_size``) bind the load to the calibration (R14):
    at enablement they are required, the coordinator's load passes them as
    explicit kwargs (so ``model.use_as_default`` cannot silently alter the
    footprint the calibration measured), and after every load the real container
    is verified against them — a mismatch is a profile violation and latches
    FAULT rather than serving an uncalibrated envelope.
    """

    name: Optional[str] = Field(
        None, description="Canonical configured model id, exactly as TabbyAPI lists it."
    )
    resident_delta_mib: Optional[int] = Field(
        None, ge=0, description="Settled weights/cache allocation above the unloaded server baseline."
    )
    load_peak_delta_mib: Optional[int] = Field(
        None, ge=0, description="Largest additional allocation observed during loading."
    )
    request_peak_extra_mib: Optional[int] = Field(
        None, ge=0, description="Peak additional allocation above settled residency for the supported request envelope."
    )
    calibration_id: Optional[str] = Field(
        None, description="Identity of the measurement this profile was calibrated from."
    )
    max_seq_len: Optional[int] = Field(
        None, ge=1, description="Calibrated context length. Bound to the load and verified post-load."
    )
    cache_size: Optional[int] = Field(
        None, ge=1, description="Calibrated cache size in tokens. Bound to the load and verified post-load."
    )
    cache_mode: Optional[str] = Field(
        None, description="Calibrated cache mode (e.g. 'FP16', 'Q4', '8,8'). Verified post-load."
    )
    chunk_size: Optional[int] = Field(
        None, gt=0, description="Calibrated prompt-ingestion chunk size. Verified post-load."
    )
    max_batch_size: Optional[int] = Field(
        None, ge=1, description="Calibrated maximum batch size. Verified post-load."
    )


class OrchestratorTelemetryConfig(_Base):  # type: ignore[misc,valid-type]
    """Sampling cadence and freshness bound for the background sampler."""

    sample_seconds: float = Field(1.0, gt=0, description="Seconds between telemetry samples.")
    max_age_seconds: float = Field(
        3.0, gt=0, description="A snapshot older than this is stale and cannot admit new work."
    )


class OrchestratorColdLoadConfig(_Base):  # type: ignore[misc,valid-type]
    """Quietness requirements for starting a *new* load.

    Intentionally a different test from external-workload priority: a moderate
    device load may block a new cold load without forcing a resident model to drain.
    """

    max_device_utilization_percent: int = Field(
        15, ge=0, le=100, description="Device utilization ceiling considered quiet for a cold load."
    )
    quiet_seconds: float = Field(
        10.0, ge=0, description="Continuous quiet/stable window required before loading."
    )
    max_external_vram_growth_mib: int = Field(
        128, ge=0, description="Tolerated external VRAM growth across the quiet window; growth resets it."
    )


class OrchestratorExternalWorkloadConfig(_Base):  # type: ignore[misc,valid-type]
    """External-workload priority thresholds, carrying the committed R04 calibration.

    These defaults are NOT candidate guesses: they are the calibration committed by
    M4.2 from measured traces on the target host (worst normal-desktop single
    process 596 MiB, desktop aggregate max 1058 MiB, lowest target-workload floor
    1488 MiB). `orchestration/config.py` and
    `evidence/m4-calibration/r04-threshold-aggregate-v2.json` must agree on every
    number here — `tests/orchestration/test_policy.py` binds the two, because a
    calibration that is committed to a document but not to the shipped defaults is
    the defect this test exists to prevent.

    A different host must recalibrate (SPEC R04) and update the artefact and these
    defaults together.
    """

    process_vram_enter_mib: int = Field(
        1152, ge=0, description="Largest single external process VRAM that asserts priority."
    )
    process_vram_release_mib: int = Field(
        896, ge=0, description="Largest single external process VRAM below which the memory trigger clears."
    )
    total_vram_enter_mib: int = Field(
        1984, ge=0, description="Aggregate external VRAM that asserts priority."
    )
    total_vram_release_mib: int = Field(
        1600, ge=0, description="Aggregate external VRAM below which the memory trigger clears."
    )
    process_activity_enter_percent: int = Field(
        25, ge=0, le=100, description="Per-process SM activity that asserts priority."
    )
    process_activity_release_percent: int = Field(
        10, ge=0, le=100, description="Per-process SM activity below which the activity trigger clears."
    )
    enter_seconds: float = Field(
        2.0, ge=0, description="Continuous qualifying evidence required before the veto is asserted."
    )
    release_seconds: float = Field(
        10.0, ge=0, description="Continuous sub-threshold evidence required before the veto releases."
    )


class OrchestratorVramConfig(_Base):  # type: ignore[misc,valid-type]
    """Capacity reserve. Required for enablement and never inferred."""

    reserve_mib: Optional[int] = Field(
        None, ge=0, description="Calibrated safety margin held free at all times. No shipped guess."
    )


class OrchestratorAdmissionConfig(_Base):  # type: ignore[misc,valid-type]
    """Request admission limits and the pre-lease deadline."""

    max_active_requests: int = Field(
        1, ge=1, le=1, description="V1 supports exactly one admitted inference request at a time."
    )
    mode: str = Field(
        "reject",
        pattern="^(reject|wait)$",
        description=(
            "'reject' fails promptly without queueing; 'wait' holds the request on a "
            "bounded in-memory FIFO until pre-lease conditions clear or the deadline "
            "expires (R09)."
        ),
    )
    max_pending_requests: int = Field(
        8, ge=0, description="Wait-list capacity in 'wait' mode. Overflow fails immediately with 429."
    )
    max_wait_seconds: float = Field(
        60.0, gt=0,
        description=(
            "Pre-lease deadline in BOTH modes: all pre-lease time (quiet-window wait, "
            "a shared load, wait-mode queueing) is bounded by it. Expiry is a 503 "
            "admission_timeout and no inference was started for that request."
        ),
    )


class OrchestratorIdleUnloadConfig(_Base):  # type: ignore[misc,valid-type]
    """Idle TTL before an unused resident model is unloaded."""

    seconds: float = Field(
        300.0, gt=0, description="No admitted inference for this long unloads the model, keeping the API process."
    )


class OrchestratorConfig(_Base):  # type: ignore[misc,valid-type]
    """Resource-aware model lifecycle management (experimental).

    When ``enabled`` is false, TabbyAPI behaves exactly as upstream: no monitor
    starts and no request path changes. When true, this process becomes the single
    lifecycle authority for one fixed model profile on one GPU.
    """

    enabled: bool = Field(
        False, description="Enable orchestrated lifecycle management. Requires a calibrated profile."
    )
    device_uuid: Optional[str] = Field(
        None, description="GPU UUID to manage, as reported by NVML."
    )
    start_paused: bool = Field(
        False, description="Start with admission paused; process-local and reset to this on restart."
    )
    model: OrchestratorModelConfig = Field(default_factory=OrchestratorModelConfig)
    telemetry: OrchestratorTelemetryConfig = Field(default_factory=OrchestratorTelemetryConfig)
    cold_load: OrchestratorColdLoadConfig = Field(default_factory=OrchestratorColdLoadConfig)
    external_workload: OrchestratorExternalWorkloadConfig = Field(
        default_factory=OrchestratorExternalWorkloadConfig
    )
    vram: OrchestratorVramConfig = Field(default_factory=OrchestratorVramConfig)
    admission: OrchestratorAdmissionConfig = Field(default_factory=OrchestratorAdmissionConfig)
    idle_unload: OrchestratorIdleUnloadConfig = Field(default_factory=OrchestratorIdleUnloadConfig)

    @model_validator(mode="after")
    def _validate_threshold_ordering(self):
        """Release thresholds must be strictly below entry thresholds (R14)."""
        from .config import threshold_ordering_errors

        problems = threshold_ordering_errors(self.external_workload)
        if problems:
            raise ValueError("; ".join(problems))
        return self


class ConfigurationError(Exception):
    """Raised when an enabled orchestration configuration cannot be served safely.

    Distinct from upstream's bare ``pydantic.ValidationError``: SPEC R14 requires a
    *clear* startup error, so enabled-mode problems are collected and reported
    together instead of surfacing as a traceback.
    """


def envelope_errors(cfg: OrchestratorConfig) -> list[str]:
    """Problems with the calibrated load envelope (R14/R05).

    A calibration is only meaningful for a specific load envelope. When the
    profile is complete, every envelope key must be present too: an unbound
    envelope would let ``model.use_as_default`` or the model folder's
    ``tabby_config.yml`` change the very footprint the calibration measured.

    F15b: ``chunk_size`` must already be normalised the way the backend loads
    it (upstream rounds the effective value up to a multiple of 256), because
    the post-load verification compares the *container's* effective value
    against the calibrated one — a calibrated 1000 would become 1024 in the
    container and read back as a spurious profile FAULT. Requiring the
    calibrated value to be expressible exactly is honest: the operator either
    records the normalised number (1024) or reduces the envelope.
    """

    errors: list[str] = []
    for key in ("max_seq_len", "cache_size", "cache_mode", "chunk_size", "max_batch_size"):
        if getattr(cfg.model, key) is None:
            errors.append(f"orchestrator.model.{key} is required to bind the calibrated envelope")
    chunk = cfg.model.chunk_size
    if chunk is not None and chunk % 256 != 0:
        errors.append(
            f"orchestrator.model.chunk_size must be a multiple of 256 (the backend "
            f"normalises the effective value up to one, so a calibrated {chunk} would "
            f"verify against {((chunk + 255) // 256) * 256} and latch a spurious FAULT); "
            f"record the normalised value instead"
        )
    # Same class of mismatch as F15b (review nit): the backend CLAMPS max_seq_len
    # down to cache_size, so a calibrated max_seq_len above the cached bound is
    # reported back lower by model_info() and latches a FAULT for a profile the
    # operator cannot actually have.
    max_seq_len = cfg.model.max_seq_len
    cache_size = cfg.model.cache_size
    if (
        max_seq_len is not None
        and cache_size is not None
        and max_seq_len > cache_size
    ):
        errors.append(
            f"orchestrator.model.max_seq_len ({max_seq_len}) must not exceed "
            f"cache_size ({cache_size}): the backend clamps the effective context to "
            f"the cache bound, so the loaded container would report {cache_size} and "
            f"latch a spurious FAULT"
        )
    return errors


def threshold_ordering_errors(ew) -> list[str]:
    """Release thresholds must be strictly below entry thresholds (R14).

    Single authority for the ordering rules: the pydantic model validator
    (:meth:`OrchestratorConfig._validate_threshold_ordering`) and
    :func:`enablement_errors` both call this, so the rule exists in exactly one
    place instead of three.
    """

    errors: list[str] = []
    if ew.process_vram_release_mib >= ew.process_vram_enter_mib:
        errors.append(
            "orchestrator.external_workload.process_vram_release_mib must be < process_vram_enter_mib"
        )
    if ew.total_vram_release_mib >= ew.total_vram_enter_mib:
        errors.append(
            "orchestrator.external_workload.total_vram_release_mib must be < total_vram_enter_mib"
        )
    if ew.process_activity_release_percent >= ew.process_activity_enter_percent:
        errors.append(
            "orchestrator.external_workload.process_activity_release_percent must be "
            "< process_activity_enter_percent"
        )
    return errors


def enablement_errors(cfg: OrchestratorConfig) -> list[str]:
    """Reasons this configuration cannot be enabled, as readable messages.

    An empty list means the configuration is coherent enough to run. This is the
    startup gate R01/R14 require: incompatible runtime configuration is rejected
    rather than silently mis-served.
    """
    errors: list[str] = []
    if not cfg.enabled:
        return errors

    if not cfg.device_uuid:
        errors.append("orchestrator.device_uuid is required when enabled")
    if not cfg.model.name:
        errors.append("orchestrator.model.name is required when enabled")

    if cfg.model.resident_delta_mib is None:
        errors.append("orchestrator.model.resident_delta_mib is not calibrated")
    if cfg.model.load_peak_delta_mib is None:
        errors.append("orchestrator.model.load_peak_delta_mib is not calibrated")
    if cfg.model.request_peak_extra_mib is None:
        errors.append("orchestrator.model.request_peak_extra_mib is not calibrated")
    if not cfg.model.calibration_id:
        errors.append("orchestrator.model.calibration_id is required to identify the calibration")
    errors.extend(envelope_errors(cfg))
    if cfg.vram.reserve_mib is None:
        errors.append("orchestrator.vram.reserve_mib is required (no shipped guess)")

    if cfg.admission.max_active_requests != 1:
        errors.append("orchestrator.admission.max_active_requests must be 1 in V1")
    if cfg.admission.max_wait_seconds <= 0:
        errors.append("orchestrator.admission.max_wait_seconds must be positive")
    if cfg.admission.max_pending_requests < 0:
        errors.append("orchestrator.admission.max_pending_requests must be non-negative")

    errors.extend(threshold_ordering_errors(cfg.external_workload))

    if cfg.cold_load.quiet_seconds < 0:
        errors.append("orchestrator.cold_load.quiet_seconds must be non-negative")
    if cfg.telemetry.max_age_seconds <= 0:
        errors.append("orchestrator.telemetry.max_age_seconds must be positive")

    return errors


def require_valid_or_raise(cfg: OrchestratorConfig) -> None:
    """Raise ``ConfigurationError`` if enabled config is incoherent."""
    problems = enablement_errors(cfg)
    if problems:
        detail = "\n  - ".join(problems)
        raise ConfigurationError(
            "Orchestration is enabled but its configuration is not usable:\n  - " + detail
        )


__all__ = [
    "ConfigurationError",
    "MIB",
    "OrchestratorAdmissionConfig",
    "OrchestratorColdLoadConfig",
    "OrchestratorConfig",
    "OrchestratorExternalWorkloadConfig",
    "OrchestratorIdleUnloadConfig",
    "OrchestratorModelConfig",
    "OrchestratorTelemetryConfig",
    "OrchestratorVramConfig",
    "enablement_errors",
    "envelope_errors",
    "require_valid_or_raise",
    "threshold_ordering_errors",
]
