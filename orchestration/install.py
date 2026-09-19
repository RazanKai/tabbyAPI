"""Production wiring for the orchestrator (SPEC R01, R12, R13).

This module is the only place that binds the coordinator to TabbyAPI's real
machinery, and the only place that decides whether the orchestration surface
exists at all:

* ``enable()`` — construct deps/telemetry/coordinator, start the sampler, and
  publish them on the module-level ``runtime`` holder. Disabled mode never
  calls it, so a disabled server exposes no coordinator, no lease helpers that
  do anything, and no changed request path (R01/R12/R13).
* ``install_config_section()`` — append ``orchestrator:`` to
  ``TabbyConfigModel``. Upstream's arg/env/file loaders iterate
  ``TabbyConfigModel.model_fields``, so one field registration feeds them all.
* ``disabled_mode_parity()`` — the R01/R13 startup checks for enabled mode:
  reject a nonempty startup ``model_name`` (it would bypass admission) and
  ``network.disable_auth`` (it makes every route admin-equivalent, which R01
  forbids for an orchestrator-governed server).

Importing this module has no side effects; every binding happens inside
``enable()`` / ``install_config_section()``, both of which are called only
from ``main.entrypoint_async`` in enabled mode.
"""

from __future__ import annotations

import asyncio
import contextlib
import pathlib
from typing import Any, Optional

MIB = 1024 * 1024

# Config-section registration happens only when install_config_section() is
# called, so importing this module never mutates upstream's config model.
_CONFIG_REGISTERED = False


def _log(message: str) -> None:
    """Lazy logging: common.logger pulls `requests`, which a minimal test
    environment does not carry. Resolved lazily like every other binding."""

    try:
        from common.logger import xlogger

        xlogger.info(f"orchestrator: {message}")
    except Exception:  # noqa: BLE001 - logging must never break wiring
        pass


# --------------------------------------------------------------------------- #
# Config section registration (R14)
# --------------------------------------------------------------------------- #


def install_config_section() -> None:
    """Append the ``orchestrator:`` section to TabbyConfigModel, once.

    Upstream derives CLI flags (common/args.py), env overrides
    (common/tabby_config.py:_from_environment), arg overrides (_from_args) and
    sample-config emission by iterating ``TabbyConfigModel.model_fields`` — so
    registering the field is the entire integration. The subclass
    ``TabbyConfig`` is rebuilt too: its ``model_fields`` are captured at class
    creation and a parent rebuild does not propagate to an already-created
    subclass.
    """

    global _CONFIG_REGISTERED
    if _CONFIG_REGISTERED:
        return

    from pydantic.fields import FieldInfo

    from common.config_models import TabbyConfigModel
    from orchestration.config import OrchestratorConfig

    if "orchestrator" in TabbyConfigModel.model_fields:  # defensive: idempotence
        _CONFIG_REGISTERED = True
        return

    info = FieldInfo(default_factory=OrchestratorConfig, annotation=Optional[OrchestratorConfig])
    TabbyConfigModel.model_fields["orchestrator"] = info
    TabbyConfigModel.model_rebuild(force=True)

    from common.tabby_config import TabbyConfig

    # A subclass captures its own model_fields at class creation and a parent
    # rebuild does not propagate (measured on pydantic 2.13.5), so mirror the
    # field into the subclass explicitly before rebuilding it.
    if "orchestrator" not in TabbyConfig.model_fields:
        TabbyConfig.model_fields["orchestrator"] = info
    TabbyConfig.model_rebuild(force=True)

    _CONFIG_REGISTERED = True


# --------------------------------------------------------------------------- #
# Enabled-mode startup checks (R01/R13)
# --------------------------------------------------------------------------- #


def disabled_mode_parity() -> Optional[str]:
    """Reject upstream startup surfaces that would bypass orchestration.

    Returns an error message, or None when the configuration may proceed.
    Called by ``enable()`` before anything is created. These are the
    incompatible-surface rejections R12/R01/T13 name: rather than serving
    ungoverned paths, startup fails with a clear error and the operator fixes
    the configuration.
    """

    from common.tabby_config import config

    if config.model.model_name:
        return (
            "Orchestration is enabled but model.model_name is set. Enabled mode "
            "requires an unloaded startup: remove model_name so the configured "
            "orchestrator profile is the only load path (R13)."
        )
    if config.network.disable_auth:
        return (
            "Orchestration is enabled but network.disable_auth is true. With auth "
            "disabled every request is admin-equivalent, which would open the "
            "orchestrator control routes to inference credentials (R01/R12)."
        )
    if "kobold" in (config.network.api_servers or []):
        return (
            "Orchestration is enabled but network.api_servers includes 'kobold'. "
            "The Kobold API surface is out of orchestrated V1 scope (R12/T13): "
            "remove 'kobold' from network.api_servers."
        )
    if config.model.use_dummy_models:
        return (
            "Orchestration is enabled but model.use_dummy_models is true. A dummy "
            "model would be served as a load bypass without any calibrated "
            "profile behind it (R01/T13); disable it."
        )
    if config.embeddings.embedding_model_name:
        return (
            "Orchestration is enabled but embeddings.embedding_model_name is set. "
            "The startup embedding load bypasses orchestration entirely (R13/T13); "
            "remove it. Embeddings are out of orchestrated V1 scope (SPEC §2)."
        )
    return None


# --------------------------------------------------------------------------- #
# Production CoordinatorDeps bindings
# --------------------------------------------------------------------------- #


def _production_deps() -> Any:
    """Build CoordinatorDeps bound to TabbyAPI's real model container.

    * ``backend_busy`` uses ``len(container.active_job_ids) > 0`` — the
      container's public job registry, which ``stream_generate``'s ``finally``
      maintains on both success and cancellation (integration map: the
      authoritative "backend work outstanding" signal) — OR a live generator
      recovery task, which cancels jobs first and would otherwise read as an
      empty registry mid-mutation (F12/R07). ``container.loaded``
      must NOT be trusted alone: it is set once at load and never reset by
      unload, so a torn-down container would report loaded=True.
    * ``container_identity`` pins the resolved model directory so a lease can
      detect a swap underneath it.
    """

    from common import model as model_module
    from orchestration.lifecycle import CoordinatorDeps

    def container_present() -> bool:
        container = model_module.container
        # The `loaded` flag is never reset by unload; require the weights to
        # actually exist (C3/INTEGRATION-MAP §8).
        return bool(
            container is not None and container.loaded and container.model is not None
        )

    def backend_busy() -> bool:
        container = model_module.container
        if container is None:
            return False
        with contextlib.suppress(Exception):
            if len(container.active_job_ids) > 0:
                return True
            # F12/R07: the recovery task cancels jobs FIRST, so an empty registry
            # is not proof of quiescence while recovery is mutating the
            # generator. Fold the recovery signal into the busy answer so every
            # synchronous caller (request_unload, pause, _safe_busy) sees it.
            task = getattr(container, "recovery_task", None)
            return task is not None and not task.done()
        return True  # unknown means busy; never unload under uncertain state

    def container_identity() -> Optional[str]:
        container = model_module.container
        if container is None:
            return None
        with contextlib.suppress(Exception):
            return str(container.model_dir)
        return None

    async def load_model() -> None:
        """Perform the configured profile's load through upstream machinery.

        The calibrated envelope keys are passed as EXPLICIT kwargs so they
        override ``model.use_as_default`` (upstream merges config defaults below
        request kwargs — integration map: apply_load_defaults priority). A
        calibration is only valid for the envelope it measured; letting YAML
        defaults silently alter it would make the footprint numbers lies (R05/R14).
        """

        from common.tabby_config import config as tabby_config

        orch_cfg = tabby_config.orchestrator
        model_path = pathlib.Path(tabby_config.model.model_dir) / orch_cfg.model.name
        # No draft overrides here: the calibrated profile is text-model-only
        # and draft parameters are outside V1's calibrated envelope (SPEC §2).
        envelope_kwargs = {
            key: getattr(orch_cfg.model, key)
            for key in ("max_seq_len", "cache_size", "cache_mode", "chunk_size", "max_batch_size")
            if getattr(orch_cfg.model, key) is not None
        }
        await model_module.load_model(model_path, **envelope_kwargs)

    async def unload_model() -> None:
        container = model_module.container
        if container is None:
            return  # idempotent by contract (CoordinatorDeps.unload_model)
        # Route through common/model.unload_model so the global container is
        # cleared on success — reconciling against a stale global would publish
        # a residency lie (R08).
        await model_module.unload_model()

    def container_envelope() -> dict[str, Any]:
        """The container's *effective* envelope, for post-load verification.

        Read from the backend's own accounting (the same fields /props
        publishes), not from any config — this is what actually got loaded.
        A value the container cannot report is returned as None and skipped
        by the verifier rather than failing the load by itself.
        """
        container = model_module.container
        if container is None:
            return {}
        with contextlib.suppress(Exception):
            info = container.model_info()
            params = info.parameters
            return {
                "max_seq_len": getattr(params, "max_seq_len", None) if params else None,
                "cache_size": getattr(params, "cache_size", None) if params else None,
                "cache_mode": getattr(params, "cache_mode", None) if params else None,
                "chunk_size": getattr(params, "chunk_size", None) if params else None,
                "max_batch_size": getattr(params, "max_batch_size", None) if params else None,
            }
        return {}

    def recovery_in_progress() -> bool:
        """True while the backend's generator-recovery task is running (F12).

        The recovery task cancels the container's jobs *before* rebuilding the
        generator, so the job registry (backend_busy) can read EMPTY mid-recovery
        while the backend is mid-mutation. This signal covers that blind spot.
        """
        container = model_module.container
        if container is None:
            return False
        task = getattr(container, "recovery_task", None)
        return bool(task is not None and not task.done())

    def adopt_recovery_task():
        """Hand the backend's in-flight recovery task to the coordinator (F12).

        Returns the task when it is live so the coordinator can retain and
        observe it; None when there is nothing to adopt. Never raises.
        """
        container = model_module.container
        if container is None:
            return None
        task = getattr(container, "recovery_task", None)
        if task is not None and not task.done():
            return task
        return None

    return CoordinatorDeps(
        load_model=load_model,
        unload_model=unload_model,
        container_present=container_present,
        backend_busy=backend_busy,
        container_identity=container_identity,
        container_envelope=container_envelope,
        recovery_in_progress=recovery_in_progress,
        adopt_recovery_task=adopt_recovery_task,
    )


# --------------------------------------------------------------------------- #
# Background telemetry sampler (R02)
# --------------------------------------------------------------------------- #


class TelemetrySampler:
    """Background task: one NVML snapshot per cadence, fed to the coordinator.

    R02: samples outside the event loop's blocking path, and a driver call that
    hangs is bounded by structure: exactly one in-flight sampling task, never a
    replacement chain. A failed or hung probe leaves the previous snapshot to
    age out — which fails admission *closed* — instead of manufacturing a quiet
    device. One sampling error is logged and skipped; the sampler task itself
    survives (a dead sampler would leave admission permanently failed closed
    with no diagnostics, which R02 forbids).
    """

    def __init__(self, coordinator: Any, telemetry: Any, sample_seconds: float):
        self.coordinator = coordinator
        self.telemetry = telemetry
        self.sample_seconds = sample_seconds
        self._task: Optional[asyncio.Task] = None
        self._wake = asyncio.Event()
        self.sample_errors = 0

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.get_running_loop().create_task(self._run())

    async def stop(self) -> None:
        task = self._task
        if task is None:
            return
        self._task = None
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    def wake(self) -> None:
        """Nudge the sampler (used at startup so status has data quickly)."""

        self._wake.set()

    async def _run(self) -> None:
        while True:
            try:
                snap = await asyncio.to_thread(
                    self.telemetry.sample,
                    self.coordinator.owned_identities,
                )
                self.coordinator.ingest_snapshot(snap)
                self.sample_errors = 0
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - sampler must never die
                self.sample_errors += 1
                _log(
                    f"sampler error ({self.sample_errors} total): "
                    f"{type(exc).__name__}: {exc}"
                )
            # Maintenance pass rides the same cadence as telemetry: the idle
            # TTL, drain progression and veto→drain transitions all advance
            # here. Without this, an idle model would stay resident forever.
            # Adoption runs BEFORE tick(): a backend recovery that started since
            # the last pass must be observed by this pass's decisions (F12/R07),
            # or the first post-recovery tick could still see an empty job
            # registry and unload mid-mutation.
            with contextlib.suppress(Exception):
                self.coordinator.adopt_recovery_task()
            with contextlib.suppress(Exception):
                action = await self.coordinator.tick()
                if action:
                    _log(f"maintenance: {action}")
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=self.sample_seconds)
            except asyncio.TimeoutError:
                pass
            self._wake.clear()


# --------------------------------------------------------------------------- #
# Runtime holder
# --------------------------------------------------------------------------- #


class _Runtime:
    """Module-level holder for the live orchestrator pieces.

    Exists because the request path must reach the coordinator without a
    circular import: endpoints import ``orchestration.install`` lazily inside
    functions, and this holder keeps that import cheap and side-effect free.
    Every attribute is None in disabled mode.
    """

    orchestrator: Any = None
    sampler: Any = None
    telemetry: Any = None


runtime = _Runtime()


def coordinator() -> Any:
    """The installed coordinator, or None in disabled mode."""

    return runtime.orchestrator


# --------------------------------------------------------------------------- #
# enable()
# --------------------------------------------------------------------------- #


def enable() -> None:
    """Build the coordinator and publish it on the ``runtime`` holder.

    Called from ``main.entrypoint_async`` before the API server starts. Raises
    ``ConfigurationError`` on an unusable enabled-mode configuration — startup
    must fail loudly rather than serve ungoverned (R14).
    """

    import os

    from common.tabby_config import config as tabby_config
    from orchestration.config import ConfigurationError, require_valid_or_raise
    from orchestration.lifecycle import LifecycleCoordinator
    from orchestration.telemetry import TelemetryAdapter

    orch_cfg = tabby_config.orchestrator
    require_valid_or_raise(orch_cfg)

    parity_error = disabled_mode_parity()
    if parity_error is not None:
        raise ConfigurationError(parity_error)

    telemetry = TelemetryAdapter(orch_cfg.device_uuid)
    try:
        telemetry.start()
    except Exception as exc:  # noqa: BLE001 - no telemetry, no governance
        raise ConfigurationError(f"orchestrator telemetry could not start: {exc}") from exc

    coordinator_obj = LifecycleCoordinator(
        orch_cfg,
        _production_deps(),
        telemetry,
        log=lambda msg: _log(msg),
    )

    # Register this server's own process identity so NVML rows for it are
    # excluded as owned (R03). Inference runs in-process on this container;
    # there are no child worker processes to register.
    coordinator_obj.register_owned_pid(os.getpid())

    runtime.orchestrator = coordinator_obj
    runtime.telemetry = telemetry
    runtime.sampler = TelemetrySampler(coordinator_obj, telemetry, orch_cfg.telemetry.sample_seconds)
    runtime.sampler.start()
    runtime.sampler.wake()


def start_sampler() -> None:
    sampler = runtime.sampler
    if sampler is not None:
        sampler.start()


async def stop_sampler() -> None:
    sampler = runtime.sampler
    if sampler is not None:
        await sampler.stop()


async def shutdown() -> dict:
    """Drain-aware shutdown hook (R13). Never raises."""

    coordinator_obj = runtime.orchestrator
    if coordinator_obj is None:
        return {"state": "disabled"}
    try:
        result = await coordinator_obj.begin_shutdown()
        sampler = runtime.sampler
        if sampler is not None:
            await sampler.stop()
        telemetry = runtime.telemetry
        if telemetry is not None:
            telemetry.stop()
        return result
    except Exception as exc:  # noqa: BLE001 - shutdown must not raise
        return {"state": "error", "error": f"{type(exc).__name__}: {exc}"}


# --------------------------------------------------------------------------- #
# Lease integration for the inference boundary (R06/R07)
# --------------------------------------------------------------------------- #


class LeaseContext:
    """Owns one request's lease from acquisition to verified backend quiescence.

    The lifecycle rules this enforces come from the integration map, not
    invented here:

    * The lease is acquired before the first live-container use (before the
      router ``load_lock`` block). A request that must wait therefore holds no
      upstream lock while waiting (integration map §1).
    * For streaming responses, returning the response object is NOT completion
      (integration map §2). ``observe`` hands the lease to a background
      observer that waits for the generation tasks to actually finish —
      success *and* cancellation both reach the backend job registry cleanup,
      so task completion is the authoritative in-process end signal. The
      wrapper ``finally`` releases only when no work was ever spawned (the C4
      early-raise paths).
    * ``release_in_finally`` is idempotent and never raises, so it is safe in
      every cleanup path.
    """

    __slots__ = ("lease", "_coordinator", "_observed")

    def __init__(self, lease: Any, coordinator: Any):
        self.lease = lease
        self._coordinator = coordinator
        self._observed = False


class LeaseDenied(Exception):
    """Admission refused; carries the R12-shaped HTTPException as ``exc``."""

    def __init__(self, exc: Any, reason: str):
        self.exc = exc
        self.reason = reason
        super().__init__(reason)


async def acquire_lease_now(request_id: str) -> Optional[LeaseContext]:
    """Acquire an inference lease immediately, or raise :class:`LeaseDenied`.

    ``None`` means the orchestrator is absent (disabled mode): the caller
    proceeds with plain upstream behaviour and no lease lifecycle at all.

    Denials are raised rather than returned so the router's existing
    ``except`` shape (raise → FastAPI error response) stays intact and no
    denial can be mistaken for a committed 200 (R09/R12).
    """

    coordinator_obj = runtime.orchestrator
    if coordinator_obj is None:
        return None

    from orchestration.api import admission_http_exception
    from orchestration.lifecycle import AdmissionOutcome

    result = await coordinator_obj.acquire_lease(request_id)
    if result.outcome is AdmissionOutcome.GRANTED:
        return LeaseContext(result.lease, coordinator_obj)

    exc = admission_http_exception(result)
    _log(f"denied request {request_id}: {result.reason.value}")
    raise LeaseDenied(exc, result.reason.value)


async def acquire_reader_pin(request_id: str) -> Any:
    """Acquire a short READER lease for a tokenization/metadata read (F10, R12).

    R12: a read that dereferences the live container needs a short lease or a
    safe snapshot; a read must never cold-load and never refresh the inference
    TTL — both properties are the READER lease kind's contract in the
    coordinator (reader pins are counted as outstanding model use for teardown
    decisions but do not reset the idle timer). While the model is not READY
    the pin is refused as ``model_transition`` rather than cold-loading (503),
    which is what makes "a read never cold-loads" true.

    Returns ``(context, coordinator)`` where ``context`` is None in disabled
    mode; callers release with :func:`release_reader_pin` in a ``finally``.
    """

    coordinator_obj = runtime.orchestrator
    if coordinator_obj is None:
        return None, None

    from orchestration.api import admission_http_exception
    from orchestration.lifecycle import AdmissionOutcome, LeaseKind

    result = await coordinator_obj.acquire_lease(
        request_id, kind=LeaseKind.READER
    )
    if result.outcome is AdmissionOutcome.GRANTED:
        return LeaseContext(result.lease, coordinator_obj), coordinator_obj

    exc = admission_http_exception(result)
    _log(f"denied reader pin {request_id}: {result.reason.value}")
    raise LeaseDenied(exc, result.reason.value)


def release_reader_pin(context: Any) -> None:
    """Release a reader pin; safe with None (disabled mode) and idempotent."""

    if context is None:
        return
    context._observed = True
    context._coordinator.release_lease_soon(context.lease)


def envelope_violation(*, choices: int, prompts: int) -> Optional[str]:
    """Reason a request's *shape* exceeds the one calibrated envelope (R06/R12).

    The calibration is measured for a single active request of a specific
    envelope; V1 supports exactly one admitted HTTP model request at a time.
    ``data.n > 1`` or multiple prompts spawns that many generation tasks while
    the capacity check charges a single ``request_peak_extra`` — so the shape
    must be refused explicitly with ``unsupported_profile`` rather than admitted
    on a budget that does not describe it (SPEC §2: "reject multi-choice/
    multi-prompt request shapes that exceed the calibrated envelope").

    Returns a readable reason, or None when the shape is within the envelope.
    """
    problems: list[str] = []
    if choices > 1:
        problems.append(f"n={choices} exceeds the calibrated single-choice envelope")
    if prompts > 1:
        problems.append(f"{prompts} prompts exceed the calibrated single-prompt envelope")
    if not problems:
        return None
    return "; ".join(problems)


class _BackendObserver:
    """Watches a request's generation tasks and releases the lease when done.

    Streaming wrappers never await ``gen_tasks`` (integration map §2), so the
    only in-process proof that backend work ended is the tasks finishing —
    which coincides with the backend job registry's own cleanup
    (``backends/exllamav3/model.py`` stream_generate ``finally``, reached on
    success and cancellation). The observer awaits the tasks under a shield so
    its own cancellation cannot strand the lease, then releases.
    """

    def __init__(self, coordinator_obj: Any, lease: Any):
        self.coordinator = coordinator_obj
        self.lease = lease

    async def run(self, gen_tasks: list) -> None:
        try:
            if gen_tasks:
                await asyncio.shield(asyncio.gather(*gen_tasks, return_exceptions=True))
        finally:
            await self.coordinator.release_lease(self.lease)


def release_lease_in_finally(context: Optional[LeaseContext], gen_tasks: list) -> None:
    """The single integration point the four wrapper ``finally`` blocks call.

    * Tasks present: delegate to the backend observer, which releases only
      after the tasks are really done. This is the streaming-correct release
      point — a bare release in the wrapper ``finally`` would fire while
      generation is still unwinding.
    * No tasks (early raise before collectors were created — the C4 paths):
      release immediately; no backend work can exist yet.
    """

    if context is None or context._observed:
        return
    context._observed = True
    try:
        tasks = list(gen_tasks)
        if tasks:
            asyncio.get_running_loop().create_task(
                _BackendObserver(context._coordinator, context.lease).run(tasks)
            )
        else:
            context._coordinator.release_lease_soon(context.lease)
    except Exception:  # noqa: BLE001 - cleanup must never mask the real error
        context._coordinator.release_lease_soon(context.lease)


__all__ = [
    "LeaseContext",
    "LeaseDenied",
    "TelemetrySampler",
    "acquire_lease_now",
    "acquire_reader_pin",
    "coordinator",
    "disabled_mode_parity",
    "enable",
    "envelope_violation",
    "install_config_section",
    "release_lease_in_finally",
    "release_reader_pin",
    "runtime",
    "shutdown",
    "start_sampler",
    "stop_sampler",
]