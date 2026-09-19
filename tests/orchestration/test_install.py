"""Mock-tier tests for orchestration/install.py — the production wiring.

Tier: **mocked** (injected clock, fakes, no GPU, no NVML). These prove the
wiring logic: config-section registration, enabled-mode parity checks,
CoordinatorDeps production bindings against a fake `common.model`, the
sampler loop, and the lease-context release/observer discipline that the
router and wrapper edits rely on.

Import convention matches the rest of this directory: helpers by module name,
no package semantics. Upstream modules that need fastapi are imported lazily
inside tests so the collection environment only needs pydantic + pytest.
"""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import orch_helpers  # noqa: E402
from orch_helpers import FakeClock, make_snapshot  # noqa: E402

from orchestration import install as install_module  # noqa: E402
from orchestration.lifecycle import (  # noqa: E402
    AdmissionOutcome,
    LeaseKind,
    Lifecycle,
    LifecycleCoordinator,
)


def _stub_upstream_optional_deps():
    """Minimal stand-ins for loguru + ruamel: the dev venv carries only
    pydantic+pytest, but common.tabby_config imports both at module level."""

    import types

    if "loguru" not in sys.modules:
        fake = types.ModuleType("loguru")
        fake.logger = types.SimpleNamespace(
            error=lambda *a, **k: None,
            warning=lambda *a, **k: None,
            info=lambda *a, **k: None,
            debug=lambda *a, **k: None,
        )
        sys.modules["loguru"] = fake

    if "ruamel.yaml" not in sys.modules:
        ruamel = types.ModuleType("ruamel")
        ruamel_yaml = types.ModuleType("ruamel.yaml")

        class _FakeYAML:  # noqa: D401 - minimal stand-in
            def __init__(self, *a, **k):
                pass

            def load(self, stream):
                return {}

            def dump(self, data, stream):
                pass

        ruamel_yaml.YAML = _FakeYAML
        ruamel_yaml.CommentedMap = dict
        ruamel_yaml.CommentedSeq = list
        ruamel_yaml.PreservedScalarString = str
        ruamel.yaml = ruamel_yaml
        sys.modules["ruamel"] = ruamel
        sys.modules["ruamel.yaml"] = ruamel_yaml
        sys.modules["ruamel.yaml.comments"] = types.ModuleType("ruamel.yaml.comments")
        sys.modules["ruamel.yaml.scalarstring"] = types.ModuleType(
            "ruamel.yaml.scalarstring"
        )
        sys.modules["ruamel.yaml.comments"].CommentedMap = dict
        sys.modules["ruamel.yaml.comments"].CommentedSeq = list
        sys.modules["ruamel.yaml.scalarstring"].PreservedScalarString = str


_stub_upstream_optional_deps()


def _fresh_coordinator(cfg, deps, clock, *, ready: bool = True):
    """A coordinator holding a fresh, valid, quiet snapshot (mocked tier).

    ``ready=True`` publishes residency so the lease tests exercise the *warm*
    admission path; the cold-load path with its quiet window is already covered
    by test_lifecycle.py.
    """

    telemetry = SimpleNamespace(boot_id="boot-fake", capabilities={}, init_error=None)
    coord = LifecycleCoordinator(cfg, deps.as_deps(), telemetry, clock=clock)
    coord.ingest_snapshot(
        make_snapshot(captured=clock(), free=10_000 * orch_helpers.MIB)
    )
    if ready:
        deps._present = True
        coord.lifecycle = Lifecycle.READY
    return coord


def _make_coordinator(cfg, deps, clock):
    telemetry = SimpleNamespace(boot_id="boot-fake", capabilities={}, init_error=None)
    return LifecycleCoordinator(cfg, deps.as_deps(), telemetry, clock=clock)


class ConfigSectionTests(unittest.TestCase):
    """install_config_section() registers the field upstream loaders iterate."""

    def test_registration_is_idempotent_and_feeds_loaders(self):
        from common.config_models import TabbyConfigModel
        from common.tabby_config import TabbyConfig
        from orchestration.config import OrchestratorConfig

        install_module.install_config_section()
        # Call again: must not duplicate or raise.
        install_module.install_config_section()

        self.assertIn("orchestrator", TabbyConfigModel.model_fields)
        self.assertIn("orchestrator", TabbyConfig.model_fields)

        # The env-loader pattern upstream uses (getattr on an instance).
        instance = TabbyConfigModel()
        self.assertIsInstance(instance.orchestrator, OrchestratorConfig)
        self.assertFalse(instance.orchestrator.enabled)

        # model_validate accepts the section (file-load path).
        validated = TabbyConfigModel.model_validate(
            {"orchestrator": {"enabled": True, "device_uuid": "GPU-x"}}
        )
        self.assertTrue(validated.orchestrator.enabled)

    def test_section_default_is_disabled_and_uncalibrated(self):
        from common.config_models import TabbyConfigModel

        cfg = TabbyConfigModel().orchestrator
        self.assertFalse(cfg.enabled)
        self.assertIsNone(cfg.model.resident_delta_mib)
        self.assertIsNone(cfg.vram.reserve_mib)


class DisabledModeParityTests(unittest.TestCase):
    def test_rejects_startup_model_name(self):
        from common.tabby_config import config

        with patch.object(config.model, "model_name", "some-model"):
            message = install_module.disabled_mode_parity()
        self.assertIsNotNone(message)
        self.assertIn("model_name", message)

    def test_rejects_disable_auth(self):
        from common.tabby_config import config

        with patch.object(config.network, "disable_auth", True):
            message = install_module.disabled_mode_parity()
        self.assertIsNotNone(message)
        self.assertIn("disable_auth", message)

    def test_accepts_a_clean_enabled_config(self):
        from common.tabby_config import config

        with patch.object(config.model, "model_name", None), patch.object(
            config.network, "disable_auth", False
        ):
            self.assertIsNone(install_module.disabled_mode_parity())

    def test_rejects_kobold_surface(self):
        """F10/T13: the Kobold surface is out of orchestrated V1 scope."""
        from common.tabby_config import config

        with patch.object(config.network, "api_servers", ["oai", "kobold"]):
            message = install_module.disabled_mode_parity()
        self.assertIsNotNone(message)
        self.assertIn("kobold", message)

    def test_rejects_dummy_models(self):
        """F10/T13: a dummy-model load bypass must not exist in enabled mode."""
        from common.tabby_config import config

        with patch.object(config.model, "use_dummy_models", True):
            message = install_module.disabled_mode_parity()
        self.assertIsNotNone(message)
        self.assertIn("dummy", message)

    def test_rejects_startup_embedding_load(self):
        """F10/R13: the startup embedding load bypasses orchestration entirely."""
        from common.tabby_config import config

        with patch.object(config.embeddings, "embedding_model_name", "bge-small"):
            message = install_module.disabled_mode_parity()
        self.assertIsNotNone(message)
        self.assertIn("embedding", message)


class _FakeContainerModule:
    """Stands in for `common.model` with observable container state."""

    def __init__(self):
        self.container = None
        self.load_calls: list = []
        self.unload_calls: list = []

    def set_container(self, model_dir="/models/fake", loaded=True, jobs=0, model=object()):
        c = SimpleNamespace(
            model_dir=Path(model_dir),
            loaded=loaded,
            active_job_ids={f"job{i}": None for i in range(jobs)},
            model=model if loaded else None,
        )
        self.container = c
        return c


class ProductionDepsTests(unittest.TestCase):
    """The CoordinatorDeps bindings against a fake common.model module."""

    def setUp(self):
        self.fakemodel = _FakeContainerModule()
        # `from common import model` reads the *common package's cached
        # attribute*, not a fresh sys.modules lookup, so the patch must target
        # that attribute (the sys.modules entry alone is not seen once the real
        # module has been imported earlier in the process — measured on kraken).
        import common

        patcher = patch.object(common, "model", self.fakemodel, create=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.deps = install_module._production_deps()

    def test_container_present_requires_real_weights(self):
        # loaded=True but model=None (the never-reset flag case) -> not present.
        c = self.fakemodel.set_container(loaded=True)
        c.model = None
        self.assertFalse(self.deps.container_present())

        c.model = object()
        self.assertTrue(self.deps.container_present())

        self.fakemodel.container = None
        self.assertFalse(self.deps.container_present())

    def test_backend_busy_reads_active_job_ids_not_loaded_flag(self):
        c = self.fakemodel.set_container(loaded=True, jobs=2)
        self.assertTrue(self.deps.backend_busy())

        c.active_job_ids.clear()
        self.assertFalse(self.deps.backend_busy())

        # Torn-down container (loaded stale True, but gone): not busy.
        self.fakemodel.container = None
        self.assertFalse(self.deps.backend_busy())

    def test_backend_busy_fails_closed_on_error(self):
        class Exploding:
            active_job_ids = property(lambda self: (_ for _ in ()).throw(RuntimeError("x")))

        self.fakemodel.container = Exploding()
        self.assertTrue(self.deps.backend_busy())

    def test_container_identity_pins_resolved_dir(self):
        self.fakemodel.set_container(model_dir="/models/qwen")
        self.assertEqual(self.deps.container_identity(), "/models/qwen")
        self.fakemodel.container = None
        self.assertIsNone(self.deps.container_identity())

    def test_unload_model_is_idempotent_when_absent(self):
        self.fakemodel.container = None
        asyncio.run(self.deps.unload_model())  # must not raise
        self.assertEqual(self.fakemodel.unload_calls, [])


class SamplerTests(unittest.TestCase):
    """TelemetrySampler: cadence, wake, error survival, clean stop."""

    def _run_sampler_briefly(self, sampler: install_module.TelemetrySampler, seconds: float):
        async def scenario():
            sampler.start()
            await asyncio.sleep(seconds)
            await sampler.stop()

        asyncio.run(scenario())

    def test_samples_are_ingested_and_stops_cleanly(self):
        ingested = []

        class FakeCoord:
            owned_identities = frozenset()

            def ingest_snapshot(self, snap):
                ingested.append(snap)

        class FakeTelemetry:
            calls = 0

            def sample(self, excluded):
                FakeTelemetry.calls += 1
                return make_snapshot(captured=FakeTelemetry.calls)

        sampler = install_module.TelemetrySampler(FakeCoord(), FakeTelemetry(), 0.01)
        self._run_sampler_briefly(sampler, 0.05)
        self.assertGreaterEqual(len(ingested), 1)
        self.assertIsNone(sampler._task)

    def test_sampler_survives_repeated_sample_errors(self):
        ingested = []

        class FakeCoord:
            owned_identities = frozenset()

            def ingest_snapshot(self, snap):
                ingested.append(snap)

        class FakeTelemetry:
            def sample(self, excluded):
                raise RuntimeError("nvml exploded")

        sampler = install_module.TelemetrySampler(FakeCoord(), FakeTelemetry(), 0.01)
        self._run_sampler_briefly(sampler, 0.05)
        # The task survived the errors and kept trying.
        self.assertGreaterEqual(sampler.sample_errors, 1)
        self.assertIsNone(sampler._task)

    def test_excluded_identities_flow_from_coordinator(self):
        seen = []

        class FakeCoord:
            owned_identities = frozenset({"boot-fake:1:10"})

            def ingest_snapshot(self, snap):
                seen.append(excluded_local[0])

        excluded_local: list = []

        class FakeTelemetry:
            def sample(self, excluded):
                excluded_local.append(excluded)
                return make_snapshot(captured=1.0)

        sampler = install_module.TelemetrySampler(FakeCoord(), FakeTelemetry(), 0.01)

        async def scenario():
            sampler.start()
            await asyncio.sleep(0.03)
            await sampler.stop()

        asyncio.run(scenario())
        self.assertEqual(excluded_local[0], frozenset({"boot-fake:1:10"}))


class LeaseContextTests(unittest.TestCase):
    """release_lease_in_finally: the C4/streaming release discipline."""

    def _coordinator_with_lease(self, cfg, deps, clock, request_id="req-1"):
        coord = _fresh_coordinator(cfg, deps, clock)

        async def scenario():
            return await coord.acquire_lease(request_id)

        result = asyncio.run(scenario())
        self.assertIs(result.outcome, AdmissionOutcome.GRANTED)
        return coord, result.lease

    def test_envelope_violation_refuses_over_envelope_request_shapes(self):
        """R06/R12 — review major F11, pinned.

        The calibrated footprint describes ONE active request of one shape. A
        multi-choice/multi-prompt request spawns more generation tasks than the
        budget charges for, so it must be refused as ``unsupported_profile``
        rather than admitted on a budget that does not describe it.
        """
        self.assertIsNone(install_module.envelope_violation(choices=1, prompts=1))
        self.assertIn("n=2", install_module.envelope_violation(choices=2, prompts=1) or "")
        self.assertIn(
            "3 prompts", install_module.envelope_violation(choices=1, prompts=3) or ""
        )

    def test_release_with_no_tasks_releases_immediately(self):
        from orch_helpers import FakeDeps

        from orchestration.config import (
            OrchestratorConfig,
            OrchestratorModelConfig,
            OrchestratorVramConfig,
        )

        cfg = OrchestratorConfig(
            enabled=True,
            device_uuid="GPU-fake",
            model=OrchestratorModelConfig(
                name="m",
                resident_delta_mib=10,
                load_peak_delta_mib=10,
                request_peak_extra_mib=1,
                calibration_id="c",
            ),
            vram=OrchestratorVramConfig(reserve_mib=1),
        )
        clock = FakeClock()
        deps = FakeDeps(clock=clock)
        coord, lease = self._coordinator_with_lease(cfg, deps, clock)

        ctx = install_module.LeaseContext(lease, coord)

        # No backend tasks (early-raise path): release scheduled immediately.
        async def wire():
            install_module.release_lease_in_finally(ctx, [])
            # release_lease_soon schedules the async release; one tick runs it.
            await asyncio.sleep(0)
            self.assertTrue(lease.released)

        asyncio.run(wire())

    def test_release_is_idempotent_after_observed(self):
        from orchestration.config import (
            OrchestratorConfig,
            OrchestratorModelConfig,
            OrchestratorVramConfig,
        )
        from orch_helpers import FakeDeps

        cfg = OrchestratorConfig(
            enabled=True,
            device_uuid="GPU-fake",
            model=OrchestratorModelConfig(
                name="m",
                resident_delta_mib=10,
                load_peak_delta_mib=10,
                request_peak_extra_mib=1,
                calibration_id="c",
            ),
            vram=OrchestratorVramConfig(reserve_mib=1),
        )
        clock = FakeClock()
        deps = FakeDeps(clock=clock)
        coord = _fresh_coordinator(cfg, deps, clock)

        async def scenario():
            return await coord.acquire_lease("req-1")

        result = asyncio.run(scenario())
        lease = result.lease
        coord.lifecycle = Lifecycle.READY

        ctx = install_module.LeaseContext(lease, coord)

        async def fake_task():
            await asyncio.sleep(0.02)
            return "done"

        # Simulate the wrapper finally with a live task, from inside a loop.
        async def wire():
            real_task = asyncio.get_running_loop().create_task(fake_task())
            install_module.release_lease_in_finally(ctx, [real_task])
            # Lease must still be held while the task runs.
            self.assertFalse(lease.released)
            await real_task
            # The observer task needs a loop tick to finish its release.
            for _ in range(5):
                await asyncio.sleep(0)
            self.assertTrue(lease.released)

        asyncio.run(wire())
        # Second call must be a no-op, not an error.
        install_module.release_lease_in_finally(ctx, [])
        self.assertTrue(lease.released)

    def test_release_with_none_context_is_noop(self):
        # Disabled mode (no lease at all) must never raise from a finally.
        install_module.release_lease_in_finally(None, [])


class ReaderPinTests(unittest.TestCase):
    """F10/R12: the production reader pin on tokenization/metadata reads.

    READER leases get a real production caller here; the contract is the
    coordinator's: a reader pin defers teardown but never refreshes the
    inference TTL, and a read is refused rather than cold-loading when the
    model is not READY.
    """

    def _coordinator(self, cfg, deps, clock, ready):
        coord = _make_coordinator(cfg, deps, clock)
        coord.ingest_snapshot(make_snapshot(captured=clock(), free=10_000 * orch_helpers.MIB))
        if ready:
            deps._present = True
            coord.lifecycle = Lifecycle.READY
        return coord

    def _runtime_ready(self, coord):
        install_module.runtime.orchestrator = coord
        self.addCleanup(setattr, install_module.runtime, "orchestrator", None)

    def test_reader_pin_granted_when_ready_and_never_refreshes_ttl(self):
        """The core R08 property: a reader pin does not reset the idle timer."""
        # The dev venv lacks fastapi (orchestration.api imports it); stub the
        # one symbol the pin path needs, exactly like the api.py docstring
        # convention for a minimal test environment.
        import types
        import unittest.mock

        try:
            from orchestration.api import admission_http_exception  # noqa: F401
        except ModuleNotFoundError:
            api_stub = types.ModuleType("orchestration.api")
            api_stub.admission_http_exception = lambda result: None
            unittest.mock.patch.dict(
                "sys.modules", {"orchestration.api": api_stub}
            ).start()
            self.addCleanup(unittest.mock.patch.dict("sys.modules").stop)

        from orch_helpers import FakeDeps

        from orchestration.config import (
            OrchestratorConfig,
            OrchestratorModelConfig,
            OrchestratorVramConfig,
        )

        cfg = OrchestratorConfig(
            enabled=True,
            device_uuid="GPU-fake",
            model=OrchestratorModelConfig(
                name="m",
                resident_delta_mib=10,
                load_peak_delta_mib=10,
                request_peak_extra_mib=1,
                calibration_id="c",
            ),
            vram=OrchestratorVramConfig(reserve_mib=1),
        )
        clock = FakeClock()
        deps = FakeDeps(clock=clock)
        coord = self._coordinator(cfg, deps, clock, ready=True)
        self._runtime_ready(coord)
        # The model has been idle since READY.
        idle_start = clock()
        coord._idle_since = idle_start

        async def scenario():
            ctx, _coord = await install_module.acquire_reader_pin("read:r1")
            self.assertIsNotNone(ctx)
            clock.advance(120.0)  # reader holds well past a typical TTL
            status = await coord.status()
            self.assertEqual(status["requests"]["reader_pins"], 1)
            # The idle clock must be UNCHANGED by the reader pin (R08/T10).
            self.assertEqual(coord._idle_since, idle_start)
            install_module.release_reader_pin(ctx)
            for _ in range(3):
                await asyncio.sleep(0)
            self.assertTrue(ctx.lease.released)

        asyncio.run(scenario())

    def test_reader_pin_never_cold_loads(self):
        """R12: a read is refused with model_transition, not served by a load."""
        # Same fastapi-less-environment stub as the TTL test above.
        import types
        import unittest.mock

        try:
            from orchestration.api import admission_http_exception  # noqa: F401
        except ModuleNotFoundError:
            api_stub = types.ModuleType("orchestration.api")
            api_stub.admission_http_exception = lambda result: None
            unittest.mock.patch.dict(
                "sys.modules", {"orchestration.api": api_stub}
            ).start()
            self.addCleanup(unittest.mock.patch.dict("sys.modules").stop)

        from orch_helpers import FakeDeps

        from orchestration.config import (
            OrchestratorConfig,
            OrchestratorModelConfig,
            OrchestratorVramConfig,
        )

        cfg = OrchestratorConfig(
            enabled=True,
            device_uuid="GPU-fake",
            model=OrchestratorModelConfig(
                name="m",
                resident_delta_mib=10,
                load_peak_delta_mib=10,
                request_peak_extra_mib=1,
                calibration_id="c",
            ),
            vram=OrchestratorVramConfig(reserve_mib=1),
        )
        clock = FakeClock()
        deps = FakeDeps(clock=clock)
        coord = self._coordinator(cfg, deps, clock, ready=False)  # UNLOADED
        self._runtime_ready(coord)
        coord.ingest_snapshot(make_snapshot(captured=clock(), free=10_000 * orch_helpers.MIB))

        async def scenario():
            with self.assertRaises(install_module.LeaseDenied) as caught:
                await install_module.acquire_reader_pin("read:r2")
            self.assertIn("model_transition", caught.exception.reason)

        asyncio.run(scenario())
        self.assertEqual(deps.load_count, 0, "a read must never trigger a load")

    def test_reader_pin_is_counted_and_blocks_unload(self):
        """R08: an outstanding reader pin defers teardown (counts as model use)."""
        # Same fastapi-less-environment stub as the TTL test above.
        import types
        import unittest.mock

        try:
            from orchestration.api import admission_http_exception  # noqa: F401
        except ModuleNotFoundError:
            api_stub = types.ModuleType("orchestration.api")
            api_stub.admission_http_exception = lambda result: None
            unittest.mock.patch.dict(
                "sys.modules", {"orchestration.api": api_stub}
            ).start()
            self.addCleanup(unittest.mock.patch.dict("sys.modules").stop)

        from orch_helpers import FakeDeps

        from orchestration.config import (
            OrchestratorConfig,
            OrchestratorModelConfig,
            OrchestratorVramConfig,
        )

        cfg = OrchestratorConfig(
            enabled=True,
            device_uuid="GPU-fake",
            model=OrchestratorModelConfig(
                name="m",
                resident_delta_mib=10,
                load_peak_delta_mib=10,
                request_peak_extra_mib=1,
                calibration_id="c",
            ),
            vram=OrchestratorVramConfig(reserve_mib=1),
        )
        clock = FakeClock()
        deps = FakeDeps(clock=clock)
        coord = self._coordinator(cfg, deps, clock, ready=True)
        self._runtime_ready(coord)
        # A completed inference started the idle clock (reader pins never do).
        coord._idle_since = clock()

        async def scenario():
            ctx, _coord = await install_module.acquire_reader_pin("read:r3")
            clock.advance(coord.cfg.idle_unload.seconds + 1.0)
            action = await coord.tick()
            self.assertIsNone(action, "an outstanding reader pin must defer unload")
            install_module.release_reader_pin(ctx)
            for _ in range(3):
                await asyncio.sleep(0)
            # Pin released: the next tick may unload.
            action = await coord.tick()
            self.assertEqual(action, "unload_idle")

        asyncio.run(scenario())

    def test_reader_pin_disabled_mode_is_noop(self):
        # A sibling test module (test_api.RouteTests) publishes a coordinator on
        # the runtime holder and does not always clear it; this test's contract
        # is the DISABLED path, so clear the holder explicitly.
        previous = install_module.runtime.orchestrator
        install_module.runtime.orchestrator = None
        self.addCleanup(setattr, install_module.runtime, "orchestrator", previous)

        async def scenario():
            ctx, coordinator_obj = await install_module.acquire_reader_pin("read:r4")
            self.assertIsNone(ctx)
            self.assertIsNone(coordinator_obj)
            # Release with None must not raise.
            install_module.release_reader_pin(None)

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()

class StatusAdmissionAgreementTests(unittest.TestCase):
    """T06 invariant: status.can_admit_now must equal what acquire_lease decides.

    Found by the independent M1 review (blocker 1): with the model UNLOADED and
    the cold quiet window incomplete, status published can_admit_now=true while
    acquire_lease denied (gpu_not_quiet); with lifecycle=DRAINING and no work
    outstanding it published true while acquire denied. Both are agreement
    violations — status and admission must answer the same question.
    """

    def _coord(self):
        from orch_helpers import FakeClock, FakeDeps, make_snapshot, MIB
        from orchestration.config import (
            OrchestratorConfig,
            OrchestratorModelConfig,
            OrchestratorVramConfig,
        )
        from orchestration.lifecycle import LifecycleCoordinator

        cfg = OrchestratorConfig(
            enabled=True,
            device_uuid="GPU-fake",
            model=OrchestratorModelConfig(
                name="m",
                resident_delta_mib=10,
                load_peak_delta_mib=10,
                request_peak_extra_mib=1,
                calibration_id="c",
            ),
            vram=OrchestratorVramConfig(reserve_mib=1),
        )
        clock = FakeClock()
        deps = FakeDeps(clock=clock)
        coord = LifecycleCoordinator(
            cfg, deps.as_deps(), SimpleNamespace(boot_id="b", capabilities={}, init_error=None), clock=clock
        )
        coord.ingest_snapshot(make_snapshot(captured=clock(), free=10_000 * MIB))
        return coord

    def test_status_agrees_with_admission_unloaded_and_draining(self):
        from orchestration.lifecycle import AdmissionOutcome, Lifecycle

        async def scenario():
            coord = self._coord()
            results = []
            # UNLOADED, quiet window incomplete.
            st = await coord.status()
            r = await coord.acquire_lease("r1")
            results.append(("unloaded", st["admission"]["can_admit_now"], r.outcome))
            # DRAINING with no outstanding work.
            self._ = None
            deps_present = True
            coord.lifecycle = Lifecycle.DRAINING
            coord._drain_requested = True
            st = await coord.status()
            r = await coord.acquire_lease("r2")
            results.append(("draining", st["admission"]["can_admit_now"], r.outcome))
            return results

        for label, can_admit, outcome in asyncio.run(scenario()):
            self.assertEqual(
                can_admit,
                outcome is AdmissionOutcome.GRANTED,
                f"{label}: status.can_admit_now={can_admit} but acquire_lease -> {outcome.value}",
            )
