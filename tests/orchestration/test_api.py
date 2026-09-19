"""Tests for the orchestrator control routes (SPEC R10, R11, R12).

Mocked tier, like the rest of ``tests/orchestration/``: a real coordinator over fake
deps, plus a real FastAPI app whose coordinator is published on
``orchestration.install.runtime`` — the same holder the production wiring and the
request path use. The app is exercised through ``httpx.ASGITransport`` — the same
way upstream's ``tests/test_context_length_errors.py`` drives its error handler —
so the routes are tested as HTTP rather than by calling the coroutine directly.

What these tests are for: the contract *around* the coordinator (status mapping,
Retry-After policy, route auth, error shape). The coordinator's own logic is covered
by ``test_lifecycle.py``; a test here that re-asserted it would add no evidence.
"""

from __future__ import annotations

import asyncio
import sys
import unittest
import unittest.mock
from pathlib import Path

from fastapi import FastAPI, HTTPException
from httpx import ASGITransport, AsyncClient

from orchestration import api as orch_api
from orchestration import install as install_module
from orchestration.lifecycle import AdmissionOutcome, AdmissionResult, LifecycleCoordinator
from orchestration.policy import Reason
from orchestration.telemetry import ActivityStatus

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import orch_helpers  # noqa: E402
from orch_helpers import MIB, FakeClock, FakeDeps, FakeTelemetry, make_snapshot  # noqa: E402
from orchestration.config import (  # noqa: E402
    OrchestratorConfig,
    OrchestratorModelConfig,
    OrchestratorVramConfig,
)


def calibrated_config() -> OrchestratorConfig:
    """A local copy of the ``calibrated_config`` fixture.

    Built here rather than importing the fixture, because ``orch_helpers`` registers it
    with ``@pytest.fixture`` and calling a fixture directly is a pytest error. This test
    class is a plain ``unittest.TestCase`` (matching the rest of the suite's convention
    of not depending on pytest-asyncio), so it must construct the config itself.
    """

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


def build_coordinator(clock=None, quiet_seconds=1.0):
    """Mirror of ``test_lifecycle.build``: a coordinator with a satisfied quiet window."""

    clock = clock or FakeClock()
    cfg = calibrated_config()
    cfg.cold_load.quiet_seconds = quiet_seconds
    deps = FakeDeps(clock=clock)
    coord = LifecycleCoordinator(
        cfg, deps.as_deps(), FakeTelemetry(), clock=clock, log=lambda _m: None
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
    coord.ingest_snapshot(
        make_snapshot(
            captured=clock.now, utilization=0, external=[], activity_status=ActivityStatus.SPARSE
        ),
        clock.now,
    )
    return coord, deps, clock


def build_app(coordinator=None, *, with_auth=True):
    """A minimal app carrying the orchestration router.

    Auth is dependency-overridden rather than satisfied with a real key: the point is
    to prove the *route* is admin-gated, not to re-test upstream's key verification,
    which upstream already covers. The coordinator is published on
    ``orchestration.install.runtime`` — the same holder the production wiring
    uses — so the routes exercise their real lookup path.
    """

    app = FastAPI()
    app.include_router(orch_api.router)
    install_module.runtime.orchestrator = coordinator
    if not with_auth:
        from common.auth import check_admin_key

        app.dependency_overrides[check_admin_key] = lambda: "test-admin"
    return app


def clear_runtime_coordinator():
    install_module.runtime.orchestrator = None


class StatusMappingTests(unittest.TestCase):
    """SPEC R12's HTTP/code table, asserted directly."""

    def test_status_codes_match_the_spec_table(self):
        self.assertEqual(orch_api.status_for(Reason.UNSUPPORTED_PROFILE), 400)
        self.assertEqual(orch_api.status_for(Reason.MODEL_NOT_CONFIGURED), 404)
        self.assertEqual(orch_api.status_for(Reason.ADMISSION_QUEUE_FULL), 429)
        for reason in (
            Reason.EXTERNAL_GPU_BUSY,
            Reason.GPU_NOT_QUIET,
            Reason.INSUFFICIENT_VRAM,
            Reason.TELEMETRY_UNAVAILABLE,
            Reason.ORCHESTRATOR_PAUSED,
            Reason.ORCHESTRATOR_FAULT,
            Reason.MODEL_TRANSITION,
            Reason.REQUEST_CAPACITY,
            Reason.ADMISSION_TIMEOUT,
        ):
            self.assertEqual(orch_api.status_for(reason), 503, reason)

    def test_retry_after_only_for_temporary_conditions(self):
        # A pause or a fault is not "retry in 5s and it will work"; advertising
        # Retry-After for them invites the retry behaviour R14 exists to bound.
        self.assertTrue(orch_api.is_temporary(Reason.EXTERNAL_GPU_BUSY))
        self.assertTrue(orch_api.is_temporary(Reason.GPU_NOT_QUIET))
        self.assertFalse(orch_api.is_temporary(Reason.ORCHESTRATOR_PAUSED))
        self.assertFalse(orch_api.is_temporary(Reason.ORCHESTRATOR_FAULT))
        self.assertFalse(orch_api.is_temporary(Reason.UNSUPPORTED_PROFILE))

    def test_admission_http_exception_carries_code_and_retry_after(self):
        result = AdmissionResult(
            outcome=AdmissionOutcome.DENIED,
            reason=Reason.INSUFFICIENT_VRAM,
            detail={"message": "need 9000 MiB, have 100"},
        )
        exc = orch_api.admission_http_exception(result)
        self.assertEqual(exc.status_code, 503)
        self.assertIn(Reason.INSUFFICIENT_VRAM.value, exc.detail)
        self.assertEqual(exc.headers, {"Retry-After": "5"})

    def test_admission_http_exception_omits_retry_after_when_paused(self):
        result = AdmissionResult(
            outcome=AdmissionOutcome.DENIED,
            reason=Reason.ORCHESTRATOR_PAUSED,
            detail={"message": "paused"},
        )
        exc = orch_api.admission_http_exception(result)
        self.assertEqual(exc.status_code, 503)
        self.assertIsNone(exc.headers)

    def test_error_body_matches_the_upstream_error_shape(self):
        body = orch_api.error_content("insufficient_vram", "msg", retryable=True)
        self.assertEqual(
            body["error"],
            {
                "message": "msg",
                "type": "orchestrator_error",
                "param": None,
                "code": "insufficient_vram",
                "retry_after_seconds": 5,
            },
        )


class RouteTests(unittest.IsolatedAsyncioTestCase):
    """The three R12 routes over a real coordinator."""

    def setUp(self):
        self.coord, self.deps, self.clock = build_coordinator()

    async def _get(self, app, method, path):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            return await client.request(method, path)

    async def test_status_route_returns_a_bounded_snapshot(self):
        app = build_app(self.coord, with_auth=False)
        response = await self._get(app, "GET", "/v1/orchestrator/status")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        # R11: orthogonal authorities, not a combined state machine.
        self.assertIn(
            payload["lifecycle"],
            {"UNLOADED", "LOADING", "READY", "DRAINING", "UNLOADING", "FAULT"},
        )
        self.assertIn(
            payload["policy"], {"UNKNOWN", "CLEAR", "CANDIDATE", "BUSY", "MANUAL_PAUSE"}
        )
        self.assertIsInstance(payload["paused"], bool)
        self.assertIn("can_admit_now", payload["admission"])
        self.assertIn("can_begin_cold_load", payload["admission"])
        # R10: pause is process-local and status must say so.
        self.assertTrue(payload["pause_is_process_local"])

    async def test_status_route_works_when_unloaded(self):
        # R12 requires status to be available even when paused/faulted/unloaded.
        app = build_app(self.coord, with_auth=False)
        response = await self._get(app, "GET", "/v1/orchestrator/status")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["lifecycle"], "UNLOADED")

    async def test_pause_is_202_idempotent_and_does_not_load(self):
        app = build_app(self.coord, with_auth=False)
        first = await self._get(app, "POST", "/v1/orchestrator/pause")
        second = await self._get(app, "POST", "/v1/orchestrator/pause")

        self.assertEqual(first.status_code, 202)
        self.assertEqual(second.status_code, 202, "pause must be idempotent (R10)")
        self.assertTrue(first.json()["paused"])
        self.assertTrue(second.json()["paused"])
        self.assertNotIn("load:start", self.deps.calls, "pause must not load (R10)")

    async def test_resume_is_200_clears_only_the_manual_veto_and_does_not_load(self):
        app = build_app(self.coord, with_auth=False)
        await self._get(app, "POST", "/v1/orchestrator/pause")
        response = await self._get(app, "POST", "/v1/orchestrator/resume")

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["paused"])
        # R10: resume must not prewarm without demand.
        self.assertNotIn("load:start", self.deps.calls, "resume must not prewarm (R10)")

    async def test_routes_are_admin_gated(self):
        # R12: no inference-scoped or unauthenticated variant of these routes exists.
        app = build_app(self.coord)  # real check_admin_key
        for method, path in (
            ("GET", "/v1/orchestrator/status"),
            ("POST", "/v1/orchestrator/pause"),
            ("POST", "/v1/orchestrator/resume"),
        ):
            response = await self._get(app, method, path)
            self.assertEqual(response.status_code, 401, f"{method} {path} must require auth")

    async def test_missing_coordinator_is_an_explicit_503_not_a_traceback(self):
        app = build_app(None, with_auth=False)
        response = await self._get(app, "GET", "/v1/orchestrator/status")
        self.assertEqual(response.status_code, 503)
        self.assertIn("orchestrator_fault", response.json()["detail"])


class UnsupportedSurfaceTests(unittest.TestCase):
    """F10/R12/T13: ungoverned mutation surfaces are rejected in enabled mode.

    These drive the REAL ``endpoints.core.router`` handlers through a real
    FastAPI app (httpx ASGITransport), so the rejection is proven at the HTTP
    boundary, not by calling a helper. Requires fastapi — runs in the staged
    venv on kraken, skipped in the minimal dev venv like the rest of the
    router-level coverage.
    """

    @classmethod
    def setUpClass(cls):
        try:
            import fastapi  # noqa: F401
            import httpx  # noqa: F401
        except ModuleNotFoundError:
            raise unittest.SkipTest("fastapi/httpx not available in the dev venv")
        # The core router reads `config.orchestrator.enabled`; the config
        # section must be registered in this test process before the attribute
        # exists (the production server does this in main.entrypoint). The
        # module-level `config` singleton was instantiated before registration,
        # so load() revalidates and copies the new field onto the instance
        # (measured on kraken: field registration alone does not extend the
        # already-built instance).
        from orchestration import install

        install.install_config_section()
        from common.tabby_config import config

        config.load()
        cls._config_patch = unittest.mock.patch.object(config.orchestrator, "enabled", True)
        cls._config_patch.start()
        cls.addClassCleanup(cls._config_patch.stop)

    def _post(self, path, json_body=None):
        import httpx
        from fastapi import FastAPI

        from endpoints.core import router as core_router

        app = FastAPI()
        app.include_router(core_router.router)

        async def call():
            # Auth is bypassed by overriding the dependency: the point is the
            # surface rejection, not upstream's key checks.
            from common.auth import check_admin_key, check_api_key

            app.dependency_overrides[check_admin_key] = lambda: "t"
            app.dependency_overrides[check_api_key] = lambda: "t"
            # The container dependency would 503 before the handler runs on
            # some routes; the surface rejection is placed FIRST in each
            # handler, so override it too.
            from common.model import check_model_container, check_embeddings_container

            async def _pass():
                return None

            app.dependency_overrides[check_model_container] = _pass
            app.dependency_overrides[check_embeddings_container] = _pass
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://t"
            ) as client:
                return await client.post(path, json=json_body or {})

        return asyncio.run(call())

    def test_lora_load_is_rejected(self):
        response = self._post("/v1/lora/load", {"loras": [{"name": "x"}]})
        self.assertEqual(response.status_code, 400)
        self.assertIn("unsupported_profile", response.json()["detail"])
        self.assertIn("lora", response.json()["detail"])

    def test_lora_unload_is_rejected(self):
        response = self._post("/v1/lora/unload")
        self.assertEqual(response.status_code, 400)
        self.assertIn("unsupported_profile", response.json()["detail"])

    def test_template_switch_is_rejected(self):
        response = self._post("/v1/template/switch", {"prompt_template_name": "x"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("unsupported_profile", response.json()["detail"])

    def test_template_unload_is_rejected(self):
        response = self._post("/v1/template/unload")
        self.assertEqual(response.status_code, 400)
        self.assertIn("unsupported_profile", response.json()["detail"])

    def test_sampler_override_switch_is_rejected(self):
        response = self._post("/v1/sampling/override/switch", {"preset": "x"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("unsupported_profile", response.json()["detail"])

    def test_sampler_override_unload_is_rejected(self):
        response = self._post("/v1/sampling/override/unload")
        self.assertEqual(response.status_code, 400)
        self.assertIn("unsupported_profile", response.json()["detail"])

    def test_embedding_load_is_rejected(self):
        response = self._post("/v1/model/embedding/load", {"embedding_model_name": "x"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("unsupported_profile", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
