"""FastAPI routes and response schemas for orchestrator control (SPEC R10–R12).

This module is deliberately thin. It owns no lifecycle state: every route reads or
mutates the :class:`~orchestration.lifecycle.LifecycleCoordinator`, which is the sole
authority. Two consequences follow, and both are load-bearing:

* ``GET /v1/orchestrator/status`` publishes whatever the coordinator reports and never
  recomputes a verdict. R11 requires status and admission to agree for the same
  snapshot; recomputing here would create a second evaluator that could disagree.
* the routes are registered by ``install.py``, not by this module, so that a
  disabled-mode server exposes no orchestration surface at all (R12/R13).

Auth follows upstream exactly: admin scope via ``check_admin_key``. The spec is
explicit that an inference-only client must *not* be granted this scope and must read
availability from inference error codes instead, so there is no inference-scoped
variant of these routes and no unauthenticated status route.

Pause/resume are process-local (R10) and the status payload says so, because an
operator who assumes otherwise would think a restart preserved their pause.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from common.auth import check_admin_key
from common.logger import xlogger
from orchestration.errors import (
    RETRY_AFTER_SECONDS,
    OrchestrationHTTPException,
    error_content,
    orchestration_error,
)
from orchestration.policy import Reason

router = APIRouter()

#: Status codes for the stable orchestration reason codes (SPEC R12). A code that is
#: missing here is a programming error, not a 500 waiting to happen: ``_status_for``
#: returns 503 for anything unknown so an unmapped reason degrades to the generic
#: "pre-admission unavailable" bucket rather than leaking a wrong status.
_RETRY_AFTER_SECONDS = RETRY_AFTER_SECONDS


def status_for(reason: Reason) -> int:
    """Map a reason code to the HTTP status the spec's table assigns it."""

    if reason is Reason.ADMISSION_TIMEOUT:
        return 503
    if reason is Reason.ADMISSION_QUEUE_FULL:
        return 429
    if reason is Reason.UNSUPPORTED_PROFILE:
        return 400
    if reason is Reason.MODEL_NOT_CONFIGURED:
        return 404
    # Everything else is a pre-admission availability failure.
    return 503


def is_temporary(reason: Reason) -> bool:
    """Whether a modest advisory Retry-After belongs on this response.

    Deliberately conservative: pause and fault are not "retry in 5s and it will work"
    conditions, and advertising Retry-After for them would invite exactly the retry
    storm R14's validation exists to prevent.
    """

    return reason in (
        Reason.EXTERNAL_GPU_BUSY,
        Reason.GPU_NOT_QUIET,
        Reason.INSUFFICIENT_VRAM,
        Reason.TELEMETRY_UNAVAILABLE,
        Reason.MODEL_TRANSITION,
        Reason.REQUEST_CAPACITY,
    )


def admission_http_exception(result, *, message: Optional[str] = None) -> HTTPException:
    """Translate an ``AdmissionResult`` into the correct HTTPException.

    Shared by the inference boundary so that every denial path produces one shape.

    The stable reason code is always included in the message. R12 makes that
    load-bearing: inference-only clients are forbidden the admin status route, so the
    error body is their *only* channel for learning why they were refused, and
    "insufficient_vram" vs "external_gpu_busy" determines whether retrying is
    reasonable at all. Omitting the code would leave them with prose.

    The returned exception is an :class:`~orchestration.errors.OrchestrationHTTPException`
    so the body is R12's structured shape (``{"error": {..., "code": ...}}``) rather
    than a bare ``{"detail": ...}``; it still *is* an ``HTTPException``, so every
    existing raise/except site is unchanged.
    """

    reason = result.reason if result.reason is not None else Reason.OK
    code = reason.value
    detail = message or result.detail.get("message") or ""
    if not detail:
        # R05: a denied memory budget must report its shortfall on the inference
        # path too, not just in status. The verdict's blockers carry the measured
        # "free X, need Y" text from evaluate_capacity.
        blockers = result.detail.get("blockers") or result.detail.get("cold_blockers") or []
        relevant = [b for b in blockers if "insufficient" in b or "capacity" in b]
        if relevant:
            detail = "; ".join(relevant)
    if code not in detail:
        detail = f"{code}: {detail}" if detail else code
    return OrchestrationHTTPException(
        status_for(reason), code, detail, retryable=is_temporary(reason)
    )


# --------------------------------------------------------------------------- #
# Response schemas
# --------------------------------------------------------------------------- #


class OrchestratorError(BaseModel):
    """The error object carried by every orchestration failure body."""

    message: str
    type: str = "orchestrator_error"
    param: Optional[str] = None
    code: str
    retry_after_seconds: Optional[int] = None


class OrchestratorErrorResponse(BaseModel):
    error: OrchestratorError


class StatusResponse(BaseModel):
    """The coordinator's bounded snapshot.

    Intentionally permissive: the coordinator owns this payload and R11 will grow it.
    Modelling every nested field here would mean duplicating the coordinator's schema
    in two places, and a drift between them would silently drop diagnostics at the
    exact moment an operator needs them.
    """

    enabled: bool
    lifecycle: str
    policy: str
    paused: bool
    admission: dict = Field(default_factory=dict)
    capacity: dict = Field(default_factory=dict)
    requests: dict = Field(default_factory=dict)
    gpu: dict = Field(default_factory=dict)
    telemetry: dict = Field(default_factory=dict)

    model_config = {"extra": "allow"}


def _coordinator_or_503(request: Request):
    """Fetch the installed coordinator, or fail with a clear reason.

    The coordinator lives on ``orchestration.install.runtime`` — the holder the
    request path already uses — and NOT on ``app.state``: publishing it in two
    places would let them disagree. A missing coordinator on an enabled-mode
    route is a wiring fault, not a transient condition, so it is surfaced as an
    explicit 503 code rather than an AttributeError traceback.
    """

    from orchestration.install import runtime

    coordinator = runtime.orchestrator
    if coordinator is None:
        raise orchestration_error(
            status_for(Reason.ORCHESTRATOR_FAULT),
            Reason.ORCHESTRATOR_FAULT.value,
            "Orchestrator is not installed in this process",
        )
    return coordinator


# --------------------------------------------------------------------------- #
# Routes (SPEC section 8 table)
# --------------------------------------------------------------------------- #


@router.get(
    "/v1/orchestrator/status",
    dependencies=[Depends(check_admin_key)],
    response_model=StatusResponse,
)
async def get_status(request: Request) -> StatusResponse:
    """Immediate current snapshot; available even when paused/faulted/unloaded."""

    coordinator = _coordinator_or_503(request)
    payload = await coordinator.status()
    return StatusResponse(**payload)


@router.post(
    "/v1/orchestrator/pause",
    dependencies=[Depends(check_admin_key)],
    responses={202: {"model": StatusResponse}},
)
async def pause(request: Request):
    """Idempotent pause. 202 plus snapshot; draining may continue."""

    coordinator = _coordinator_or_503(request)
    await coordinator.pause()
    payload = await coordinator.status()
    xlogger.info("Orchestrator paused", {"lifecycle": payload.get("lifecycle")})
    return JSONResponse(status_code=202, content=StatusResponse(**payload).model_dump())


@router.post(
    "/v1/orchestrator/resume",
    dependencies=[Depends(check_admin_key)],
    response_model=StatusResponse,
)
async def resume(request: Request) -> StatusResponse:
    """Idempotent resume. 200 plus snapshot; does not load."""

    coordinator = _coordinator_or_503(request)
    await coordinator.resume()
    payload = await coordinator.status()
    xlogger.info("Orchestrator resumed", {"lifecycle": payload.get("lifecycle")})
    return StatusResponse(**payload)


def setup():
    """Mirror the upstream router convention (``endpoints/*/router.setup``)."""

    return router
