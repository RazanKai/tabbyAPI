import asyncio
from asyncio import CancelledError, InvalidStateError

from fastapi import APIRouter, Depends, HTTPException, Request
from sse_starlette import EventSourceResponse

from common import model
from common.auth import check_api_key
from common.debug_requests import (
    log_chat_completion_request,
    write_chat_completion_prompt_log,
)
from common.model import check_embeddings_container, check_model_container
from common.networking import (
    get_sse_ping_interval,
    handle_request_error,
    request_tag,
    DisconnectHandler,
    run_with_request_disconnect,
)
from common.tabby_config import config
from common.logger import xlogger
from endpoints.OAI.types.completion import CompletionRequest, CompletionResponse
from endpoints.OAI.types.chat_completion import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from endpoints.OAI.types.embedding import EmbeddingsRequest, EmbeddingsResponse
from endpoints.OAI.utils.common_ import load_inline_model
from endpoints.OAI.utils.chat_completion import (
    apply_chat_template,
    generate_chat_completion,
    stream_generate_chat_completion,
)
from endpoints.OAI.utils.completion import (
    generate_completion,
    stream_generate_completion,
)
from endpoints.OAI.utils.embeddings import get_embeddings


api_name = "OAI"
router = APIRouter()
urls = {
    "Completions": "http://{host}:{port}/v1/completions",
    "Chat completions": "http://{host}:{port}/v1/chat/completions",
}

# Block when model is still loading while second inline load request comes in
load_lock: asyncio.Lock = asyncio.Lock()


def _orchestrator_enabled() -> bool:
    """Whether orchestration is installed in this process (enabled mode)."""

    from orchestration import install

    return install.runtime.orchestrator is not None


def setup():
    return router


# Completions endpoint
@router.post(
    "/v1/completions",
    dependencies=[Depends(check_api_key)],
)
async def completion_request(request: Request, data: CompletionRequest) -> CompletionResponse:
    """
    Generates a completion from a prompt.

    If stream = true, this returns an SSE stream.
    """

    raw_json = await request.json()
    xlogger.debug("[ENDPOINT] /v1/completions", {"raw": raw_json})

    # Orchestrated admission (enabled mode only): the lease is acquired before
    # the first live-container use and before the router load lock, so a
    # request waiting for admission holds no upstream lock (R06/R07, map §1).
    # In disabled mode this is a no-op returning None. A denial is re-raised
    # as the R12-shaped HTTPException directly, so a pre-admission failure is
    # a real non-2xx response rather than a 500 (R09/R12).
    from orchestration.install import (
        LeaseDenied,
        acquire_lease_now,
        envelope_violation,
        release_lease_in_finally,
    )

    lease_ctx = None
    if _orchestrator_enabled():
        # R06/R12: refuse a request shape the calibration does not describe,
        # before any load or lease — `n>1`/multi-prompt would spawn more
        # generation tasks while the budget charges one request peak.
        shape_problem = envelope_violation(
            choices=int(getattr(data, "n", 1) or 1),
            prompts=len(data.prompt) if isinstance(data.prompt, list) else 1,
        )
        if shape_problem is not None:
            from orchestration.errors import orchestration_error
            from orchestration.policy import Reason

            raise orchestration_error(
                400, Reason.UNSUPPORTED_PROFILE.value, shape_problem
            )
        try:
            lease_ctx = await acquire_lease_now(request.state.id)
        except LeaseDenied as denied:
            raise denied.exc from None

    reached_wrapper = False  # the wrapper's finally owns the lease once entered
    try:
        async with load_lock:
            if _orchestrator_enabled():
                # R01/R12: the coordinator is the only load authority in enabled
                # mode. A nonempty model field must be exactly the configured
                # canonical id — anything else is model_not_configured — and no
                # inline load runs at all: the lease acquisition already performed
                # the narrowly preauthorized configured-profile load (or denied).
                # This removes the admin-key arbitrary-path bypass entirely.
                if data.model and data.model != config.orchestrator.model.name:
                    from orchestration.errors import orchestration_error
                    from orchestration.policy import Reason

                    raise orchestration_error(
                        404,
                        Reason.MODEL_NOT_CONFIGURED.value,
                        f"{data.model!r} is not the configured "
                        f"model ({config.orchestrator.model.name!r})",
                    )
                await check_model_container()
            elif data.model:
                await load_inline_model(data.model, request)
            else:
                await check_model_container()
            model_path = model.container.model_dir

        # Prepare raw prompt (will be str or list[str])
        prompt = data.prompt

        # Set an empty JSON schema if the request wants a JSON response
        if data.response_format.type == "json":
            data.json_schema = {"type": "object"}

        # Also accept specific schema from response_format
        if data.response_format.type == "json_schema":
            data.json_schema = data.response_format.json_schema

        disconnect_handler = DisconnectHandler(request, f"{request_tag(request)} completions")
        await disconnect_handler.poll()

        if data.stream and not config.developer.disable_request_streaming:
            model.check_context_length(prompt, data)
            response = EventSourceResponse(
                stream_generate_completion(
                    prompt,
                    data,
                    request,
                    model_path,
                    disconnect_handler,
                    lease_ctx=lease_ctx,
                ),
                ping=get_sse_ping_interval(),
            )
            reached_wrapper = True
            return response
        else:
            response = await generate_completion(
                prompt,
                data,
                request,
                model_path,
                disconnect_handler,
                lease_ctx=lease_ctx,
            )
            reached_wrapper = True
            return response

    except (CancelledError, InvalidStateError) as ex:
        raise HTTPException(422, "/v1/completions request cancelled by user.") from ex

    finally:
        # C4: the upstream handlers had no finally; every early raise between
        # lease acquisition and wrapper entry (inline load failure, container
        # 503, context length, cancellation) must release. Once a wrapper was
        # entered, ITS finally owns the lease — for streaming it runs later
        # than this one (the response object is returned, not consumed), so
        # releasing here would fire while generation is still unwinding.
        if not reached_wrapper:
            release_lease_in_finally(lease_ctx, [])


# Chat completions endpoint
@router.post(
    "/v1/chat/completions",
    dependencies=[Depends(check_api_key), Depends(log_chat_completion_request)],
)
async def chat_completion_request(
    request: Request, data: ChatCompletionRequest
) -> ChatCompletionResponse:
    """
    Generates a chat completion from a prompt.

    If stream = true, this returns an SSE stream.
    """

    raw_json = await request.json()
    xlogger.debug("[ENDPOINT] /v1/chat/completions", {"raw": raw_json})

    # Orchestrated admission (enabled mode only) — see the completions handler
    # for the rationale and the LeaseDenied handling.
    from orchestration.install import (
        LeaseDenied,
        acquire_lease_now,
        envelope_violation,
        release_lease_in_finally,
    )

    lease_ctx = None
    if _orchestrator_enabled():
        # R06/R12: refuse a request shape the calibration does not describe (see
        # the completions handler). `n>1` spawns multiple generation tasks while
        # the budget charges a single request peak.
        shape_problem = envelope_violation(
            choices=int(getattr(data, "n", 1) or 1),
            prompts=1,
        )
        if shape_problem is not None:
            from orchestration.errors import orchestration_error
            from orchestration.policy import Reason

            raise orchestration_error(400, Reason.UNSUPPORTED_PROFILE.value, shape_problem)
        try:
            lease_ctx = await acquire_lease_now(request.state.id)
        except LeaseDenied as denied:
            raise denied.exc from None

    reached_wrapper = False  # the wrapper's finally owns the lease once entered
    try:
        async with load_lock:
            if _orchestrator_enabled():
                # R01/R12: the coordinator is the only load authority in enabled
                # mode. A nonempty model field must be exactly the configured
                # canonical id — anything else is model_not_configured — and no
                # inline load runs at all: the lease acquisition already performed
                # the narrowly preauthorized configured-profile load (or denied).
                # This removes the admin-key arbitrary-path bypass entirely.
                if data.model and data.model != config.orchestrator.model.name:
                    from orchestration.errors import orchestration_error
                    from orchestration.policy import Reason

                    raise orchestration_error(
                        404,
                        Reason.MODEL_NOT_CONFIGURED.value,
                        f"{data.model!r} is not the configured "
                        f"model ({config.orchestrator.model.name!r})",
                    )
                await check_model_container()
            elif data.model:
                await load_inline_model(data.model, request)
            else:
                await check_model_container()
            model_path = model.container.model_dir

        # Prepare raw prompt
        if model.container.prompt_template is None:
            error_message = handle_request_error(
                "Chat completions are disabled because a prompt template is not set.",
                exc_info=False,
            ).error.message
            raise HTTPException(422, error_message)
        prompt, mm_embeddings = await apply_chat_template(data)
        await write_chat_completion_prompt_log(request, prompt)

        # Set an empty JSON schema if the request wants a JSON response
        if data.response_format.type == "json":
            data.json_schema = {"type": "object"}

        # Also accept specific schema from response_format
        if data.response_format.type == "json_schema":
            data.json_schema = data.response_format.json_schema

        disconnect_handler = DisconnectHandler(request, f"{request_tag(request)} chat/completions")
        await disconnect_handler.poll()

        if data.stream and not config.developer.disable_request_streaming:
            model.check_context_length(prompt, data, mm_embeddings)
            response = EventSourceResponse(
                stream_generate_chat_completion(
                    prompt,
                    mm_embeddings,
                    data,
                    request,
                    model_path,
                    disconnect_handler,
                    lease_ctx=lease_ctx,
                ),
                ping=get_sse_ping_interval(),
            )
            reached_wrapper = True
            return response
        else:
            response = await generate_chat_completion(
                prompt,
                mm_embeddings,
                data,
                request,
                model_path,
                disconnect_handler,
                lease_ctx=lease_ctx,
            )
            reached_wrapper = True
            return response

    except (CancelledError, InvalidStateError) as ex:
        raise HTTPException(422, "/v1/chat/completions request cancelled by user.") from ex

    finally:
        # C4 release — see the completions handler for the rationale.
        if not reached_wrapper:
            release_lease_in_finally(lease_ctx, [])


# Apply template endpoint (llama-server compatible)
@router.post("/apply-template", dependencies=[Depends(check_api_key)])
@router.post("/v1/apply-template", dependencies=[Depends(check_api_key)])
async def apply_template_request(request: Request, data: ChatCompletionRequest) -> dict:
    """
    Renders the chat template for the given messages without generating and
    returns the resulting prompt. Clients use this to probe the template, e.g.
    whether it reacts to a thinking toggle.
    """

    # F10/R12: a template render dereferences the live container/tokenizer, so
    # in enabled mode it holds a short reader pin (never cold-loads, never
    # refreshes the inference TTL). Disabled mode keeps upstream behaviour.
    from orchestration.install import LeaseDenied, acquire_reader_pin, release_reader_pin

    pin = None
    try:
        pin, _ = await acquire_reader_pin(f"read:{request.state.id}")
    except LeaseDenied as denied:
        raise denied.exc from None

    try:
        await check_model_container()

        if model.container.prompt_template is None:
            error_message = handle_request_error(
                "Cannot apply a template because a prompt template is not set.",
                exc_info=False,
            ).error.message
            raise HTTPException(422, error_message)

        prompt, _ = await apply_chat_template(data)
        return {"prompt": prompt}
    finally:
        release_reader_pin(pin)


# Embeddings endpoint
@router.post(
    "/v1/embeddings",
    dependencies=[Depends(check_api_key), Depends(check_embeddings_container)],
)
async def embeddings(request: Request, data: EmbeddingsRequest) -> EmbeddingsResponse:
    embeddings_task = asyncio.create_task(get_embeddings(data, request))
    response = await run_with_request_disconnect(
        request,
        embeddings_task,
        f"{request_tag(request)} embeddings cancelled by client",
    )

    return response
