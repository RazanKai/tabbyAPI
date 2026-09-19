import asyncio
import pathlib
from typing import Optional
from common.multimodal import MultimodalEmbeddingWrapper
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from sse_starlette import EventSourceResponse

from common import model, sampling
from common.auth import check_admin_key, check_api_key, get_key_permission
from common.downloader import hf_repo_download
from common.model import check_embeddings_container, check_model_container
from common.networking import (
    get_sse_ping_interval,
    handle_request_error,
    run_with_request_disconnect,
)
from common.tabby_config import config
from common.templating import PromptTemplate, get_all_templates
from common.utils import unwrap
from common.health import HealthManager
from endpoints.OAI.utils.chat_completion import format_messages_with_template
from endpoints.core.types.auth import AuthPermissionResponse
from endpoints.core.types.download import DownloadRequest, DownloadResponse
from endpoints.core.types.lora import LoraList, LoraLoadRequest, LoraLoadResponse
from endpoints.core.types.model import (
    EmbeddingModelLoadRequest,
    ModelCard,
    ModelDefaultGenerationSettings,
    ModelList,
    ModelLoadRequest,
    ModelLoadResponse,
    ModelPropsModalities,
    ModelPropsResponse,
)
from endpoints.core.types.health import HealthCheckResponse
from endpoints.core.types.sampler_overrides import (
    SamplerOverrideListResponse,
    SamplerOverrideSwitchRequest,
)
from endpoints.core.types.template import TemplateList, TemplateSwitchRequest
from endpoints.core.types.token import (
    TokenDecodeRequest,
    TokenDecodeResponse,
    TokenEncodeRequest,
    TokenEncodeResponse,
)
from endpoints.core.utils.lora import get_active_loras, get_lora_list
from endpoints.core.utils.model import (
    get_current_model,
    get_current_model_list,
    get_dummy_models,
    get_model_list,
    stream_explicit_load_progress,
    stream_model_load,
    _load_tasks,
)
from orchestration.errors import orchestration_error
from orchestration.lifecycle import LifecycleError
from orchestration.policy import Reason


router = APIRouter()


def _reject_unsupported_surface(name: str) -> None:
    """Fail an admin mutation surface orchestrated V1 does not govern (F10/R12).

    LoRA/template mutation and process-wide sampler overrides alter model state
    (or global generation defaults) outside the coordinator's admission boundary.
    In disabled mode these routes keep upstream behaviour byte-for-byte; in
    enabled mode they are rejected explicitly with a stable code rather than
    left as ungoverned paths (SPEC §2: "Reject incompatible configuration or
    mutation instead of leaving a bypass").
    """

    if config.orchestrator.enabled:
        raise orchestration_error(
            400,
            Reason.UNSUPPORTED_PROFILE.value,
            f"{name} is rejected in orchestrated mode; "
            "LoRA/template mutation and sampler overrides are outside the "
            "calibrated single-profile scope (R12)",
        )


def _lifecycle_reason_code(message: str) -> str:
    """Extract the stable reason code a ``LifecycleError`` message carries.

    ``reserve_explicit_load`` raises with the code as the whole message when it
    comes from the blocker list (`_first_reason(blockers).value`), or as
    ``transition already active: <kind>`` when a transition is in flight. Both
    shapes are mapped here so the admin load reports the same code family an
    inference client would see for the same condition, instead of collapsing
    everything to `model_transition` (R12).
    """

    text = (message or "").strip()
    if text.startswith("transition already active"):
        return Reason.MODEL_TRANSITION.value
    # The blocker form is exactly a Reason value; anything unrecognised degrades to
    # the generic transition code rather than inventing a new one.
    for reason in Reason:
        if text == reason.value:
            return reason.value
    return Reason.MODEL_TRANSITION.value


def _reason_is_temporary(code: str) -> bool:
    """Whether a modest Retry-After belongs on this refusal.

    Mirrors ``orchestration.api.is_temporary`` for the codes the admin load can
    emit: pause and fault are not "retry in 5s and it will work" conditions.
    """

    from orchestration.api import is_temporary

    for reason in Reason:
        if reason.value == code:
            return is_temporary(reason)
    return False


# Healthcheck endpoint
@router.get("/health")
async def healthcheck(response: Response) -> HealthCheckResponse:
    """Get the current service health status"""
    healthy, issues = await HealthManager.is_service_healthy()

    if not healthy:
        response.status_code = 503

    return HealthCheckResponse(status="healthy" if healthy else "unhealthy", issues=issues)


@router.get("/.well-known/serviceinfo")
async def service_info():
    return JSONResponse(
        content={
            "version": 0.1,
            "software": {
                "name": "TabbyAPI",
                "repository": "https://github.com/theroyallab/tabbyAPI",
                "homepage": "https://github.com/theroyallab/tabbyAPI",
            },
            "api": {
                "openai": {
                    "name": "OpenAI API",
                    "relative_url": "/v1",
                    "documentation": "https://theroyallab.github.io/tabbyAPI",
                    "version": 1,
                },
                "koboldai": {
                    "name": "KoboldAI API",
                    "relative_url": "/api",
                    "documentation": "https://theroyallab.github.io/tabbyAPI",
                    "version": 1,
                },
            },
        }
    )


# Model list endpoint
@router.get("/v1/models", dependencies=[Depends(check_api_key)])
@router.get("/v1/model/list", dependencies=[Depends(check_api_key)])
async def list_models(request: Request) -> ModelList:
    """
    Lists all models in the model directory.

    Requires an admin key to see all models.
    """

    model_dir = config.model.model_dir
    model_path = pathlib.Path(model_dir)

    draft_model_dir = config.draft_model.draft_model_dir

    # F10/R12 (review minor): the non-admin branch dereferences the live
    # container (`get_current_model_list` reads `container.model_dir` and
    # `model_info()`), so it is a container reader and needs the same short pin
    # as the routed readers — a soft reader that touches the container must not
    # race a teardown and surface an ungoverned 500.
    from orchestration.install import LeaseDenied, acquire_reader_pin, release_reader_pin

    pin = None
    try:
        if get_key_permission(request) == "admin":
            models = get_model_list(model_path.resolve(), draft_model_dir)
        else:
            pin, _ = await acquire_reader_pin(f"read:{request.state.id}")
            models = await get_current_model_list()
    except LeaseDenied as denied:
        raise denied.exc from None
    finally:
        release_reader_pin(pin)

    if config.model.use_dummy_models:
        models.data[:0] = get_dummy_models()

    return models


# Currently loaded model endpoint
@router.get(
    "/v1/model",
    dependencies=[Depends(check_api_key), Depends(check_model_container)],
)
async def current_model(request: Request) -> ModelCard:
    """Returns the currently loaded model."""

    # F10/R12: this read dereferences the live container, so it holds a short
    # reader pin: it cannot cold-load (refused 503 while the model is not
    # READY) and it never refreshes the inference TTL (R08). Disabled mode
    # keeps upstream behaviour exactly.
    from orchestration.install import LeaseDenied, acquire_reader_pin, release_reader_pin

    pin = None
    try:
        pin, _ = await acquire_reader_pin(f"read:{request.state.id}")
        return get_current_model()
    except LeaseDenied as denied:
        raise denied.exc from None
    finally:
        release_reader_pin(pin)


@router.get("/props", dependencies=[Depends(check_api_key), Depends(check_model_container)])
async def model_props(request: Request) -> ModelPropsResponse:
    """
    Returns specific properties of a model for clients.

    To get all properties, use /v1/model instead.
    """

    # F10/R12: reader pin — same contract as /v1/model above.
    from orchestration.install import LeaseDenied, acquire_reader_pin, release_reader_pin

    pin = None
    try:
        pin, _ = await acquire_reader_pin(f"read:{request.state.id}")
        current_model_card = get_current_model()
        resp = ModelPropsResponse(
            total_slots=current_model_card.parameters.max_batch_size,
            model_path=str(model.container.model_dir),
            default_generation_settings=ModelDefaultGenerationSettings(
                n_ctx=current_model_card.parameters.max_seq_len,
            ),
            modalities=ModelPropsModalities(vision=bool(current_model_card.parameters.use_vision)),
        )

        if current_model_card.parameters.prompt_template_content:
            resp.chat_template = current_model_card.parameters.prompt_template_content

        return resp
    except LeaseDenied as denied:
        raise denied.exc from None
    finally:
        release_reader_pin(pin)


@router.get("/v1/model/draft/list", dependencies=[Depends(check_api_key)])
async def list_draft_models(request: Request) -> ModelList:
    """
    Lists all draft models in the model directory.

    Requires an admin key to see all draft models.
    """

    # F10/R12 (review minor): the non-admin branch is a container reader
    # (`get_current_model_list(model_type="draft")` reads the live container).
    from orchestration.install import LeaseDenied, acquire_reader_pin, release_reader_pin

    pin = None
    try:
        if get_key_permission(request) == "admin":
            draft_model_dir = config.draft_model.draft_model_dir
            draft_model_path = pathlib.Path(draft_model_dir)

            models = get_model_list(draft_model_path.resolve())
        else:
            pin, _ = await acquire_reader_pin(f"read:{request.state.id}")
            models = await get_current_model_list(model_type="draft")
    except LeaseDenied as denied:
        raise denied.exc from None
    finally:
        release_reader_pin(pin)

    return models


def _load_overrides_rejected(data) -> Optional[str]:
    """Name client-supplied load fields an orchestrated load must not accept.

    R12/SPEC §2: the coordinator performs a narrowly preauthorized load of ONE
    calibrated profile. The orchestrated branch builds its load kwargs from
    ``config.orchestrator.model`` only, so any other client-supplied field that
    would change the effective load (a draft model, an ad-hoc context/cache
    size, a prompt template, vision) would be *silently ignored* — accepted but
    ungoverned. Refusing explicitly is what R12 asks for.

    ``None`` means the request carries no such override.
    """

    if data.draft_model is not None:
        return (
            "draft_model is rejected in orchestrated mode; drafting is outside "
            "V1's calibrated envelope (SPEC §2)"
        )
    for field in ("max_seq_len", "cache_size", "cache_mode", "chunk_size", "max_batch_size"):
        if getattr(data, field, None) is not None:
            return (
                f"{field} is rejected in orchestrated mode; the calibrated "
                f"profile in orchestrator.model is the only load envelope (R14)"
            )
    for field in ("prompt_template", "vision", "gpu_split", "rope_scale", "rope_alpha"):
        value = getattr(data, field, None)
        if value not in (None, [], False):
            return (
                f"{field} is rejected in orchestrated mode; it is not part of the "
                f"calibrated profile (R12/SPEC §2)"
            )
    return None


# Load model endpoint
@router.post("/v1/model/load", dependencies=[Depends(check_admin_key)])
async def load_model(data: ModelLoadRequest) -> ModelLoadResponse:
    """Loads a model into the model container. This returns an SSE stream."""

    # Verify request parameters
    if not data.model_name:
        error_message = handle_request_error(
            "A model name was not provided for load.",
            exc_info=False,
        ).error.message

        raise HTTPException(400, error_message)

    model_path = pathlib.Path(config.model.model_dir)
    model_path = model_path / data.model_name

    if not model_path.exists():
        error_message = handle_request_error(
            "Could not find the model path for load. Check model name or config.yml?",
            exc_info=False,
        ).error.message

        raise HTTPException(400, error_message)

    # Enabled mode: explicit loads use the SAME admission/transition machinery as
    # inference demand (R12) — the coordinator is the only load authority. The
    # configured model may be loaded through admission (quiet window + capacity +
    # transition token); anything else is refused, and skip_queue is rejected
    # because it could interrupt active work.
    if config.orchestrator.enabled:
        from orchestration.install import runtime

        configured = config.orchestrator.model.name
        if data.model_name != configured:
            raise orchestration_error(
                404,
                Reason.MODEL_NOT_CONFIGURED.value,
                f"{data.model_name!r} is not the configured model ({configured!r})",
            )
        if data.skip_queue:
            raise orchestration_error(
                400,
                Reason.UNSUPPORTED_PROFILE.value,
                "skip_queue is rejected in orchestrated mode; "
                "explicit loads use the same admission machinery as inference (R12)",
            )
        # R12/SPEC §2: the orchestrated load uses the CALIBRATED profile only, so
        # any client-supplied field that would alter the effective load must be
        # REFUSED rather than silently ignored. Silently dropping them left a
        # load-altering request accepted-but-ungoverned (review major); a draft
        # model in particular is outside V1's calibrated envelope entirely.
        rejected = _load_overrides_rejected(data)
        if rejected is not None:
            raise orchestration_error(400, Reason.UNSUPPORTED_PROFILE.value, rejected)
        coordinator_obj = runtime.orchestrator
        if coordinator_obj is None:
            raise orchestration_error(
                503,
                Reason.ORCHESTRATOR_FAULT.value,
                "coordinator is not installed",
            )
        try:
            # Reserve the transition (this is the admission boundary for explicit
            # demand). Raises LifecycleError carrying the REAL reason code in its
            # message (`orchestrator_paused`, `external_gpu_busy`, `gpu_not_quiet`,
            # `insufficient_vram`, …), which must be reported as-is: R12's table
            # exists so a client can tell "paused" from "retrying is reasonable",
            # and collapsing every refusal to `model_transition` destroys exactly
            # that distinction.
            coordinator_obj.reserve_explicit_load()
        except LifecycleError as exc:
            reason = _lifecycle_reason_code(str(exc))
            raise orchestration_error(
                503, reason, str(exc), retryable=_reason_is_temporary(reason)
            ) from exc
        except Exception as exc:
            raise orchestration_error(
                503, Reason.GPU_NOT_QUIET.value, str(exc), retryable=True
            ) from exc

        # The load is executed by the coordinator's production deps (same
        # calibrated-envelope kwargs as inference demand) and observed to
        # completion, exactly like the detached admin load upstream keeps in
        # _load_tasks — integrate, not duplicate.
        load_task = asyncio.create_task(coordinator_obj.execute_explicit_load())
        _load_tasks.add(load_task)
        load_task.add_done_callback(_load_tasks.discard)

        return EventSourceResponse(
            stream_explicit_load_progress(load_task, model_path),
            ping=get_sse_ping_interval(),
        )

    return EventSourceResponse(stream_model_load(data, model_path), ping=get_sse_ping_interval())


# Unload model endpoint
@router.post(
    "/v1/model/unload",
    dependencies=[Depends(check_admin_key), Depends(check_model_container)],
)
async def unload_model():
    """Unloads the currently loaded model."""

    # Enabled mode: an explicit unload initiates a safe DRAIN, not a forced
    # interruption — upstream's hard-coded skip_wait=True cancels active jobs,
    # which R08 forbids under orchestration (integration map §5). The drain
    # reason is EXPLICIT_UNLOAD so a later resume() cannot cancel it.
    if config.orchestrator.enabled:
        from orchestration.install import runtime

        coordinator_obj = runtime.orchestrator
        if coordinator_obj is not None:
            await coordinator_obj.request_unload(Reason.EXPLICIT_UNLOAD)
            return

    await model.unload_model(skip_wait=True)


@router.post("/v1/download", dependencies=[Depends(check_admin_key)])
async def download_model(request: Request, data: DownloadRequest) -> DownloadResponse:
    """Downloads a model from HuggingFace."""

    try:
        download_task = asyncio.create_task(hf_repo_download(**data.model_dump()))

        # For now, the downloader and request data are 1:1
        download_path = await run_with_request_disconnect(
            request,
            download_task,
            "Download request cancelled by user. Files have been cleaned up.",
        )

        return DownloadResponse(download_path=str(download_path))
    except Exception as exc:
        error_message = handle_request_error(str(exc)).error.message

        raise HTTPException(400, error_message) from exc


# Lora list endpoint
@router.get("/v1/loras", dependencies=[Depends(check_api_key)])
@router.get("/v1/lora/list", dependencies=[Depends(check_api_key)])
async def list_all_loras(request: Request) -> LoraList:
    """
    Lists all LoRAs in the lora directory.

    Requires an admin key to see all LoRAs.
    """

    if get_key_permission(request) == "admin":
        lora_path = pathlib.Path(config.lora.lora_dir)
        loras = get_lora_list(lora_path.resolve())
    else:
        # F10/R12 (review minor): `get_active_loras()` reads the live container's
        # LoRA state, so it is a container reader and takes the same short pin.
        from orchestration.install import LeaseDenied, acquire_reader_pin, release_reader_pin

        pin = None
        try:
            pin, _ = await acquire_reader_pin(f"read:{request.state.id}")
            loras = get_active_loras()
        except LeaseDenied as denied:
            raise denied.exc from None
        finally:
            release_reader_pin(pin)

    return loras


# Currently loaded loras endpoint
@router.get(
    "/v1/lora",
    dependencies=[Depends(check_api_key), Depends(check_model_container)],
)
async def active_loras() -> LoraList:
    """Returns the currently loaded loras."""

    return get_active_loras()


# Load lora endpoint
@router.post(
    "/v1/lora/load",
    dependencies=[Depends(check_admin_key), Depends(check_model_container)],
)
async def load_lora(data: LoraLoadRequest) -> LoraLoadResponse:
    """Loads a LoRA into the model container."""

    _reject_unsupported_surface("POST /v1/lora/load")

    if not data.loras:
        error_message = handle_request_error(
            "List of loras to load is not found.",
            exc_info=False,
        ).error.message

        raise HTTPException(400, error_message)

    lora_dir = pathlib.Path(config.lora.lora_dir)
    if not lora_dir.exists():
        error_message = handle_request_error(
            "A parent lora directory does not exist for load. Check your config.yml?",
            exc_info=False,
        ).error.message

        raise HTTPException(400, error_message)

    load_result = await model.load_loras(lora_dir, **data.model_dump(), skip_wait=data.skip_queue)

    return LoraLoadResponse(
        success=unwrap(load_result.get("success"), []),
        failure=unwrap(load_result.get("failure"), []),
    )


# Unload lora endpoint
@router.post(
    "/v1/lora/unload",
    dependencies=[Depends(check_admin_key), Depends(check_model_container)],
)
async def unload_loras():
    """Unloads the currently loaded loras."""

    _reject_unsupported_surface("POST /v1/lora/unload")

    await model.unload_loras()


@router.get("/v1/model/embedding/list", dependencies=[Depends(check_api_key)])
async def list_embedding_models(request: Request) -> ModelList:
    """
    Lists all embedding models in the model directory.

    Requires an admin key to see all embedding models.
    """

    if get_key_permission(request) == "admin":
        embedding_model_dir = config.embeddings.embedding_model_dir
        embedding_model_path = pathlib.Path(embedding_model_dir)

        models = get_model_list(embedding_model_path.resolve())
    else:
        models = await get_current_model_list(model_type="embedding")

    return models


@router.get(
    "/v1/model/embedding",
    dependencies=[Depends(check_api_key), Depends(check_embeddings_container)],
)
async def get_embedding_model() -> ModelCard:
    """Returns the currently loaded embedding model."""
    models = await get_current_model_list(model_type="embedding")

    return models.data[0]


@router.post("/v1/model/embedding/load", dependencies=[Depends(check_admin_key)])
async def load_embedding_model(
    request: Request, data: EmbeddingModelLoadRequest
) -> ModelLoadResponse:
    _reject_unsupported_surface("POST /v1/model/embedding/load")

    # Verify request parameters
    if not data.embedding_model_name:
        error_message = handle_request_error(
            "A model name was not provided for load.",
            exc_info=False,
        ).error.message

        raise HTTPException(400, error_message)

    embedding_model_dir = pathlib.Path(config.embeddings.embedding_model_dir)
    embedding_model_path = embedding_model_dir / data.embedding_model_name

    if not embedding_model_path.exists():
        error_message = handle_request_error(
            "Could not find the embedding model path for load. "
            + "Check model name or config.yml?",
            exc_info=False,
        ).error.message

        raise HTTPException(400, error_message)

    try:
        load_task = asyncio.create_task(
            model.load_embedding_model(embedding_model_path, **data.model_dump())
        )
        await run_with_request_disconnect(
            request, load_task, "Embedding model load request cancelled by user."
        )
    except Exception as exc:
        error_message = handle_request_error(str(exc)).error.message

        raise HTTPException(400, error_message) from exc

    response = ModelLoadResponse(
        model_type="embedding_model", module=1, modules=1, status="finished"
    )

    return response


@router.post(
    "/v1/model/embedding/unload",
    dependencies=[Depends(check_admin_key), Depends(check_embeddings_container)],
)
async def unload_embedding_model():
    """Unloads the current embedding model."""

    await model.unload_embedding_model()


# Encode tokens endpoint
@router.post(
    "/v1/token/encode",
    dependencies=[Depends(check_api_key), Depends(check_model_container)],
)
async def encode_tokens(request: Request, data: TokenEncodeRequest) -> TokenEncodeResponse:
    """Encodes a string or chat completion messages into tokens."""

    # F10/R12: reader pin — a tokenization read dereferences the live
    # container/tokenizer, so it holds a short pin (no cold-load, no TTL
    # refresh). Disabled mode keeps upstream behaviour exactly.
    from orchestration.install import LeaseDenied, acquire_reader_pin, release_reader_pin

    pin = None
    try:
        pin, _ = await acquire_reader_pin(f"read:{request.state.id}")
    except LeaseDenied as denied:
        raise denied.exc from None

    try:
        return await _encode_tokens_impl(request, data)
    finally:
        release_reader_pin(pin)


async def _encode_tokens_impl(request: Request, data: TokenEncodeRequest) -> TokenEncodeResponse:
    """The upstream encode_tokens body, unchanged (disabled-mode parity)."""

    mm_embeddings: Optional[MultimodalEmbeddingWrapper] = None

    if isinstance(data.text, str):
        text = data.text
    elif isinstance(data.text, list):
        if "oai" not in config.network.api_servers:
            error_message = handle_request_error(
                "Enable the OAI server to handle chat completion messages.",
                exc_info=False,
            ).error.message

            raise HTTPException(422, error_message)

        if not model.container.prompt_template:
            error_message = handle_request_error(
                "Cannot tokenize chat completion message because "
                + "a prompt template is not set.",
                exc_info=False,
            ).error.message

            raise HTTPException(422, error_message)

        template_vars = {
            **(data.template_vars or {}),
            "add_generation_prompt": False,
        }

        text, mm_embeddings, rendered_template_vars = await format_messages_with_template(
            data.text, template_vars
        )

        # Let encode_tokens be the sole authority on whether BOS is added.
        bos_token = rendered_template_vars.get("bos_token")
        if bos_token and text.startswith(bos_token):
            text = text.removeprefix(bos_token)
    else:
        error_message = handle_request_error(
            "Unable to tokenize the provided text. Check your formatting?",
            exc_info=False,
        ).error.message

        raise HTTPException(422, error_message)

    raw_tokens = model.container.encode_tokens(text, embeddings=mm_embeddings, **data.get_params())
    tokens = unwrap(raw_tokens, [])
    response = TokenEncodeResponse(tokens=tokens, length=len(tokens))

    return response


# Decode tokens endpoint
@router.post(
    "/v1/token/decode",
    dependencies=[Depends(check_api_key), Depends(check_model_container)],
)
async def decode_tokens(request: Request, data: TokenDecodeRequest) -> TokenDecodeResponse:
    """Decodes tokens into a string."""

    # F10/R12: reader pin — see encode_tokens above.
    from orchestration.install import LeaseDenied, acquire_reader_pin, release_reader_pin

    pin = None
    try:
        pin, _ = await acquire_reader_pin(f"read:{request.state.id}")
        message = model.container.decode_tokens(data.tokens, **data.get_params())
        response = TokenDecodeResponse(text=unwrap(message, ""))
        return response
    except LeaseDenied as denied:
        raise denied.exc from None
    finally:
        release_reader_pin(pin)


@router.get("/v1/auth/permission", dependencies=[Depends(check_api_key)])
async def key_permission(request: Request) -> AuthPermissionResponse:
    """
    Gets the access level/permission of a provided key in headers.

    Priority:
    - X-admin-key
    - X-api-key
    - Authorization
    """

    try:
        permission = get_key_permission(request)
        return AuthPermissionResponse(permission=permission)
    except ValueError as exc:
        error_message = handle_request_error(str(exc)).error.message

        raise HTTPException(400, error_message) from exc


@router.get("/v1/templates", dependencies=[Depends(check_api_key)])
@router.get("/v1/template/list", dependencies=[Depends(check_api_key)])
async def list_templates(request: Request) -> TemplateList:
    """
    Get a list of all templates.

    Requires an admin key to see all templates.
    """

    template_strings = []
    if get_key_permission(request) == "admin":
        templates = get_all_templates()
        template_strings = [template.stem for template in templates]
    else:
        if model.container and model.container.prompt_template:
            template_strings.append(model.container.prompt_template.name)

    return TemplateList(data=template_strings)


@router.post(
    "/v1/template/switch",
    dependencies=[Depends(check_admin_key), Depends(check_model_container)],
)
async def switch_template(data: TemplateSwitchRequest):
    """Switch the currently loaded template."""

    _reject_unsupported_surface("POST /v1/template/switch")

    if not data.prompt_template_name:
        error_message = handle_request_error(
            "New template name not found.",
            exc_info=False,
        ).error.message

        raise HTTPException(400, error_message)

    try:
        template_path = pathlib.Path("templates") / data.prompt_template_name
        model.container.prompt_template = await PromptTemplate.from_file(template_path)
    except FileNotFoundError as e:
        error_message = handle_request_error(
            f"The template name {data.prompt_template_name} doesn't exist. "
            + "Check the spelling?",
            exc_info=False,
        ).error.message

        raise HTTPException(400, error_message) from e


@router.post(
    "/v1/template/unload",
    dependencies=[Depends(check_admin_key), Depends(check_model_container)],
)
async def unload_template():
    """Unloads the currently selected template"""

    _reject_unsupported_surface("POST /v1/template/unload")

    model.container.prompt_template = None


# Sampler override endpoints
@router.get("/v1/sampling/overrides", dependencies=[Depends(check_api_key)])
@router.get("/v1/sampling/override/list", dependencies=[Depends(check_api_key)])
async def list_sampler_overrides(request: Request) -> SamplerOverrideListResponse:
    """
    List all currently applied sampler overrides.

    Requires an admin key to see all override presets.
    """

    if get_key_permission(request) == "admin":
        presets = sampling.get_all_presets()
    else:
        presets = []

    return SamplerOverrideListResponse(presets=presets, **sampling.overrides_container.model_dump())


@router.post(
    "/v1/sampling/override/switch",
    dependencies=[Depends(check_admin_key)],
)
async def switch_sampler_override(data: SamplerOverrideSwitchRequest):
    """Switch the currently loaded override preset"""

    _reject_unsupported_surface("POST /v1/sampling/override/switch")

    if data.preset:
        try:
            await sampling.overrides_from_file(data.preset)
        except FileNotFoundError as e:
            error_message = handle_request_error(
                f"Sampler override preset with name {data.preset} does not exist. "
                + "Check the spelling?",
                exc_info=False,
            ).error.message

            raise HTTPException(400, error_message) from e
    elif data.overrides:
        sampling.overrides_from_dict(data.overrides)
    else:
        error_message = handle_request_error(
            "A sampler override preset or dictionary wasn't provided.",
            exc_info=False,
        ).error.message

        raise HTTPException(400, error_message)


@router.post(
    "/v1/sampling/override/unload",
    dependencies=[Depends(check_admin_key)],
)
async def unload_sampler_override():
    """Unloads the currently selected override preset"""

    _reject_unsupported_surface("POST /v1/sampling/override/unload")

    sampling.overrides_from_dict({})
