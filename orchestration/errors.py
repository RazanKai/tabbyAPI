"""R12 structured error bodies for orchestration rejections.

SPEC R12 says orchestration failures "follow upstream error-body conventions" —
``{"error": {"message", "type", "param", "code"}}``, the shape
``common/errors.py:context_length_error_content`` already emits — and assigns each
rejection a stable code family. This module is the single authority for that shape so
every rejection site produces one body rather than ten ad-hoc ``{"detail": ...}``
strings.

Why it exists at all: an inference-only client is forbidden the admin status route
(R12), so the error body is its *only* channel for learning why it was refused, and
only the structured ``code`` is machine-readable. The reason code was always present
in the human-readable message; the shape is what this module adds.

``OrchestrationHTTPException`` subclasses ``HTTPException`` so a raise site keeps the
ordinary FastAPI control flow (raise → response) and, crucially, so any caller that
never installs the app-level handler still gets a sane ``detail`` body rather than a
traceback.
"""

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

#: The modest advisory Retry-After carried by temporary availability errors. Not a
#: guarantee and not a command to retry indefinitely (R12).
RETRY_AFTER_SECONDS = 5


def error_content(code: str, message: str, *, retryable: bool = False) -> dict:
    """Build the stable orchestration error body.

    Mirrors ``common/errors.py:context_length_error_content`` so orchestration errors
    look like every other structured error this server already emits, rather than
    introducing a third shape.
    """

    body = {
        "error": {
            "message": message,
            "type": "orchestrator_error",
            "param": None,
            "code": code,
        }
    }
    if retryable:
        body["error"]["retry_after_seconds"] = RETRY_AFTER_SECONDS
    return body


class OrchestrationHTTPException(HTTPException):
    """An HTTPException whose body follows R12's structured shape.

    ``detail`` stays the human-readable message (so ``exc.detail`` keeps working for
    existing callers), while ``code`` carries the stable machine-readable reason.
    """

    def __init__(self, status_code: int, code: str, message: str, *, retryable: bool = False):
        super().__init__(status_code=status_code, detail=message)
        self.code = code
        self.retryable = retryable
        if retryable:
            self.headers = {"Retry-After": str(RETRY_AFTER_SECONDS)}
        else:
            self.headers = None


def orchestration_error(
    status_code: int, code: str, message: str, *, retryable: bool = False
) -> OrchestrationHTTPException:
    """Build an R12-shaped rejection.

    The stable code is always prefixed into the message as well, because some
    intermediaries (and humans reading a log line) only ever see ``detail``.
    """

    if code not in message:
        message = f"{code}: {message}" if message else code
    return OrchestrationHTTPException(status_code, code, message, retryable=retryable)


async def orchestration_exception_handler(
    request: Request, exc: OrchestrationHTTPException
) -> JSONResponse:
    """Render :class:`OrchestrationHTTPException` in the R12 structured shape."""

    return JSONResponse(
        status_code=exc.status_code,
        content=error_content(exc.code, exc.detail, retryable=exc.retryable),
        headers=exc.headers,
    )
