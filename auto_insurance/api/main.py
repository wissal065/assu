"""Application entry point for the auto insurance FastAPI service."""

import logging
import time
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from starlette.responses import Response

from auto_insurance.api.endpoints.health import router as health_router
from auto_insurance.api.endpoints.predict import router as predict_router
from auto_insurance.api.logging_utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AutoAssur — Motor Insurance Pricing API",
    description=(
        "REST API powered by two XGBoost models to compute the pure premium in real time.\n\n"
        "**Actuarial formula**: Pure Premium = Frequency x Severity\n\n"
        "- `/predict/frequency` — claim probability\n"
        "- `/predict/severity` — average claim cost\n"
        "- `/predict/premium` — full pure premium\n"
        "- `/predict/explain` — pure premium + risk factors (SHAP)"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.include_router(health_router)
app.include_router(predict_router)


@app.middleware("http")
async def add_request_context(request: Request, call_next) -> Response:
    """
    Attach a unique request ID to every request and log the full lifecycle.

    Reads X-Request-ID from incoming headers if present, otherwise
    generates a short UUID. Attaches the ID to the response headers
    and logs method, path, status code, and latency for every request.
    """
    request_id = request.headers.get("x-request-id", str(uuid4())[:8])
    start_time = time.perf_counter()

    logger.info(
        "Request started | id=%s method=%s path=%s",
        request_id,
        request.method,
        request.url.path,
    )

    response = await call_next(request)

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
    response.headers["X-Request-ID"] = request_id

    logger.info(
        "Request completed | id=%s method=%s path=%s status=%s latency_ms=%.2f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        latency_ms,
    )

    return response


_DASHBOARD = Path(__file__).parent / "dashboard.html"


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def landing_page() -> HTMLResponse:
    """Serve the interactive pricing dashboard."""
    return HTMLResponse(content=_DASHBOARD.read_text(encoding="utf-8"))
