from __future__ import annotations

import logging

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.core.config import get_settings
from app.core.errors import PipelineError

settings = get_settings()

logging.basicConfig(level=settings.log_level)
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(
        getattr(logging, settings.log_level.upper(), logging.INFO)
    ),
)
logger = structlog.get_logger(__name__)

app = FastAPI(
    title="HH Goa 2026 — Task 2 Voice RAG",
    description=(
        "Bootstrap/scaffolding stage. Core RAG pipeline is NOT implemented yet. "
        "See docs/ARCHITECTURE.md and docs/DEEPSEEK_IMPLEMENTATION.md."
    ),
    version="0.1.0",
)


@app.exception_handler(PipelineError)
async def pipeline_error_handler(request: Request, exc: PipelineError) -> JSONResponse:
    """
    Typed error -> JSON response. Never exposes stack traces or secrets —
    only the typed error_code and a human-readable detail message that the
    exception itself was constructed with.
    """
    request_id = getattr(request.state, "request_id", None)
    logger.error(
        "pipeline_error",
        error_code=exc.error_code,
        detail=exc.detail,
        request_id=request_id,
    )
    return JSONResponse(
        status_code=exc.http_status,
        content={"error_code": exc.error_code, "detail": exc.detail, "request_id": request_id},
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
