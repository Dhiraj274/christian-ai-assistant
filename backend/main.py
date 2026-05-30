"""
FastAPI application entrypoint.
Configures CORS, middleware, lifespan events, and mounts all routers.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.api.routes.chat import router as chat_router
from backend.api.routes.image import router as image_router
from backend.api.routes.verse import router as verse_router
from backend.core.config import get_settings
from backend.core.logging import get_logger, setup_logging
from backend.models.schemas import HealthResponse

settings = get_settings()
logger = setup_logging(settings.log_level)
app_logger = get_logger("main")


# ── Lifespan ────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application startup/shutdown lifecycle manager.
    Initializes connections to external services on startup.
    """
    app_logger.info("Christianity-Focused AI Assistant starting up...")

    # Initialize SQLite Bible database on startup
    from backend.data.bible_sqlite import init_bible_db
    await init_bible_db(settings.sqlite_db_path)
    app_logger.info("Bible SQLite database initialized")

    # Initialize Langfuse tracing if configured
    if settings.observability_enabled:
        try:
            from langfuse import Langfuse
            app.state.langfuse = Langfuse(
                secret_key=settings.langfuse_secret_key,
                public_key=settings.langfuse_public_key,
                host=settings.langfuse_host,
            )
            app_logger.info("Langfuse observability initialized")
        except Exception as e:
            app_logger.warning(f"Langfuse init failed (continuing without tracing): {e}")

    app_logger.info("✝ Application ready to serve theological queries")
    yield

    # Cleanup
    app_logger.info("Application shutting down...")
    if hasattr(app.state, "langfuse"):
        app.state.langfuse.flush()


# ── App Factory ────────────────────────────────────────────────────────────


def create_app() -> FastAPI:
    """
    Application factory — creates and configures the FastAPI instance.
    Using a factory enables easier testing (fresh app per test).
    """
    app = FastAPI(
        title="Christianity-Focused AI Assistant",
        description=(
            "A production-grade, multi-agent AI assistant for Christian theological "
            "Q&A with deterministic hallucination prevention and strict safety guardrails."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
    )

    # ── CORS ───────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # ── Request Logging Middleware ────────────────────────────────────────
    @app.middleware("http")
    async def log_requests(request: Request, call_next):  # type: ignore
        import time
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        app_logger.info(
            "HTTP request",
            extra={
                "extra": {
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": round(duration_ms, 2),
                }
            },
        )
        return response

    # ── Global Exception Handler ──────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        app_logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "An unexpected error occurred. Please try again."},
        )

    # ── Static Files (Frontend) ───────────────────────────────────────────
    # Serves frontend/index.html at http://localhost:8000
    # Individual assets (styles.css, app.js) served from /static/*
    import os
    static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
    if os.path.isdir(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
        app_logger.info(f"Static files mounted from: {static_dir}")

    # ── Routes ────────────────────────────────────────────────────────────
    app.include_router(chat_router)
    app.include_router(image_router)
    app.include_router(verse_router)

    # Health check
    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health_check() -> HealthResponse:
        """Returns service health status and dependency availability."""
        services: dict[str, bool] = {
            "openai": bool(settings.openai_api_key),
            "pinecone": bool(settings.pinecone_api_key),
            "langfuse": settings.observability_enabled,
        }
        all_healthy = services["openai"] and services["pinecone"]  # langfuse is optional
        return HealthResponse(
            status="healthy" if all_healthy else "degraded",
            services=services,
        )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def root() -> HTMLResponse:
        """Serve the frontend UI at the root URL."""
        import os
        index_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "static", "index.html"
        )
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        return HTMLResponse(content="<h1>Frontend not found. Run from project root.</h1>", status_code=404)

    return app


# ── WSGI Entry Point ──────────────────────────────────────────────────────────

app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.app_env == "development",
        log_level=settings.log_level.lower(),
    )
    # Trigger reload
