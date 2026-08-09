import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.core.config import Settings
from app.core.logging import get_logger

from app.api.routes import router
from app.api.auth import router as auth_router
from app.api.reports import router as reports_router
from app.api.appointments import router as appointments_router

from app.ml.inference import SkinCancerInference

settings = Settings()
logger = get_logger(__name__)

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[settings.rate_limit]
)


@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("Starting Skin Cancer backend")

    app.state.inference = SkinCancerInference(
        model_path=settings.model_path,
        device=settings.device
    )

    app.state.inference.load_model()

    logger.info("Model loaded successfully")

    yield

    logger.info("Shutting down Skin Cancer backend")


app = FastAPI(
    title="Skin Cancer Detection API",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter

app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler
)


@app.middleware("http")
async def add_request_timing(request: Request, call_next):

    if request.method == "OPTIONS":

        response = Response(status_code=204)

        response.headers["access-control-allow-origin"] = "*"

        response.headers["access-control-allow-methods"] = (
            "GET, POST, PUT, PATCH, DELETE, OPTIONS"
        )

        response.headers["access-control-allow-headers"] = (
            request.headers.get(
                "access-control-request-headers",
                "*"
            )
        )

        response.headers["access-control-max-age"] = "600"

        return response

    start = time.perf_counter()

    response = await call_next(request)

    response.headers["access-control-allow-origin"] = "*"

    duration_ms = round(
        (time.perf_counter() - start) * 1000,
        2
    )

    logger.info(
        "%s %s -> %s in %.2f ms",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms
    )

    return response


@app.exception_handler(Exception)
async def handle_unexpected_error(
    request: Request,
    exc: Exception
):

    if isinstance(exc, HTTPException):

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "detail": exc.detail
            },
        )

    logger.exception(
        "Unhandled exception for %s",
        request.url.path
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
        },
    )


# MAIN AI PREDICTION ROUTES
app.include_router(router)

# AUTH ROUTES
app.include_router(
    auth_router,
    prefix="/auth"
)

# REPORT ROUTES
app.include_router(
    reports_router,
    prefix="/reports"
)

# APPOINTMENT ROUTES
app.include_router(
    appointments_router,
    prefix="/appointments"
)


@app.get("/")
async def root():
    return {
        "status": "healthy",
        "message": "Skin Cancer Detection API is running. Go to /docs for the API documentation.",
        "documentation": "/docs",
        "health": "/api/health"
    }


if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=int(os.environ.get("PORT", settings.port)),
        reload=False
    )