"""GET /api/health and GET /api/info endpoints."""

from fastapi import APIRouter

from app.engine import get_engine
from app.schemas import HealthResponse, InfoResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, status_code=200)
def health() -> HealthResponse:
    """
    Liveness + readiness probe.

    Always returns HTTP 200. The `ready` field indicates whether the SAM
    model has finished loading. Gateway health probes should treat
    ``ready == false`` as a temporary warm-up state rather than a failure.
    """
    engine = get_engine()
    return HealthResponse(
        status="ok",
        ready=engine.ready,
        model_variant=engine.model_variant,
        load_error=engine.load_error,
    )


@router.get("/info", response_model=InfoResponse, status_code=200)
def info() -> InfoResponse:
    """Return GPU device info, VRAM usage, model variant, and readiness status."""
    engine = get_engine()
    vram = engine.vram_info()
    return InfoResponse(
        ready=engine.ready,
        model_variant=engine.model_variant,
        device=engine.device,
        device_name=vram.get("device_name"),
        vram_total_mb=vram.get("vram_total_mb"),
        vram_reserved_mb=vram.get("vram_reserved_mb"),
        vram_allocated_mb=vram.get("vram_allocated_mb"),
        load_error=engine.load_error,
    )
