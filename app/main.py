import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.engine import get_engine
from app.routers import cv, health, segment

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start background model loading on startup; clean up on shutdown."""
    engine = get_engine()
    t = threading.Thread(target=engine.load, name="sam-loader", daemon=True)
    t.start()
    yield


app = FastAPI(
    title="CV SAM Service",
    description="FastAPI service for Meta's Segment Anything Model (SAM)",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(health.router, prefix="/api")
app.include_router(segment.router, prefix="/api")
app.include_router(cv.router, prefix="/api")

# Serve pre-built React UI static assets when present.
# The /assets mount is registered only when the directory exists (i.e. inside
# the Docker image after the multi-stage build).  JS/CSS chunks are served
# directly; everything else falls through to the SPA index.html handler.
_ASSETS_DIR = os.path.join(_STATIC_DIR, "assets")
if os.path.isdir(_ASSETS_DIR):
    app.mount("/assets", StaticFiles(directory=_ASSETS_DIR), name="assets")


@app.get("/{full_path:path}", include_in_schema=False)
async def spa_fallback(full_path: str) -> FileResponse:
    """Serve index.html for any path not matched by an API route.

    Returns HTTP 404 when the UI has not been built yet so that operator
    errors produce a clear message rather than a Python traceback.
    """
    index = os.path.join(_STATIC_DIR, "index.html")
    if not os.path.isfile(index):
        raise HTTPException(
            status_code=404,
            detail="UI not built – run 'cd ui && npm run build' first.",
        )
    return FileResponse(index)
