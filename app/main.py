import asyncio
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.engine import get_engine
from app.routers import cv, health, segment


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
