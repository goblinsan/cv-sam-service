"""Shared pytest fixtures for cv-sam-service tests."""

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image


# ---------------------------------------------------------------------------
# Mock engine fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_engine():
    """A MagicMock SAMEngine that behaves as fully ready."""
    engine = MagicMock()
    engine.ready = True
    engine.model_variant = "vit_b"
    engine.device = "cpu"
    engine.load_error = None

    engine.vram_info.return_value = {
        "device_name": "cpu",
        "vram_total_mb": None,
        "vram_reserved_mb": None,
        "vram_allocated_mb": None,
    }

    # Simulate predict(): returns (masks, scores, logits)
    h, w = 64, 64
    masks = np.zeros((3, h, w), dtype=bool)
    masks[0, 20:44, 20:44] = True
    scores = np.array([0.95, 0.80, 0.70], dtype=np.float32)
    logits = np.zeros((3, 256, 256), dtype=np.float32)
    engine.predict.return_value = (masks, scores, logits)

    # Simulate predict_auto(): returns list of segment dicts
    auto_mask = np.zeros((h, w), dtype=bool)
    auto_mask[10:30, 10:30] = True
    engine.predict_auto.return_value = [
        {
            "segmentation": auto_mask,
            "area": 400,
            "bbox": [10, 10, 20, 20],
            "predicted_iou": 0.92,
            "stability_score": 0.97,
        }
    ]
    return engine


@pytest.fixture()
def client(mock_engine):
    """TestClient with the global engine replaced by mock_engine."""
    from app.main import app

    with (
        patch("app.routers.health.get_engine", return_value=mock_engine),
        patch("app.routers.segment.get_engine", return_value=mock_engine),
        patch("app.engine.get_engine", return_value=mock_engine),
    ):
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


@pytest.fixture()
def test_image_bytes() -> bytes:
    """Minimal 64×64 RGB PNG image as raw bytes."""
    img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
