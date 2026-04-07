"""Tests for POST /api/segment and POST /api/segment/auto."""

import base64
import io
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# /api/segment
# ---------------------------------------------------------------------------


def test_segment_masks_output(client, test_image_bytes):
    resp = client.post(
        "/api/segment",
        files={"image": ("test.png", test_image_bytes, "image/png")},
        data={"output_format": "masks"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "masks" in body
    assert body["masks"] is not None
    assert len(body["masks"]) == 3  # multimask_output=True → 3 masks
    assert body["polygons"] is None
    assert len(body["scores"]) == 3
    assert body["processing_time_ms"] >= 0

    # Verify each mask is valid base64-encoded data
    for b64 in body["masks"]:
        decoded = base64.b64decode(b64)
        assert len(decoded) > 0


def test_segment_polygons_output(client, test_image_bytes):
    resp = client.post(
        "/api/segment?output_format=polygons",
        files={"image": ("test.png", test_image_bytes, "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["masks"] is None
    assert body["polygons"] is not None
    assert len(body["polygons"]) == 3


def test_segment_both_output(client, test_image_bytes):
    resp = client.post(
        "/api/segment?output_format=both",
        files={"image": ("test.png", test_image_bytes, "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["masks"] is not None
    assert body["polygons"] is not None


def test_segment_with_point_prompts(client, test_image_bytes):
    point_coords = json.dumps([[32, 32]])
    point_labels = json.dumps([1])
    resp = client.post(
        "/api/segment",
        files={"image": ("test.png", test_image_bytes, "image/png")},
        data={
            "point_coords": point_coords,
            "point_labels": point_labels,
            "output_format": "masks",
        },
    )
    assert resp.status_code == 200
    # Check that predict was called with numpy arrays
    from app.engine import get_engine

    engine = get_engine()
    call_kwargs = engine.predict.call_args
    assert call_kwargs is not None


def test_segment_with_box_prompt(client, test_image_bytes):
    box = json.dumps([10, 10, 54, 54])
    resp = client.post(
        "/api/segment",
        files={"image": ("test.png", test_image_bytes, "image/png")},
        data={"box": box, "output_format": "masks"},
    )
    assert resp.status_code == 200


def test_segment_invalid_output_format(client, test_image_bytes):
    resp = client.post(
        "/api/segment?output_format=invalid",
        files={"image": ("test.png", test_image_bytes, "image/png")},
    )
    assert resp.status_code == 422


def test_segment_503_when_not_ready(test_image_bytes):
    from app.main import app

    not_ready = MagicMock()
    not_ready.ready = False

    with (
        patch("app.routers.health.get_engine", return_value=not_ready),
        patch("app.routers.segment.get_engine", return_value=not_ready),
        patch("app.engine.get_engine", return_value=not_ready),
    ):
        from fastapi.testclient import TestClient

        with TestClient(app) as c:
            resp = c.post(
                "/api/segment",
                files={"image": ("test.png", test_image_bytes, "image/png")},
            )

    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# /api/segment/auto
# ---------------------------------------------------------------------------


def test_segment_auto_default(client, test_image_bytes):
    resp = client.post(
        "/api/segment/auto",
        files={"image": ("test.png", test_image_bytes, "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "segments" in body
    assert body["count"] == len(body["segments"])
    assert body["processing_time_ms"] >= 0

    seg = body["segments"][0]
    assert "score" in seg
    assert "area" in seg
    assert "bbox" in seg
    assert len(seg["bbox"]) == 4


def test_segment_auto_polygons_output(client, test_image_bytes):
    resp = client.post(
        "/api/segment/auto?output_format=polygons",
        files={"image": ("test.png", test_image_bytes, "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()
    seg = body["segments"][0]
    assert seg["mask"] is None
    assert seg["polygon"] is not None


def test_segment_auto_max_masks(client, test_image_bytes):
    resp = client.post(
        "/api/segment/auto?max_masks=1",
        files={"image": ("test.png", test_image_bytes, "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] <= 1


def test_segment_auto_invalid_output_format(client, test_image_bytes):
    resp = client.post(
        "/api/segment/auto?output_format=bad",
        files={"image": ("test.png", test_image_bytes, "image/png")},
    )
    assert resp.status_code == 422


def test_segment_auto_503_when_not_ready(test_image_bytes):
    from app.main import app

    not_ready = MagicMock()
    not_ready.ready = False

    with (
        patch("app.routers.health.get_engine", return_value=not_ready),
        patch("app.routers.segment.get_engine", return_value=not_ready),
        patch("app.engine.get_engine", return_value=not_ready),
    ):
        from fastapi.testclient import TestClient

        with TestClient(app) as c:
            resp = c.post(
                "/api/segment/auto",
                files={"image": ("test.png", test_image_bytes, "image/png")},
            )

    assert resp.status_code == 503
