"""Tests for optional image_url input on /api/analyze, /api/extract-palette,
/api/segment, and /api/segment/auto endpoints."""

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import requests as requests_lib
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_png_bytes(width: int = 64, height: int = 64) -> bytes:
    """Return a minimal solid-colour PNG as raw bytes."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:, :] = [100, 150, 200]
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _mock_response(content: bytes, status_code: int = 200) -> MagicMock:
    """Build a mock requests.Response whose iter_content yields *content*."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = {}

    chunk_size = 65536
    chunks = [content[i : i + chunk_size] for i in range(0, len(content), chunk_size)] or [b""]
    resp.iter_content.return_value = iter(chunks)
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# POST /api/analyze – image_url
# ---------------------------------------------------------------------------


def test_analyze_via_image_url(client):
    png = _make_png_bytes()

    with patch("app.utils.requests.get", return_value=_mock_response(png)) as mock_get:
        resp = client.post("/api/analyze?image_url=http://example.com/img.png")

    assert resp.status_code == 200
    body = resp.json()
    assert body["width"] == 64
    assert body["height"] == 64
    mock_get.assert_called_once()


def test_analyze_no_image_returns_422(client):
    """Calling /api/analyze without an image or image_url must return 422."""
    resp = client.post("/api/analyze")
    assert resp.status_code == 422


def test_analyze_bad_scheme_returns_422(client):
    resp = client.post("/api/analyze?image_url=ftp://example.com/img.png")
    assert resp.status_code == 422
    assert "scheme" in resp.json()["detail"].lower()


def test_analyze_http_error_returns_422(client):
    err = requests_lib.HTTPError("404 Not Found")
    mock_resp = _mock_response(b"", status_code=404)
    mock_resp.raise_for_status.side_effect = err

    with patch("app.utils.requests.get", return_value=mock_resp):
        resp = client.post("/api/analyze?image_url=http://example.com/missing.png")

    assert resp.status_code == 422
    assert "Failed to fetch" in resp.json()["detail"]


def test_analyze_oversized_image_url_returns_422(client):
    """Content-Length header reporting an oversized payload must be rejected."""
    mock_resp = _mock_response(b"x")
    mock_resp.headers = {"content-length": str(11 * 1024 * 1024)}  # 11 MB > 10 MB limit

    with patch("app.utils.requests.get", return_value=mock_resp):
        resp = client.post("/api/analyze?image_url=http://example.com/huge.png")

    assert resp.status_code == 422
    assert "limit" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# POST /api/extract-palette – image_url
# ---------------------------------------------------------------------------


def test_extract_palette_via_image_url(client):
    png = _make_png_bytes()

    with patch("app.utils.requests.get", return_value=_mock_response(png)):
        resp = client.post(
            "/api/extract-palette?image_url=http://example.com/img.png&num_colors=3"
        )

    assert resp.status_code == 200
    body = resp.json()
    assert len(body["colors"]) == 3


def test_extract_palette_no_image_returns_422(client):
    resp = client.post("/api/extract-palette")
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /api/segment – image_url
# ---------------------------------------------------------------------------


def test_segment_via_image_url(client):
    png = _make_png_bytes()

    with patch("app.utils.requests.get", return_value=_mock_response(png)):
        resp = client.post("/api/segment?image_url=http://example.com/img.png")

    assert resp.status_code == 200
    body = resp.json()
    assert "masks" in body
    assert body["masks"] is not None


def test_segment_no_image_returns_422(client):
    resp = client.post("/api/segment")
    assert resp.status_code == 422


def test_segment_bad_url_scheme_returns_422(client):
    resp = client.post("/api/segment?image_url=file:///etc/passwd")
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /api/segment/auto – image_url
# ---------------------------------------------------------------------------


def test_segment_auto_via_image_url(client):
    png = _make_png_bytes()

    with patch("app.utils.requests.get", return_value=_mock_response(png)):
        resp = client.post("/api/segment/auto?image_url=http://example.com/img.png")

    assert resp.status_code == 200
    body = resp.json()
    assert "segments" in body


def test_segment_auto_no_image_returns_422(client):
    resp = client.post("/api/segment/auto")
    assert resp.status_code == 422
