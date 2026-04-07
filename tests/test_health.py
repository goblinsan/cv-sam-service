"""Tests for GET /api/health and GET /api/info."""

from unittest.mock import MagicMock, patch


def test_health_returns_200_when_ready(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["ready"] is True
    assert body["model_variant"] == "vit_b"
    assert body["load_error"] is None


def test_health_returns_200_when_not_ready():
    """Health must return HTTP 200 even before the model finishes loading."""
    from app.main import app

    not_ready_engine = MagicMock()
    not_ready_engine.ready = False
    not_ready_engine.model_variant = "vit_b"
    not_ready_engine.load_error = None
    not_ready_engine.vram_info.return_value = {
        "device_name": "cpu",
        "vram_total_mb": None,
        "vram_reserved_mb": None,
        "vram_allocated_mb": None,
    }

    with (
        patch("app.routers.health.get_engine", return_value=not_ready_engine),
        patch("app.routers.segment.get_engine", return_value=not_ready_engine),
        patch("app.engine.get_engine", return_value=not_ready_engine),
    ):
        from fastapi.testclient import TestClient

        with TestClient(app) as c:
            resp = c.get("/api/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["ready"] is False


def test_health_reports_load_error():
    from app.main import app

    error_engine = MagicMock()
    error_engine.ready = False
    error_engine.model_variant = "vit_h"
    error_engine.load_error = "CUDA out of memory"
    error_engine.vram_info.return_value = {
        "device_name": "cpu",
        "vram_total_mb": None,
        "vram_reserved_mb": None,
        "vram_allocated_mb": None,
    }

    with (
        patch("app.routers.health.get_engine", return_value=error_engine),
        patch("app.routers.segment.get_engine", return_value=error_engine),
        patch("app.engine.get_engine", return_value=error_engine),
    ):
        from fastapi.testclient import TestClient

        with TestClient(app) as c:
            resp = c.get("/api/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["load_error"] == "CUDA out of memory"


def test_info_returns_cpu_metrics(client):
    resp = client.get("/api/info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["device"] == "cpu"
    assert body["device_name"] == "cpu"
    assert body["vram_total_mb"] is None
    assert body["ready"] is True
    assert body["model_variant"] == "vit_b"
