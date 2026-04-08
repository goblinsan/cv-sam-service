"""Tests for the local React UI serving (SPA fallback route)."""

import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(mock_engine, static_dir: str | None = None):
    """Return a TestClient with the engine mocked and _STATIC_DIR optionally overridden."""
    from app.main import app

    patches = [
        patch("app.routers.health.get_engine", return_value=mock_engine),
        patch("app.routers.segment.get_engine", return_value=mock_engine),
        patch("app.engine.get_engine", return_value=mock_engine),
    ]
    if static_dir is not None:
        patches.append(patch("app.main._STATIC_DIR", static_dir))

    ctx = __import__("contextlib").ExitStack()
    for p in patches:
        ctx.enter_context(p)

    return ctx, TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_spa_fallback_404_when_ui_not_built(client):
    """Root path returns 404 when index.html doesn't exist."""
    resp = client.get("/")
    assert resp.status_code == 404
    assert "not built" in resp.json()["detail"]


def test_spa_fallback_404_for_deep_path_when_ui_not_built(client):
    """Any non-API path returns 404 when index.html doesn't exist."""
    resp = client.get("/some/nested/route")
    assert resp.status_code == 404


def test_api_routes_unaffected_by_spa_fallback(client):
    """API routes are served normally regardless of UI build state."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_spa_fallback_serves_index_html_when_built(tmp_path, mock_engine):
    """Root path returns index.html when the UI has been built."""
    static_dir = tmp_path / "static"
    static_dir.mkdir()
    index = static_dir / "index.html"
    index.write_text("<html><body>CV SAM UI</body></html>")

    from app.main import app

    with (
        patch("app.routers.health.get_engine", return_value=mock_engine),
        patch("app.routers.segment.get_engine", return_value=mock_engine),
        patch("app.engine.get_engine", return_value=mock_engine),
        patch("app.main._STATIC_DIR", str(static_dir)),
    ):
        with TestClient(app, raise_server_exceptions=True) as c:
            resp = c.get("/")

    assert resp.status_code == 200
    assert "CV SAM UI" in resp.text


def test_spa_fallback_serves_index_for_deep_paths(tmp_path, mock_engine):
    """Deep SPA routes that don't map to files also return index.html."""
    static_dir = tmp_path / "static"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<html>SPA</html>")

    from app.main import app

    with (
        patch("app.routers.health.get_engine", return_value=mock_engine),
        patch("app.routers.segment.get_engine", return_value=mock_engine),
        patch("app.engine.get_engine", return_value=mock_engine),
        patch("app.main._STATIC_DIR", str(static_dir)),
    ):
        with TestClient(app, raise_server_exceptions=True) as c:
            resp = c.get("/dashboard/settings")

    assert resp.status_code == 200
    assert "SPA" in resp.text


def test_api_takes_priority_over_spa(tmp_path, mock_engine):
    """/api/* routes are handled by the API even when UI is built."""
    static_dir = tmp_path / "static"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<html>UI</html>")

    from app.main import app

    with (
        patch("app.routers.health.get_engine", return_value=mock_engine),
        patch("app.routers.segment.get_engine", return_value=mock_engine),
        patch("app.engine.get_engine", return_value=mock_engine),
        patch("app.main._STATIC_DIR", str(static_dir)),
    ):
        with TestClient(app, raise_server_exceptions=True) as c:
            resp = c.get("/api/health")

    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
