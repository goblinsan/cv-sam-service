"""Tests for POST /api/analyze, POST /api/transform, POST /api/extract-palette."""

import io
import json

import numpy as np
import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Extra fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def colorful_image_bytes() -> bytes:
    """64×64 image with four distinct-colour quadrants: red, green, blue, white."""
    img_array = np.zeros((64, 64, 3), dtype=np.uint8)
    img_array[:32, :32] = [255, 0, 0]    # red     – top-left
    img_array[:32, 32:] = [0, 255, 0]    # green   – top-right
    img_array[32:, :32] = [0, 0, 255]    # blue    – bottom-left
    img_array[32:, 32:] = [255, 255, 255]  # white – bottom-right
    img = Image.fromarray(img_array, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# POST /api/analyze
# ---------------------------------------------------------------------------


def test_analyze_response_structure(client, colorful_image_bytes):
    resp = client.post(
        "/api/analyze",
        files={"image": ("test.png", colorful_image_bytes, "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()

    assert body["width"] == 64
    assert body["height"] == 64
    assert body["channels"] == 3
    assert body["format"] == "PNG"

    # dominant_colors
    assert "dominant_colors" in body
    assert len(body["dominant_colors"]) == 5  # default num_colors
    for color in body["dominant_colors"]:
        assert color["hex"].startswith("#")
        assert len(color["hex"]) == 7
        assert len(color["rgb"]) == 3
        assert 0.0 <= color["frequency"] <= 1.0

    # frequencies should sum to ~1
    total_freq = sum(c["frequency"] for c in body["dominant_colors"])
    assert abs(total_freq - 1.0) < 0.05

    # edge_density
    assert "edge_density" in body
    assert 0.0 <= body["edge_density"] <= 1.0

    # histogram_stats
    hs = body["histogram_stats"]
    for key in ("mean", "std", "min", "max"):
        assert key in hs
        assert len(hs[key]) == 3  # three channels


def test_analyze_custom_num_colors(client, colorful_image_bytes):
    resp = client.post(
        "/api/analyze?num_colors=4",
        files={"image": ("test.png", colorful_image_bytes, "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["dominant_colors"]) == 4


def test_analyze_sorted_by_frequency(client, colorful_image_bytes):
    """Colours must be returned in descending frequency order."""
    resp = client.post(
        "/api/analyze?num_colors=4",
        files={"image": ("test.png", colorful_image_bytes, "image/png")},
    )
    assert resp.status_code == 200
    freqs = [c["frequency"] for c in resp.json()["dominant_colors"]]
    assert freqs == sorted(freqs, reverse=True)


def test_analyze_edge_density_black_image(client, test_image_bytes):
    """A uniform black image should have zero (or near-zero) edge density."""
    resp = client.post(
        "/api/analyze",
        files={"image": ("black.png", test_image_bytes, "image/png")},
    )
    assert resp.status_code == 200
    assert resp.json()["edge_density"] == pytest.approx(0.0, abs=0.01)


def test_analyze_histogram_black_image(client, test_image_bytes):
    """A pure-black image should have mean=0, std=0, min=0, max=0 per channel."""
    resp = client.post(
        "/api/analyze",
        files={"image": ("black.png", test_image_bytes, "image/png")},
    )
    assert resp.status_code == 200
    hs = resp.json()["histogram_stats"]
    for ch_val in hs["mean"]:
        assert ch_val == pytest.approx(0.0)
    for ch_val in hs["max"]:
        assert ch_val == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# POST /api/transform
# ---------------------------------------------------------------------------


def test_transform_no_ops_returns_image(client, colorful_image_bytes):
    """Empty operations list should return the original image unchanged."""
    resp = client.post(
        "/api/transform",
        files={"image": ("test.png", colorful_image_bytes, "image/png")},
        data={"operations": "[]", "output_format": "PNG"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    result = Image.open(io.BytesIO(resp.content))
    assert result.size == (64, 64)


def test_transform_resize(client, colorful_image_bytes):
    ops = json.dumps([{"op": "resize", "width": 32, "height": 16}])
    resp = client.post(
        "/api/transform",
        files={"image": ("test.png", colorful_image_bytes, "image/png")},
        data={"operations": ops},
    )
    assert resp.status_code == 200
    result = Image.open(io.BytesIO(resp.content))
    assert result.size == (32, 16)


def test_transform_crop(client, colorful_image_bytes):
    ops = json.dumps([{"op": "crop", "x": 0, "y": 0, "width": 32, "height": 32}])
    resp = client.post(
        "/api/transform",
        files={"image": ("test.png", colorful_image_bytes, "image/png")},
        data={"operations": ops},
    )
    assert resp.status_code == 200
    result = Image.open(io.BytesIO(resp.content))
    assert result.size == (32, 32)


def test_transform_crop_out_of_bounds(client, colorful_image_bytes):
    ops = json.dumps([{"op": "crop", "x": 50, "y": 50, "width": 64, "height": 64}])
    resp = client.post(
        "/api/transform",
        files={"image": ("test.png", colorful_image_bytes, "image/png")},
        data={"operations": ops},
    )
    assert resp.status_code == 422


def test_transform_rotate(client, colorful_image_bytes):
    ops = json.dumps([{"op": "rotate", "angle": 90}])
    resp = client.post(
        "/api/transform",
        files={"image": ("test.png", colorful_image_bytes, "image/png")},
        data={"operations": ops},
    )
    assert resp.status_code == 200
    result = Image.open(io.BytesIO(resp.content))
    # Square image – dimensions unchanged after 90° rotation
    assert result.size == (64, 64)


def test_transform_blur(client, colorful_image_bytes):
    ops = json.dumps([{"op": "blur", "kernel_size": 5}])
    resp = client.post(
        "/api/transform",
        files={"image": ("test.png", colorful_image_bytes, "image/png")},
        data={"operations": ops},
    )
    assert resp.status_code == 200
    result = Image.open(io.BytesIO(resp.content))
    assert result.size == (64, 64)


def test_transform_blur_even_kernel_auto_corrected(client, colorful_image_bytes):
    """Even kernel_size should be auto-incremented to nearest odd value."""
    ops = json.dumps([{"op": "blur", "kernel_size": 4}])
    resp = client.post(
        "/api/transform",
        files={"image": ("test.png", colorful_image_bytes, "image/png")},
        data={"operations": ops},
    )
    assert resp.status_code == 200


def test_transform_sharpen(client, colorful_image_bytes):
    ops = json.dumps([{"op": "sharpen"}])
    resp = client.post(
        "/api/transform",
        files={"image": ("test.png", colorful_image_bytes, "image/png")},
        data={"operations": ops},
    )
    assert resp.status_code == 200
    result = Image.open(io.BytesIO(resp.content))
    assert result.size == (64, 64)


def test_transform_edge_detect(client, colorful_image_bytes):
    ops = json.dumps([{"op": "edge-detect"}])
    resp = client.post(
        "/api/transform",
        files={"image": ("test.png", colorful_image_bytes, "image/png")},
        data={"operations": ops},
    )
    assert resp.status_code == 200
    result = Image.open(io.BytesIO(resp.content))
    assert result.size == (64, 64)


def test_transform_chained_operations(client, colorful_image_bytes):
    ops = json.dumps(
        [
            {"op": "resize", "width": 48, "height": 48},
            {"op": "rotate", "angle": 45},
            {"op": "blur", "kernel_size": 3},
        ]
    )
    resp = client.post(
        "/api/transform",
        files={"image": ("test.png", colorful_image_bytes, "image/png")},
        data={"operations": ops},
    )
    assert resp.status_code == 200
    result = Image.open(io.BytesIO(resp.content))
    assert result.size == (48, 48)


def test_transform_jpeg_output(client, colorful_image_bytes):
    resp = client.post(
        "/api/transform",
        files={"image": ("test.png", colorful_image_bytes, "image/png")},
        data={"operations": "[]", "output_format": "JPEG"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"


def test_transform_webp_output(client, colorful_image_bytes):
    resp = client.post(
        "/api/transform",
        files={"image": ("test.png", colorful_image_bytes, "image/png")},
        data={"operations": "[]", "output_format": "WEBP"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/webp"


def test_transform_invalid_output_format(client, colorful_image_bytes):
    resp = client.post(
        "/api/transform",
        files={"image": ("test.png", colorful_image_bytes, "image/png")},
        data={"operations": "[]", "output_format": "BMP"},
    )
    assert resp.status_code == 422


def test_transform_unknown_operation(client, colorful_image_bytes):
    ops = json.dumps([{"op": "flip"}])
    resp = client.post(
        "/api/transform",
        files={"image": ("test.png", colorful_image_bytes, "image/png")},
        data={"operations": ops},
    )
    assert resp.status_code == 422


def test_transform_invalid_json_operations(client, colorful_image_bytes):
    resp = client.post(
        "/api/transform",
        files={"image": ("test.png", colorful_image_bytes, "image/png")},
        data={"operations": "not-json"},
    )
    assert resp.status_code == 422


def test_transform_resize_invalid_dimensions(client, colorful_image_bytes):
    ops = json.dumps([{"op": "resize", "width": 0, "height": 64}])
    resp = client.post(
        "/api/transform",
        files={"image": ("test.png", colorful_image_bytes, "image/png")},
        data={"operations": ops},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /api/extract-palette
# ---------------------------------------------------------------------------


def test_extract_palette_default(client, colorful_image_bytes):
    resp = client.post(
        "/api/extract-palette",
        files={"image": ("test.png", colorful_image_bytes, "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()

    assert "colors" in body
    assert len(body["colors"]) == 6  # default num_colors
    assert body["kulrs"] is None  # not requested

    for color in body["colors"]:
        assert color["hex"].startswith("#")
        assert len(color["hex"]) == 7
        assert len(color["rgb"]) == 3
        assert 0.0 <= color["weight"] <= 1.0


def test_extract_palette_sorted_by_weight(client, colorful_image_bytes):
    resp = client.post(
        "/api/extract-palette?num_colors=4",
        files={"image": ("test.png", colorful_image_bytes, "image/png")},
    )
    assert resp.status_code == 200
    weights = [c["weight"] for c in resp.json()["colors"]]
    assert weights == sorted(weights, reverse=True)


def test_extract_palette_custom_num_colors(client, colorful_image_bytes):
    resp = client.post(
        "/api/extract-palette?num_colors=3",
        files={"image": ("test.png", colorful_image_bytes, "image/png")},
    )
    assert resp.status_code == 200
    assert len(resp.json()["colors"]) == 3


def test_extract_palette_kulrs_format(client, colorful_image_bytes):
    resp = client.post(
        "/api/extract-palette?num_colors=4&kulrs_format=true",
        files={"image": ("test.png", colorful_image_bytes, "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["kulrs"] is not None
    assert "colors" in body["kulrs"]
    assert len(body["kulrs"]["colors"]) == 4
    for hex_color in body["kulrs"]["colors"]:
        assert hex_color.startswith("#")
        assert len(hex_color) == 7


def test_extract_palette_kulrs_matches_palette(client, colorful_image_bytes):
    """Kulrs hex list must match the order of the main palette."""
    resp = client.post(
        "/api/extract-palette?num_colors=4&kulrs_format=true",
        files={"image": ("test.png", colorful_image_bytes, "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()
    palette_hexes = [c["hex"] for c in body["colors"]]
    assert body["kulrs"]["colors"] == palette_hexes
