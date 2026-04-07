"""POST /api/analyze, POST /api/transform, POST /api/extract-palette endpoints."""

import io
import json
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response
from PIL import Image

from app.schemas import (
    AnalyzeResponse,
    ColorInfo,
    ExtractPaletteResponse,
    HistogramStats,
    KulrsPalette,
    PaletteColor,
)

router = APIRouter(tags=["cv"])

_OUTPUT_CONTENT_TYPES: Dict[str, str] = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "WEBP": "image/webp",
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_image_with_format(data: bytes):
    """Decode uploaded image bytes to an RGB numpy array; also return PIL format string."""
    buf = io.BytesIO(data)
    pil_img = Image.open(buf)
    fmt = pil_img.format  # e.g. "PNG", "JPEG", None
    return np.array(pil_img.convert("RGB")), fmt


def _kmeans_colors(image_rgb: np.ndarray, k: int) -> List[Dict[str, Any]]:
    """Run k-means on image pixels and return dominant colour info sorted by frequency."""
    pixels = image_rgb.reshape(-1, 3).astype(np.float32)

    # Subsample for speed on large images
    max_samples = 10_000
    if len(pixels) > max_samples:
        rng = np.random.default_rng(seed=0)
        idx = rng.choice(len(pixels), max_samples, replace=False)
        pixels = pixels[idx]

    # k cannot exceed the number of sample pixels
    k = min(k, len(pixels))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    labels = labels.flatten()
    freq = np.bincount(labels, minlength=k) / len(labels)

    results: List[Dict[str, Any]] = []
    for i, center in enumerate(centers):
        r, g, b = int(round(center[0])), int(round(center[1])), int(round(center[2]))
        results.append(
            {
                "hex": f"#{r:02x}{g:02x}{b:02x}",
                "rgb": [r, g, b],
                "frequency": round(float(freq[i]), 4),
            }
        )

    results.sort(key=lambda c: c["frequency"], reverse=True)
    return results


def _edge_density(image_rgb: np.ndarray) -> float:
    """Return the fraction of pixels detected as edges via Canny."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return float(np.count_nonzero(edges)) / float(edges.size)


def _histogram_stats(image_rgb: np.ndarray) -> Dict[str, List[float]]:
    """Per-channel mean, std, min, max."""
    stats: Dict[str, List[float]] = {"mean": [], "std": [], "min": [], "max": []}
    for c in range(image_rgb.shape[2]):
        ch = image_rgb[:, :, c].astype(np.float64)
        stats["mean"].append(round(float(ch.mean()), 3))
        stats["std"].append(round(float(ch.std()), 3))
        stats["min"].append(round(float(ch.min()), 3))
        stats["max"].append(round(float(ch.max()), 3))
    return stats


def _apply_operation(image_rgb: np.ndarray, op: Dict[str, Any]) -> np.ndarray:
    """Apply a single named transform to an RGB image and return the result."""
    name = op.get("op", "")

    if name == "resize":
        w, h = int(op["width"]), int(op["height"])
        if w <= 0 or h <= 0:
            raise HTTPException(status_code=422, detail="resize: width and height must be > 0")
        return cv2.resize(image_rgb, (w, h), interpolation=cv2.INTER_LANCZOS4)

    if name == "crop":
        x, y = int(op["x"]), int(op["y"])
        cw, ch = int(op["width"]), int(op["height"])
        ih, iw = image_rgb.shape[:2]
        if x < 0 or y < 0 or x + cw > iw or y + ch > ih:
            raise HTTPException(status_code=422, detail="crop: region is out of bounds")
        return image_rgb[y : y + ch, x : x + cw]

    if name == "rotate":
        angle = float(op["angle"])
        ih, iw = image_rgb.shape[:2]
        M = cv2.getRotationMatrix2D((iw / 2, ih / 2), -angle, 1.0)
        return cv2.warpAffine(
            image_rgb, M, (iw, ih), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )

    if name == "blur":
        ks = int(op.get("kernel_size", 5))
        if ks < 1:
            raise HTTPException(status_code=422, detail="blur: kernel_size must be >= 1")
        if ks % 2 == 0:
            ks += 1  # GaussianBlur requires an odd kernel size
        return cv2.GaussianBlur(image_rgb, (ks, ks), 0)

    if name == "sharpen":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(image_rgb, -1, kernel)

    if name == "edge-detect":
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    raise HTTPException(status_code=422, detail=f"Unknown operation: {name!r}")


def _encode_image(image_rgb: np.ndarray, fmt: str) -> bytes:
    """Encode an RGB numpy array to the given format and return raw bytes."""
    pil_img = Image.fromarray(image_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# POST /api/analyze
# ---------------------------------------------------------------------------


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    image: UploadFile = File(..., description="Input image (any common raster format)"),
    num_colors: int = Query(
        5, ge=1, le=20, description="Number of dominant colors to extract (k-means k)"
    ),
) -> AnalyzeResponse:
    """
    Analyze an uploaded image with pure OpenCV – no GPU required.

    Returns dominant colors (k-means), Canny edge density, per-channel
    histogram statistics, pixel dimensions, and detected image format.
    """
    data = await image.read()
    image_rgb, fmt = _load_image_with_format(data)
    h, w = image_rgb.shape[:2]
    channels = image_rgb.shape[2] if image_rgb.ndim == 3 else 1

    raw_colors = _kmeans_colors(image_rgb, k=num_colors)
    edge_dens = _edge_density(image_rgb)
    hist = _histogram_stats(image_rgb)

    return AnalyzeResponse(
        width=w,
        height=h,
        channels=channels,
        format=fmt,
        dominant_colors=[ColorInfo(**c) for c in raw_colors],
        edge_density=round(edge_dens, 6),
        histogram_stats=HistogramStats(**hist),
    )


# ---------------------------------------------------------------------------
# POST /api/transform
# ---------------------------------------------------------------------------


@router.post("/transform")
async def transform(
    image: UploadFile = File(..., description="Input image (any common raster format)"),
    operations: str = Form(
        "[]",
        description=(
            "JSON array of operation objects applied in order. "
            "Supported ops: resize {width, height}, crop {x, y, width, height}, "
            "rotate {angle}, blur {kernel_size}, sharpen, edge-detect."
        ),
    ),
    output_format: str = Form(
        "PNG",
        description="Output image format: PNG, JPEG, or WEBP",
    ),
) -> Response:
    """
    Apply a sequence of image-processing operations and return the result.

    Operations are applied in the order listed.  The transformed image is
    returned as binary data with the appropriate ``Content-Type`` header.
    """
    fmt_upper = output_format.upper()
    if fmt_upper not in _OUTPUT_CONTENT_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"output_format must be one of: {', '.join(_OUTPUT_CONTENT_TYPES)}",
        )

    try:
        ops: List[Dict[str, Any]] = json.loads(operations)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid JSON in operations: {exc}") from exc

    if not isinstance(ops, list):
        raise HTTPException(status_code=422, detail="operations must be a JSON array")

    data = await image.read()
    image_rgb, _ = _load_image_with_format(data)

    for op in ops:
        if not isinstance(op, dict):
            raise HTTPException(status_code=422, detail="Each operation must be a JSON object")
        image_rgb = _apply_operation(image_rgb, op)

    output_bytes = _encode_image(image_rgb, fmt_upper)
    return Response(content=output_bytes, media_type=_OUTPUT_CONTENT_TYPES[fmt_upper])


# ---------------------------------------------------------------------------
# POST /api/extract-palette
# ---------------------------------------------------------------------------


@router.post("/extract-palette", response_model=ExtractPaletteResponse)
async def extract_palette(
    image: UploadFile = File(..., description="Input image (any common raster format)"),
    num_colors: int = Query(
        6, ge=1, le=32, description="Number of palette colors to extract"
    ),
    kulrs_format: bool = Query(
        False, description="When true, include a Kulrs-compatible palette in the response"
    ),
) -> ExtractPaletteResponse:
    """
    Extract a color palette from an image using k-means clustering.

    Returns hex values, RGB triplets, and frequency weights for each color.
    Pass ``kulrs_format=true`` to also receive a Kulrs-compatible palette object.
    """
    data = await image.read()
    image_rgb, _ = _load_image_with_format(data)

    raw_colors = _kmeans_colors(image_rgb, k=num_colors)

    palette = [
        PaletteColor(hex=c["hex"], rgb=c["rgb"], weight=c["frequency"])
        for c in raw_colors
    ]

    kulrs: Optional[KulrsPalette] = None
    if kulrs_format:
        kulrs = KulrsPalette(colors=[c.hex for c in palette])

    return ExtractPaletteResponse(colors=palette, kulrs=kulrs)
