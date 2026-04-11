"""POST /api/segment and POST /api/segment/auto endpoints."""

import base64
import io
import json
import time
from typing import List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from PIL import Image

from app.engine import get_engine
from app.schemas import AutoSegment, AutoSegmentResponse, SegmentResponse
from app.utils import resolve_image_bytes

router = APIRouter(tags=["segment"])


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _load_image(data: bytes) -> np.ndarray:
    """Decode uploaded image bytes to an RGB numpy array."""
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return np.array(img)


def _mask_to_base64(mask: np.ndarray) -> str:
    """Convert a boolean (H, W) mask to a base64-encoded grayscale PNG."""
    uint8_mask = mask.astype(np.uint8) * 255
    img = Image.fromarray(uint8_mask, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _mask_to_polygon(mask: np.ndarray) -> Optional[List[List[float]]]:
    """
    Return the largest contour of a boolean mask as a list of [x, y] points.
    Returns ``None`` when no contour is found.
    """
    uint8_mask = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        uint8_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 3:
        return None
    return largest.squeeze(axis=1).tolist()


# ---------------------------------------------------------------------------
# POST /api/segment
# ---------------------------------------------------------------------------


@router.post("/segment", response_model=SegmentResponse)
async def segment(
    image: Optional[UploadFile] = File(None, description="Input image (any common raster format)"),
    image_url: Optional[str] = Query(
        None,
        description="Public URL of the image to segment (http/https, max 10 MB). "
        "Supply either this or the 'image' file upload, not both.",
    ),
    point_coords: Optional[str] = Form(
        None,
        description="JSON array of [x, y] foreground/background point prompts, e.g. [[320,240]]",
    ),
    point_labels: Optional[str] = Form(
        None,
        description="JSON array of labels matching point_coords (1=foreground, 0=background)",
    ),
    box: Optional[str] = Form(
        None,
        description="JSON array [x1, y1, x2, y2] bounding-box prompt",
    ),
    multimask_output: bool = Form(
        True,
        description="When True SAM returns 3 candidate masks ranked by score",
    ),
    output_format: str = Query(
        "masks",
        description="Output format: 'masks' (base64 PNG), 'polygons', or 'both'",
    ),
) -> SegmentResponse:
    """
    Run prompted SAM segmentation on an uploaded image.

    At least one prompt (``point_coords`` or ``box``) should be supplied for
    meaningful results; omitting all prompts asks SAM to segment the whole image.

    Provide the image either as a multipart file upload (``image``) or as a
    publicly reachable URL (``image_url`` query parameter).
    """
    engine = get_engine()
    try:
        engine.ensure_ready()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if output_format not in ("masks", "polygons", "both"):
        raise HTTPException(
            status_code=422,
            detail="output_format must be 'masks', 'polygons', or 'both'",
        )

    # --- parse prompts --------------------------------------------------
    np_point_coords: Optional[np.ndarray] = None
    np_point_labels: Optional[np.ndarray] = None
    np_box: Optional[np.ndarray] = None

    if point_coords is not None:
        coords = json.loads(point_coords)
        np_point_coords = np.array(coords, dtype=np.float32)
    if point_labels is not None:
        labels = json.loads(point_labels)
        np_point_labels = np.array(labels, dtype=np.int32)
    if box is not None:
        np_box = np.array(json.loads(box), dtype=np.float32)

    # --- resolve image --------------------------------------------------
    upload_bytes = (await image.read()) if image is not None else None
    image_bytes = resolve_image_bytes(upload_bytes, image_url)
    image_array = _load_image(image_bytes)

    t0 = time.perf_counter()
    masks, scores, _ = engine.predict(
        image_array,
        point_coords=np_point_coords,
        point_labels=np_point_labels,
        box=np_box,
        multimask_output=multimask_output,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # --- build response -------------------------------------------------
    encoded_masks: Optional[List[str]] = None
    polygons: Optional[List[List[List[float]]]] = None

    if output_format in ("masks", "both"):
        encoded_masks = [_mask_to_base64(m) for m in masks]
    if output_format in ("polygons", "both"):
        polygons = [_mask_to_polygon(m) or [] for m in masks]

    return SegmentResponse(
        masks=encoded_masks,
        polygons=polygons,
        scores=scores.tolist(),
        processing_time_ms=round(elapsed_ms, 2),
    )


# ---------------------------------------------------------------------------
# POST /api/segment/auto
# ---------------------------------------------------------------------------


@router.post("/segment/auto", response_model=AutoSegmentResponse)
async def segment_auto(
    image: Optional[UploadFile] = File(None, description="Input image (any common raster format)"),
    image_url: Optional[str] = Query(
        None,
        description="Public URL of the image to segment (http/https, max 10 MB). "
        "Supply either this or the 'image' file upload, not both.",
    ),
    max_masks: int = Query(
        50,
        ge=1,
        le=1000,
        description="Maximum number of segments to return (sorted by score, descending)",
    ),
    output_format: str = Query(
        "masks",
        description="Output format per segment: 'masks' (base64 PNG), 'polygons', or 'both'",
    ),
) -> AutoSegmentResponse:
    """
    Run SAM automatic mask generation on an uploaded image.

    Returns up to ``max_masks`` detected segments, sorted by predicted IoU
    score (highest first).

    Provide the image either as a multipart file upload (``image``) or as a
    publicly reachable URL (``image_url`` query parameter).
    """
    engine = get_engine()
    try:
        engine.ensure_ready()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if output_format not in ("masks", "polygons", "both"):
        raise HTTPException(
            status_code=422,
            detail="output_format must be 'masks', 'polygons', or 'both'",
        )

    upload_bytes = (await image.read()) if image is not None else None
    image_bytes = resolve_image_bytes(upload_bytes, image_url)
    image_array = _load_image(image_bytes)

    t0 = time.perf_counter()
    raw_masks = engine.predict_auto(image_array)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Sort by predicted IoU descending, then cap
    raw_masks.sort(key=lambda m: m.get("predicted_iou", 0.0), reverse=True)
    raw_masks = raw_masks[:max_masks]

    segments: List[AutoSegment] = []
    for m in raw_masks:
        seg_mask: np.ndarray = m["segmentation"]
        encoded: Optional[str] = None
        polygon: Optional[List[List[float]]] = None

        if output_format in ("masks", "both"):
            encoded = _mask_to_base64(seg_mask)
        if output_format in ("polygons", "both"):
            polygon = _mask_to_polygon(seg_mask)

        segments.append(
            AutoSegment(
                mask=encoded,
                polygon=polygon,
                score=float(m.get("predicted_iou", 0.0)),
                stability_score=float(m.get("stability_score", 0.0)),
                area=int(m["area"]),
                bbox=[float(v) for v in m["bbox"]],
            )
        )

    return AutoSegmentResponse(
        segments=segments,
        count=len(segments),
        processing_time_ms=round(elapsed_ms, 2),
    )
