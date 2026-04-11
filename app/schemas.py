"""Pydantic schemas for cv-sam-service request and response models."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Health / Info
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str = Field("ok", description="Always 'ok' while the process is running")
    ready: bool = Field(..., description="True once the SAM model is fully loaded")
    loading: bool = Field(False, description="True while the SAM model is actively loading")
    model_variant: str = Field(..., description="SAM model variant in use (vit_b, vit_l, vit_h)")
    load_error: Optional[str] = Field(None, description="Error message if model loading failed")


class InfoResponse(BaseModel):
    ready: bool
    loading: bool = False
    model_variant: str
    device: Optional[str] = None
    device_name: Optional[str] = None
    vram_total_mb: Optional[float] = None
    vram_reserved_mb: Optional[float] = None
    vram_allocated_mb: Optional[float] = None
    load_error: Optional[str] = None


# ---------------------------------------------------------------------------
# Segment (prompted)
# ---------------------------------------------------------------------------


class SegmentResponse(BaseModel):
    masks: Optional[List[str]] = Field(
        None,
        description="Base64-encoded PNG masks, one per predicted mask",
    )
    polygons: Optional[List[List[List[float]]]] = Field(
        None,
        description=(
            "Polygon contours for each mask as [[x, y], …]. "
            "The largest contour is returned when a mask has multiple regions."
        ),
    )
    scores: List[float] = Field(..., description="Confidence score per mask")
    processing_time_ms: float = Field(..., description="Wall-clock time for prediction (ms)")


# ---------------------------------------------------------------------------
# Auto-segment
# ---------------------------------------------------------------------------


class AutoSegment(BaseModel):
    mask: Optional[str] = Field(None, description="Base64-encoded PNG mask")
    polygon: Optional[List[List[float]]] = Field(
        None, description="Largest contour as [[x, y], …]"
    )
    score: float = Field(..., description="Predicted IoU score")
    stability_score: float = Field(..., description="SAM stability score")
    area: int = Field(..., description="Mask area in pixels")
    bbox: List[float] = Field(..., description="[x, y, w, h] bounding box")


class AutoSegmentResponse(BaseModel):
    segments: List[AutoSegment]
    count: int = Field(..., description="Number of segments returned")
    processing_time_ms: float = Field(..., description="Wall-clock time for generation (ms)")


# ---------------------------------------------------------------------------
# Analyze
# ---------------------------------------------------------------------------


class ColorInfo(BaseModel):
    hex: str = Field(..., description="Hex color string, e.g. '#ff0000'")
    rgb: List[int] = Field(..., description="[R, G, B] integer values 0–255")
    frequency: float = Field(..., description="Fraction of sampled pixels closest to this color")


class HistogramStats(BaseModel):
    mean: List[float] = Field(..., description="Per-channel mean pixel value")
    std: List[float] = Field(..., description="Per-channel standard deviation")
    min: List[float] = Field(..., description="Per-channel minimum pixel value")
    max: List[float] = Field(..., description="Per-channel maximum pixel value")


class AnalyzeResponse(BaseModel):
    width: int = Field(..., description="Image width in pixels")
    height: int = Field(..., description="Image height in pixels")
    channels: int = Field(..., description="Number of color channels")
    format: Optional[str] = Field(None, description="Detected image format, e.g. 'PNG', 'JPEG'")
    dominant_colors: List[ColorInfo] = Field(..., description="Dominant colors sorted by frequency")
    edge_density: float = Field(..., description="Fraction of pixels detected as edges (Canny)")
    histogram_stats: HistogramStats


# ---------------------------------------------------------------------------
# Extract-palette
# ---------------------------------------------------------------------------


class PaletteColor(BaseModel):
    hex: str = Field(..., description="Hex color string, e.g. '#ff0000'")
    rgb: List[int] = Field(..., description="[R, G, B] integer values 0–255")
    weight: float = Field(..., description="Relative frequency of this color in the image")


class KulrsPalette(BaseModel):
    colors: List[str] = Field(..., description="Ordered list of hex color strings")


class ExtractPaletteResponse(BaseModel):
    colors: List[PaletteColor] = Field(..., description="Palette colors sorted by weight descending")
    kulrs: Optional[KulrsPalette] = Field(
        None, description="Kulrs-compatible palette (only present when kulrs_format=true)"
    )
