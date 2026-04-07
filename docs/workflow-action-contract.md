# CV SAM Service – Workflow Action Contract

> **Audience:** gateway-api workflow authors  
> **Status:** Stable (v1)  
> **Base URL:** `http://<cv-sam-service-host>:5201`

This document specifies the request/response contract for each CV SAM Service
endpoint so that gateway-api workflow steps can call them reliably.  All
endpoints accept images either as a **multipart file upload** or as a remote
**image URL** (see [Image Input Modes](#image-input-modes)).

---

## Table of Contents

1. [Service Readiness](#service-readiness)
2. [Image Input Modes](#image-input-modes)
3. [Actions](#actions)
   - [segment](#action-segment)
   - [segment/auto](#action-segmentauto)
   - [analyze](#action-analyze)
   - [extract-palette](#action-extract-palette)
4. [Common Error Shapes](#common-error-shapes)
5. [Downstream Step Patterns](#downstream-step-patterns)
6. [Follow-up Notes](#follow-up-notes)

---

## Service Readiness

Before sending inference requests, workflow steps should confirm the model is
loaded:

```
GET /api/health
```

**Response (always HTTP 200):**

```json
{
  "status": "ok",
  "ready": true,
  "model_variant": "vit_b",
  "load_error": null
}
```

| Field | Type | Notes |
|---|---|---|
| `status` | `"ok"` | Always `"ok"` while the process is alive |
| `ready` | bool | `false` during warm-up (~90 s on first start) |
| `model_variant` | string | `vit_b`, `vit_l`, or `vit_h` |
| `load_error` | string \| null | Non-null when model loading failed |

**Workflow guidance:** treat `ready == false` as a temporary warm-up state.
Retry with back-off up to `startPeriodSeconds` (120 s by default).  Only
raise an alert when `load_error` is non-null.

---

## Image Input Modes

Every inference endpoint supports two mutually exclusive input modes:

### Mode A – multipart file upload (preferred for large images)

```
POST /api/<action>
Content-Type: multipart/form-data

image=<binary image data>
```

### Mode B – image URL

```
POST /api/<action>?image_url=https://example.com/photo.jpg
```

Constraints for URL mode:
- Scheme must be `http` or `https` (no `ftp`, `file`, etc.)
- Maximum payload: **10 MB**
- Fetch timeout: **10 s**
- The service resolves the URL at request time; the workflow does not need to
  pre-fetch the image

---

## Actions

### Action: `segment`

Runs Meta SAM prompted segmentation.  **Requires the GPU model to be ready.**

```
POST /api/segment
POST /api/segment?image_url=<url>
```

#### Request fields

| Field | Source | Type | Required | Default | Description |
|---|---|---|---|---|---|
| `image` | form-file | binary | ✓ *or* `image_url` | — | Input image |
| `image_url` | query | string | ✓ *or* `image` | — | Public image URL |
| `point_coords` | form | JSON string | — | `null` | `[[x,y],…]` foreground/background points |
| `point_labels` | form | JSON string | — | `null` | `[1,0,…]` (1=fg, 0=bg) matching `point_coords` |
| `box` | form | JSON string | — | `null` | `[x1,y1,x2,y2]` bounding-box prompt |
| `multimask_output` | form | bool | — | `true` | Return 3 ranked candidate masks |
| `output_format` | query | string | — | `"masks"` | `masks` \| `polygons` \| `both` |

#### Response shape

```json
{
  "masks": ["<base64-PNG>", "<base64-PNG>", "<base64-PNG>"],
  "polygons": null,
  "scores": [0.95, 0.80, 0.70],
  "processing_time_ms": 312.4
}
```

| Field | Type | Notes |
|---|---|---|
| `masks` | list[string] \| null | Base64-encoded grayscale PNGs; null when `output_format` is `polygons` |
| `polygons` | list[list[[x,y]]] \| null | Contour point lists; null when `output_format` is `masks` |
| `scores` | list[float] | Confidence per mask (0–1) |
| `processing_time_ms` | float | Wall-clock prediction time |

#### Example – file upload with point prompt

```bash
curl -X POST http://localhost:5201/api/segment \
  -F "image=@/path/to/photo.jpg" \
  -F 'point_coords=[[320,240]]' \
  -F 'point_labels=[1]'
```

#### Example – image URL with bounding-box prompt

```bash
curl -X POST \
  "http://localhost:5201/api/segment?image_url=https://example.com/photo.jpg" \
  -F 'box=[100,80,540,420]'
```

---

### Action: `segment/auto`

Runs SAM automatic mask generation with no prompt required.  **Requires the
GPU model to be ready.**

```
POST /api/segment/auto
POST /api/segment/auto?image_url=<url>
```

#### Request fields

| Field | Source | Type | Required | Default | Description |
|---|---|---|---|---|---|
| `image` | form-file | binary | ✓ *or* `image_url` | — | Input image |
| `image_url` | query | string | ✓ *or* `image` | — | Public image URL |
| `max_masks` | query | int (1–1000) | — | `50` | Max segments returned, sorted by score |
| `output_format` | query | string | — | `"masks"` | `masks` \| `polygons` \| `both` |

#### Response shape

```json
{
  "segments": [
    {
      "mask": "<base64-PNG>",
      "polygon": null,
      "score": 0.92,
      "stability_score": 0.97,
      "area": 14823,
      "bbox": [42.0, 18.0, 310.0, 275.0]
    }
  ],
  "count": 1,
  "processing_time_ms": 1840.5
}
```

| Field | Type | Notes |
|---|---|---|
| `segments[].mask` | string \| null | Base64 PNG per segment |
| `segments[].polygon` | list[[x,y]] \| null | Largest contour |
| `segments[].score` | float | Predicted IoU (0–1) |
| `segments[].stability_score` | float | SAM stability score (0–1) |
| `segments[].area` | int | Mask area in pixels |
| `segments[].bbox` | [x,y,w,h] | Bounding box |
| `count` | int | `len(segments)` |
| `processing_time_ms` | float | Wall-clock time |

---

### Action: `analyze`

Analyzes an image with OpenCV (no GPU required).

```
POST /api/analyze
POST /api/analyze?image_url=<url>
```

#### Request fields

| Field | Source | Type | Required | Default | Description |
|---|---|---|---|---|---|
| `image` | form-file | binary | ✓ *or* `image_url` | — | Input image |
| `image_url` | query | string | ✓ *or* `image` | — | Public image URL |
| `num_colors` | query | int (1–20) | — | `5` | Number of dominant colors to extract |

#### Response shape

```json
{
  "width": 1920,
  "height": 1080,
  "channels": 3,
  "format": "JPEG",
  "dominant_colors": [
    { "hex": "#3a5f8c", "rgb": [58, 95, 140], "frequency": 0.3421 },
    { "hex": "#e8c47a", "rgb": [232, 196, 122], "frequency": 0.2187 }
  ],
  "edge_density": 0.0842,
  "histogram_stats": {
    "mean": [112.3, 98.7, 87.1],
    "std": [54.2, 49.8, 44.3],
    "min": [0.0, 0.0, 0.0],
    "max": [255.0, 255.0, 255.0]
  }
}
```

| Field | Type | Notes |
|---|---|---|
| `width` / `height` | int | Pixel dimensions |
| `channels` | int | Color channels (3 for RGB) |
| `format` | string \| null | PIL-detected format (`JPEG`, `PNG`, …) |
| `dominant_colors[].hex` | string | 7-char hex, e.g. `#3a5f8c` |
| `dominant_colors[].rgb` | [R,G,B] | Integer values 0–255 |
| `dominant_colors[].frequency` | float | Fraction of sampled pixels closest to this color |
| `edge_density` | float | Fraction of pixels detected as edges (Canny) |
| `histogram_stats` | object | Per-channel mean/std/min/max |

---

### Action: `extract-palette`

Extracts a color palette using k-means (no GPU required).

```
POST /api/extract-palette
POST /api/extract-palette?image_url=<url>
```

#### Request fields

| Field | Source | Type | Required | Default | Description |
|---|---|---|---|---|---|
| `image` | form-file | binary | ✓ *or* `image_url` | — | Input image |
| `image_url` | query | string | ✓ *or* `image` | — | Public image URL |
| `num_colors` | query | int (1–32) | — | `6` | Number of palette colors |
| `kulrs_format` | query | bool | — | `false` | Include Kulrs-compatible palette |

#### Response shape

```json
{
  "colors": [
    { "hex": "#3a5f8c", "rgb": [58, 95, 140], "weight": 0.3421 },
    { "hex": "#e8c47a", "rgb": [232, 196, 122], "weight": 0.2187 }
  ],
  "kulrs": {
    "colors": ["#3a5f8c", "#e8c47a"]
  }
}
```

| Field | Type | Notes |
|---|---|---|
| `colors[].hex` | string | 7-char hex |
| `colors[].rgb` | [R,G,B] | 0–255 per channel |
| `colors[].weight` | float | Relative frequency (sums to ~1) |
| `kulrs` | object \| null | Present only when `kulrs_format=true` |
| `kulrs.colors` | list[string] | Ordered hex list matching `colors` |

---

## Common Error Shapes

All errors return a JSON body:

```json
{ "detail": "<human-readable message>" }
```

| HTTP status | Meaning |
|---|---|
| `422` | Validation error (bad params, invalid JSON, unsupported URL scheme, image too large, fetch failure, private host blocked) |
| `503` | SAM model not yet ready; retry with back-off |

---

## Downstream Step Patterns

### Pattern 1 – Palette → style-generation step

```
extract-palette response → colors[].hex → style-generator input
```

Workflow step A calls `POST /api/extract-palette?num_colors=8` on a product
image.  Step B receives `colors` and formats them into a prompt for a
downstream text-to-image or CSS-generation step.

### Pattern 2 – Segment → crop → further analysis

```
segment response → polygons[0] → bounding rect → POST /api/analyze on crop
```

Step A calls `POST /api/segment` with a bounding-box prompt.  The resulting
`polygons[0]` is used to compute a tight crop rectangle.  Step B calls
`POST /api/analyze` on the cropped region for focused color analysis.

### Pattern 3 – URL-based batch processing

When the workflow holds image URLs (e.g., from an asset store):

```
for url in asset_urls:
    POST /api/analyze?image_url={url}&num_colors=5
```

No pre-fetch step is needed; the CV service resolves the URL internally.

---

## Follow-up Notes

- **Authentication:** No auth is enforced today.  Future versions may add
  `Authorization: Bearer <token>` header validation.  Workflow steps should
  be written to pass an optional `auth_token` variable through to an
  `Authorization` header so the change is backward-compatible.
- **Concurrent request limit:** Configure the gateway to cap concurrent
  inference calls to `cv-sam-service` at **2** (segment endpoints) to stay
  within VRAM budget.  Analyze and extract-palette are CPU-only and can be
  called more freely.
- **Max image size via URL:** 10 MB.  For larger images, use the file-upload
  mode or pre-resize before calling the service.
- **Readiness timeout:** The SAM model takes ~90 s to load on first start.
  Workflow orchestrators should set `startPeriodSeconds ≥ 120` on the health
  check (already set in `docs/workload-manifest.example.yml`).
