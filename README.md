# cv-sam-service

FastAPI service that loads [Meta's Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) on an NVIDIA GPU and exposes REST segmentation endpoints.

## Quick start (Docker Compose)

```bash
# Uses vit_b by default; weights are downloaded to /data/models/sam on first run.
docker compose up --build
```

The service is published on port **5201** (override with `HOST_PORT`).

```bash
# Health check
curl http://localhost:5201/api/health

# Interactive API docs
open http://localhost:5201/docs
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `CV_SAM_VARIANT` | `vit_b` | SAM model variant: `vit_b` (~1.5 GB VRAM), `vit_l`, or `vit_h` (~2.5 GB VRAM) |
| `MODEL_DIR` | `/data/models/sam` | Directory where SAM checkpoint is cached |
| `HOST_PORT` | `5201` | Host port mapped in `docker-compose.yml` |

## API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Liveness + readiness probe (always HTTP 200) |
| `GET` | `/api/info` | GPU info, VRAM usage, model variant, readiness |
| `POST` | `/api/segment` | Prompted segmentation (point / box prompts) |
| `POST` | `/api/segment/auto` | Automatic mask generation (no prompts required) |

### POST /api/segment

Accepts `multipart/form-data`:

| Field | Type | Description |
|---|---|---|
| `image` | file | Input image (JPEG, PNG, …) |
| `point_coords` | string (JSON) | `[[x, y], …]` – foreground/background point prompts |
| `point_labels` | string (JSON) | `[1, 0, …]` – `1` = foreground, `0` = background |
| `box` | string (JSON) | `[x1, y1, x2, y2]` – bounding-box prompt |
| `multimask_output` | bool (form) | Return 3 candidate masks (default `true`) |

Query parameter: `output_format` – `masks` (base64 PNG), `polygons`, or `both` (default `masks`).

### POST /api/segment/auto

Accepts `multipart/form-data`:

| Field | Type | Description |
|---|---|---|
| `image` | file | Input image |

Query parameters: `max_masks` (default `50`), `output_format` (same as above).

## Development

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest
```
