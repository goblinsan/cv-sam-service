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

# Local test UI (served by the same container)
open http://localhost:5201/

# Interactive API docs
open http://localhost:5201/docs
```

## Local test UI

A Vite + React + TypeScript single-page app ships inside the Docker image and
is served from the root path (`/`) by the FastAPI process:

| Feature | Details |
|---|---|
| Service status panel | Polls `/api/health` and `/api/info` every 10 s; shows SAM variant, GPU readiness, VRAM metrics |
| Image workspace | Drag-and-drop upload (PNG · JPEG · WebP), thumbnail preview, dimensions/format/size metadata, replace and clear actions |

### Compose service routing

The `docker-compose.yml` defines a **single service** (`cv-sam-api`).  There
is no separate UI container — the pre-built React SPA is embedded in the API
image during the multi-stage Docker build and served by the FastAPI process on
the same port (`5201`):

| Path prefix | Handler |
|---|---|
| `/api/*` | FastAPI routers (inference, health, info) |
| `/assets/*` | Vite build artifacts (JS, CSS, fonts) |
| `/*` | SPA fallback → `index.html` |

Under gateway-control-plane the stack deploys as a single `container-service`
using `build.strategy: repo-compose`.  Only container port `5201` needs to be
published to the host — both the API and the browser UI are accessible on
that port.

### Keeping API and UI in sync

When adding or changing an API endpoint, update the React UI in `ui/` to
expose the new capability from the browser.  Because the UI bundle is compiled
into the Docker image during the multi-stage build, **any UI change requires a
full image rebuild**:

```bash
docker compose up --build
```

The same applies to gateway-control-plane deployments: the `repo-compose`
strategy always runs `docker compose up --build` from a fresh checkout, so
both the updated API code **and** the updated `ui/` source must be present in
the repository before the deploy is triggered.

### Running the UI in development (hot-reload)

```bash
# Terminal 1 – start the API (model warmup runs in background)
docker compose up

# Terminal 2 – start the Vite dev server (proxies /api/* to localhost:5201)
cd ui
npm install
npm run dev
# open http://localhost:5173
```

### Building the UI standalone

```bash
cd ui
npm install
npm run build    # output written to ui/dist/
```

The `docker compose up --build` command already runs this build step automatically
via the multi-stage Dockerfile; no manual build is required for the Docker workflow.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `CV_SAM_VARIANT` | `vit_b` | SAM model variant: `vit_b` (~1.5 GB VRAM), `vit_l`, or `vit_h` (~2.5 GB VRAM) |
| `MODELS_HOST_DIR` | `/data/models` | Host path mounted to `/data/models` inside the container |
| `MODEL_DIR` | `/data/models/sam` | Directory where SAM checkpoint is cached |
| `HOST_PORT` | `5201` | Host port mapped in `docker-compose.yml` |

## API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Liveness + readiness probe (always HTTP 200) |
| `GET` | `/api/info` | GPU info, VRAM usage, model variant, readiness |
| `POST` | `/api/segment` | Prompted segmentation (point / box prompts) |
| `POST` | `/api/segment/auto` | Automatic mask generation (no prompts required) |
| `POST` | `/api/analyze` | Dominant colors, edge density, histogram stats |
| `POST` | `/api/extract-palette` | Color palette extraction (k-means) |
| `POST` | `/api/transform` | Image transform pipeline (resize, crop, rotate, …) |

### Image input

Every inference endpoint (`segment`, `segment/auto`, `analyze`,
`extract-palette`) accepts images in two ways:

* **Multipart file upload** – include the image as a `multipart/form-data`
  `image` field.
* **Remote URL** – pass `?image_url=https://…` as a query parameter.
  The service fetches the image (http/https only, max 10 MB, 10 s timeout).

### POST /api/segment

Accepts `multipart/form-data`:

| Field | Type | Description |
|---|---|---|
| `image` | file | Input image (JPEG, PNG, …) – or use `image_url` |
| `point_coords` | string (JSON) | `[[x, y], …]` – foreground/background point prompts |
| `point_labels` | string (JSON) | `[1, 0, …]` – `1` = foreground, `0` = background |
| `box` | string (JSON) | `[x1, y1, x2, y2]` – bounding-box prompt |
| `multimask_output` | bool (form) | Return 3 candidate masks (default `true`) |

Query parameters: `output_format` – `masks` (base64 PNG), `polygons`, or `both` (default `masks`); `image_url` – remote image URL.

### POST /api/segment/auto

Accepts `multipart/form-data`:

| Field | Type | Description |
|---|---|---|
| `image` | file | Input image – or use `image_url` |

Query parameters: `max_masks` (default `50`), `output_format` (same as above), `image_url`.

## Integration docs

| Document | Audience | Description |
|---|---|---|
| `docs/workflow-action-contract.md` | gateway-api workflow authors | Request/response contract, examples (URL + upload), downstream patterns |
| `docs/agent-tool-openapi.yml` | chat-platform / LLM agent developers | OpenAPI 3.1 spec with input limits, auth notes, stable response examples |
| `docs/workload-manifest.example.yml` | gateway-control-plane operators | Container workload manifest |

## Control-plane deployment

### Example workload manifest

`docs/workload-manifest.example.yml` contains a ready-to-copy
`container-service` workload manifest for
[gateway-control-plane](https://github.com/goblinsan/gateway-control-plane).
Key settings:

| Field | Value |
|---|---|
| `build.strategy` | `repo-compose` |
| Published port | `5201` |
| `networkMode` | `bridge` |
| `runtimeClass` | `nvidia` |
| Health check | `GET /api/health` |
| Volume contract | host path → `/data/models` in container |

### Operator verification steps

Run these checks locally with Docker Compose before registering the workload
in the control plane:

```bash
# 1. Build and start the service
docker compose up --build -d

# 2. Confirm the health probe is green
curl -f http://localhost:5201/api/health   # expect HTTP 200, "status":"ok"

# 3. Check GPU runtime and VRAM
curl http://localhost:5201/api/info        # confirm device != "cpu", vram_total_mb reported

# 4. VRAM headroom: if STT (Whisper large-v3) is already running on the same
#    node, expect ~3.1 GB already consumed.  vram_total_mb - (3100 + 1500)
#    should be positive before accepting traffic.

# 5. Stop the service
docker compose down
```

### Local operator follow-up steps (outside this repo)

The following steps require access to your private gateway config and must be
performed locally — do **not** commit private node IDs, host names, or volume
paths to this repository:

1. **Register the workload** – paste the content of
   `docs/workload-manifest.example.yml` into your local `gateway.config.json`
   `workloads` array (or the equivalent Nodes-tab form in the control-plane
   UI).
2. **Set node identity** – replace the `<REPLACE_ME>` repo URL and set the
   real `nodeId` / `host` values for the target GPU node in your untracked
   config file.
3. **Choose durable volume paths** – pick a stable host path for
   `volumes[0].hostPath` (e.g. `/mnt/nvme/models`) so the SAM checkpoint
   persists across container restarts.
4. **Deploy** – trigger deployment from the Nodes tab in the control-plane UI
   or via the CLI: `gcp workload deploy cv-sam-service --node <nodeId>`.
5. **Confirm monitoring** – verify that the health probe turns green in the
   control-plane dashboard within ~2 minutes of deploy.

## VRAM budget and co-tenancy

| Service | Model | Approx. VRAM |
|---|---|---|
| cv-sam-service | SAM `vit_b` | ~1.5 GB |
| cv-sam-service | SAM `vit_h` | ~2.5 GB |
| stt-service | Whisper `large-v3` | ~3.1 GB |
| **Total (vit_b + large-v3)** | | **~4.6 GB** |

An 8 GB card (e.g. RTX 3070) has approximately **3.4 GB of headroom** when
both services are idle-loaded.  Under simultaneous inference the combined
peak can reach ~6–7 GB — stay within budget by using `vit_b` for the CV
service.

### Load-shedding / queue strategy

Both services are independently deployed and share the GPU passively (PyTorch
allocates from the same CUDA context).  Recommended mitigations when both are
under heavy concurrent load:

* **Queue at the API gateway** – configure the gateway to limit concurrent
  requests to `cv-sam-service` to **2** simultaneous inference calls and to
  `stt-service` to **1** long transcription job.  Excess requests receive
  HTTP 429 and should be retried by the caller.
* **Model variant selection** – prefer `vit_b` over `vit_l`/`vit_h` unless
  the use-case demands higher mask quality; this keeps VRAM pressure low.
* **Health-gate inference** – the `/api/info` endpoint reports
  `vram_reserved_mb`; clients or the gateway can check this value and reject
  requests early when free VRAM drops below a configurable threshold (e.g.
  500 MB).

## Development

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest
```
