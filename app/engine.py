"""SAM engine: thread-safe singleton that loads Meta SAM on startup."""

import gc
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_CHECKPOINT_URLS: Dict[str, str] = {
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
}

_CHECKPOINT_NAMES: Dict[str, str] = {
    "vit_b": "sam_vit_b_01ec64.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_h": "sam_vit_h_4b8939.pth",
}


class SAMEngine:
    """Thread-safe wrapper around Meta's Segment Anything Model."""

    def __init__(self) -> None:
        self._ready = False
        self._load_lock = threading.Lock()
        self._load_condition = threading.Condition(self._load_lock)
        self._loading = False
        self._predict_lock = threading.Lock()

        self._sam: Any = None
        self._predictor: Any = None
        self._auto_generator: Any = None

        self._model_variant: str = os.environ.get("CV_SAM_VARIANT", "vit_b")
        self._model_dir: Path = Path(os.environ.get("MODEL_DIR", "/data/models/sam"))
        self._device: Optional[str] = None
        self._load_error: Optional[str] = None
        self._start_time: float = time.time()

        # VRAM idle-unload (mirrors pyannote pattern in stt-service).  When
        # ``CV_SAM_IDLE_UNLOAD_SEC`` > 0, a daemon thread unloads the SAM model
        # after that many seconds of inactivity, freeing ~2 GB of VRAM until
        # the next request lazily reloads it.
        try:
            self._idle_unload_sec: int = int(
                os.environ.get("CV_SAM_IDLE_UNLOAD_SEC", "0")
            )
        except ValueError:
            logger.warning(
                "Invalid CV_SAM_IDLE_UNLOAD_SEC=%r; disabling idle-unload",
                os.environ.get("CV_SAM_IDLE_UNLOAD_SEC"),
            )
            self._idle_unload_sec = 0
        self._last_used: float = time.monotonic()
        self._reaper_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load model weights.  Idempotent; safe to call from multiple threads."""
        with self._load_condition:
            if self._ready or self._loading:
                return
            self._loading = True
            self._load_error = None
            self._start_time = time.time()

        # Actual loading is done outside the short critical section so the lock
        # is not held during the (potentially minutes-long) download + load.
        try:
            self._do_load()
        finally:
            with self._load_condition:
                self._loading = False
                self._load_condition.notify_all()

    def ensure_ready(self) -> None:
        """Block until the model is ready, loading it on-demand when needed."""
        self.load()
        with self._load_condition:
            while self._loading and not self._ready and self._load_error is None:
                self._load_condition.wait()
            if not self._ready:
                raise RuntimeError(self._load_error or "Model not ready")

    def unload(self) -> None:
        """Release the SAM model so the next request lazily reloads it."""
        with self._load_condition:
            if self._loading:
                raise RuntimeError("Model is still loading")
            sam = self._sam
            predictor = self._predictor
            auto_generator = self._auto_generator
            if sam is None and predictor is None and auto_generator is None:
                return
            self._sam = None
            self._predictor = None
            self._auto_generator = None
            self._ready = False
            self._load_error = None

        with self._predict_lock:
            # Clear any image features the predictor has cached on GPU.  Without
            # this the SamPredictor keeps tensors referenced internally and
            # ``torch.cuda.empty_cache()`` cannot reclaim the blocks.
            try:
                if predictor is not None and hasattr(predictor, "reset_image"):
                    predictor.reset_image()
            except Exception:
                logger.debug(
                    "predictor.reset_image() failed during unload",
                    exc_info=True,
                )

            # Move SAM weights off the GPU before dropping references.  Just
            # ``del``-ing the python objects is not enough: PyTorch's caching
            # allocator only returns blocks to the OS once the underlying
            # tensors are released, and `sam.to("cpu")` is the most reliable
            # way to force that release for every submodule (image_encoder,
            # prompt_encoder, mask_decoder).  Observed effect on a tags-node
            # RTX 4060: cv-sam process drops from ~2.1 GiB to ~0.3 GiB
            # (the residual is the unavoidable per-process CUDA context).
            try:
                if sam is not None and hasattr(sam, "to"):
                    sam.to("cpu")
            except Exception:
                logger.debug(
                    "sam.to('cpu') failed during unload", exc_info=True,
                )

            del sam
            del predictor
            del auto_generator
            gc.collect()
            self._release_cuda_memory()
            logger.info("Unloaded SAM model to free VRAM")

    def predict(
        self,
        image: np.ndarray,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        multimask_output: bool = True,
    ):
        """Run prompted SAM prediction.  Thread-safe (serialized internally)."""
        self.ensure_ready()
        with self._predict_lock:
            self._predictor.set_image(image)
            try:
                return self._predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=box,
                    multimask_output=multimask_output,
                )
            finally:
                self._last_used = time.monotonic()

    def predict_auto(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Run automatic mask generation.  Thread-safe (serialized internally)."""
        self.ensure_ready()
        with self._predict_lock:
            try:
                return self._auto_generator.generate(image)
            finally:
                self._last_used = time.monotonic()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def loading(self) -> bool:
        return self._loading

    @property
    def model_variant(self) -> str:
        return self._model_variant

    @property
    def device(self) -> Optional[str]:
        return self._device

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def vram_info(self) -> Dict[str, Any]:
        """Return GPU/VRAM metrics (all None when running on CPU)."""
        try:
            import torch  # noqa: PLC0415
        except ImportError:
            return {
                "device_name": "cpu",
                "vram_total_mb": None,
                "vram_reserved_mb": None,
                "vram_allocated_mb": None,
            }
        if self._device == "cuda" and torch.cuda.is_available():
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            total = props.total_memory / (1024 ** 2)
            reserved = torch.cuda.memory_reserved(idx) / (1024 ** 2)
            allocated = torch.cuda.memory_allocated(idx) / (1024 ** 2)
            return {
                "device_name": props.name,
                "vram_total_mb": round(total, 1),
                "vram_reserved_mb": round(reserved, 1),
                "vram_allocated_mb": round(allocated, 1),
            }
        return {
            "device_name": "cpu",
            "vram_total_mb": None,
            "vram_reserved_mb": None,
            "vram_allocated_mb": None,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _do_load(self) -> None:
        try:
            import torch  # noqa: PLC0415

            # Import lazily so the module can be imported without segment_anything
            # being installed (e.g. during unit-test collection).
            from segment_anything import (  # type: ignore[import]
                SamAutomaticMaskGenerator,
                SamPredictor,
                sam_model_registry,
            )

            requested = os.environ.get("CV_SAM_DEVICE", "auto").strip().lower()
            if requested not in {"auto", "cuda", "cpu"}:
                logger.warning(
                    "Invalid CV_SAM_DEVICE=%r; falling back to 'auto'",
                    requested,
                )
                requested = "auto"
            if requested == "cuda":
                if not torch.cuda.is_available():
                    raise RuntimeError(
                        "CV_SAM_DEVICE=cuda but torch.cuda.is_available() is False"
                    )
                self._device = "cuda"
            elif requested == "cpu":
                self._device = "cpu"
            else:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            if self._device == "cpu":
                logger.warning("Running on CPU (slow); CV_SAM_DEVICE=%s", requested)

            variant = self._model_variant
            if variant not in _CHECKPOINT_URLS:
                raise ValueError(
                    f"Unknown SAM variant: {variant!r}. "
                    f"Choose from {list(_CHECKPOINT_URLS)}"
                )

            self._model_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = self._model_dir / _CHECKPOINT_NAMES[variant]

            if not checkpoint_path.exists():
                self._download_checkpoint(variant, checkpoint_path)

            logger.info("Loading SAM %s on %s …", variant, self._device)
            sam = sam_model_registry[variant](checkpoint=str(checkpoint_path))
            sam.to(self._device)

            predictor = SamPredictor(sam)
            auto_generator = SamAutomaticMaskGenerator(sam)
            with self._load_condition:
                self._sam = sam
                self._predictor = predictor
                self._auto_generator = auto_generator
                self._ready = True
                self._load_error = None
                self._last_used = time.monotonic()
            logger.info(
                "SAM model ready in %.1f s", time.time() - self._start_time
            )
            self._ensure_reaper()
        except Exception as exc:
            logger.exception("Failed to load SAM model: %s", exc)
            with self._load_condition:
                self._sam = None
                self._predictor = None
                self._auto_generator = None
                self._ready = False
                self._load_error = str(exc)

    def _release_cuda_memory(self) -> None:
        try:
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            logger.debug("CUDA cache release skipped", exc_info=True)

    # ------------------------------------------------------------------
    # Idle-unload reaper
    # ------------------------------------------------------------------

    def _ensure_reaper(self) -> None:
        """Start the idle-unload daemon thread on first successful load.

        No-op when ``CV_SAM_IDLE_UNLOAD_SEC`` is 0 (disabled) or the reaper
        is already running.
        """
        if self._idle_unload_sec <= 0:
            return
        if self._reaper_thread is not None and self._reaper_thread.is_alive():
            return
        t = threading.Thread(
            target=self._reaper,
            args=(self._idle_unload_sec,),
            daemon=True,
            name="cv-sam-reaper",
        )
        t.start()
        self._reaper_thread = t
        logger.info(
            "SAM idle-unload reaper started (timeout=%ds)",
            self._idle_unload_sec,
        )

    def _reaper(self, idle_timeout_sec: int) -> None:
        """Background thread: unload SAM after it has been idle long enough."""
        # Wake at most every 10 s, but no less often than 1/6th of the timeout
        # so we never miss the deadline by more than ~17% (matches the
        # pyannote reaper in stt-service).
        _REAPER_MIN_SLEEP_SEC = 10
        _REAPER_FRACTION = 6
        sleep_sec = max(
            _REAPER_MIN_SLEEP_SEC, idle_timeout_sec // _REAPER_FRACTION
        )
        while True:
            time.sleep(sleep_sec)
            try:
                with self._load_condition:
                    if not self._ready or self._loading:
                        continue
                    idle = time.monotonic() - self._last_used
                if idle >= idle_timeout_sec:
                    logger.info(
                        "SAM idle for %.0fs (limit %ds) — unloading to free VRAM",
                        idle, idle_timeout_sec,
                    )
                    try:
                        self.unload()
                    except RuntimeError:
                        # Concurrent load won the race; try again next tick.
                        logger.debug(
                            "Skipped idle-unload: model is loading",
                            exc_info=True,
                        )
            except Exception:
                logger.exception("SAM reaper iteration failed; continuing")

    def _download_checkpoint(self, variant: str, dest: Path) -> None:
        import requests  # noqa: PLC0415

        url = _CHECKPOINT_URLS[variant]
        logger.info("Downloading SAM checkpoint %s → %s", url, dest)
        with requests.get(url, stream=True, timeout=600) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with open(dest, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=65_536):
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if total and downloaded % (50 * 1024 * 1024) < 65_536:
                        logger.info(
                            "Download %.1f %%", 100 * downloaded / total
                        )
        logger.info("Checkpoint saved: %s", dest)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_engine: Optional[SAMEngine] = None
_engine_lock = threading.Lock()


def get_engine() -> SAMEngine:
    """Return the process-wide SAMEngine singleton (creates it if necessary)."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = SAMEngine()
    return _engine
