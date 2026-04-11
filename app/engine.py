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
            return self._predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=multimask_output,
            )

    def predict_auto(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Run automatic mask generation.  Thread-safe (serialized internally)."""
        self.ensure_ready()
        with self._predict_lock:
            return self._auto_generator.generate(image)

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

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            if self._device == "cpu":
                logger.warning("CUDA not available – running on CPU (slow)")

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
            logger.info(
                "SAM model ready in %.1f s", time.time() - self._start_time
            )
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
