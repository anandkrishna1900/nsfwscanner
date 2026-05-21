"""
moderation/models/loader.py — Lazy model loader with GPU memory management.

All methods are synchronous — call them inside asyncio.to_thread().
Only one PyTorch GPU model may be loaded at a time per AGENTS.md hardware constraints.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

logger = logging.getLogger(__name__)

# ── Registry entry structure ──────────────────────────────────────────────────
# Each entry: {"hf_repo": str, "filename": str | None, "type": "onnx" | "nudenet"}
_REGISTRY: Dict[str, Dict[str, Any]] = {
    "prefilter": {
        "hf_repo": "AdamCodd/vit-base-nsfw-detector",
        "filename": "model.onnx",
        "type": "onnx",
    },
    "anime_real_cls": {
        "hf_repo": "deepghs/anime_real_cls",
        "filename": "model.onnx",
        "type": "onnx",
    },
    "wdv3_tagger": {
        "hf_repo": "SmilingWolf/wd-vit-large-tagger-v3",
        "filename": "model.onnx",
        "type": "onnx",
    },
    "anime_rating": {
        "hf_repo": "deepghs/anime_rating",
        "filename": "model.onnx",
        "type": "onnx",
    },
    "nudenet": {
        "hf_repo": None,
        "filename": None,
        "type": "nudenet",
    },
}


class ModelLoader:
    """
    Lazy model loader.  Only one model is kept in memory at a time (except the
    prefilter, which stays loaded permanently as per AGENTS.md).
    """

    def __init__(self, cache_dir: str, device: str = "cuda") -> None:
        self._cache_dir = cache_dir
        self._device = device.lower()
        self._loaded: Dict[str, Any] = {}
        os.makedirs(cache_dir, exist_ok=True)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _onnx_session(self, model_path: str):
        """Create an ONNX InferenceSession with CPUExecutionProvider."""
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.log_severity_level = 3  # suppress ONNX verbose logs
        session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        logger.debug("ONNX session created: %s", model_path)
        return session

    def _resolve_model_path(self, name: str) -> str:
        """Download the model from HuggingFace if not cached, return local path."""
        from huggingface_hub import hf_hub_download

        entry = _REGISTRY[name]
        repo = entry["hf_repo"]
        filename = entry["filename"]

        local_path = hf_hub_download(
            repo_id=repo,
            filename=filename,
            cache_dir=self._cache_dir,
            local_files_only=False,
        )
        logger.debug("Model path resolved: %s → %s", name, local_path)
        return local_path

    def _clear_vram(self) -> None:
        """Release CUDA VRAM if available."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("CUDA cache cleared")
        except Exception:
            pass

    # ── Public API ────────────────────────────────────────────────────────────

    def load_onnx(self, name: str):
        """
        Load an ONNX model by registry name.
        Returns an onnxruntime.InferenceSession.
        """
        if name not in _REGISTRY:
            raise KeyError(f"Unknown model: {name!r}")
        if _REGISTRY[name]["type"] != "onnx":
            raise ValueError(f"Model {name!r} is not an ONNX model")

        model_path = self._resolve_model_path(name)
        session = self._onnx_session(model_path)
        logger.info("[ModelLoader] Loaded ONNX model: %s", name)
        return session

    def load_nudenet(self):
        """
        Import and return a NudeDetector instance.
        NudeNet manages its own ONNX model internally.
        """
        from nudenet import NudeDetector

        detector = NudeDetector()
        logger.info("[ModelLoader] NudeDetector loaded")
        return detector

    def get_or_load(self, name: str) -> Any:
        """
        Lazy load: return cached model or load fresh.
        The prefilter is cached permanently; all others are loaded on demand.
        """
        if name not in self._loaded:
            if name == "nudenet":
                self._loaded[name] = self.load_nudenet()
            else:
                self._loaded[name] = self.load_onnx(name)
        return self._loaded[name]

    def unload(self, name: str) -> None:
        """
        Delete the model reference and free VRAM if applicable.
        Never unloads 'prefilter' (it stays loaded permanently).
        """
        if name == "prefilter":
            logger.debug("Skipping unload of prefilter (permanent)")
            return
        if name in self._loaded:
            del self._loaded[name]
            self._clear_vram()
            logger.info("[ModelLoader] Unloaded model: %s", name)

    @contextmanager
    def use(self, name: str) -> Generator[Any, None, None]:
        """
        Context manager: load the model, yield it, then unload.

        Usage:
            with loader.use("anime_real_cls") as session:
                result = session.run(...)
        """
        model = self.get_or_load(name)
        try:
            yield model
        finally:
            self.unload(name)

    @property
    def loaded_models(self) -> list[str]:
        """Return the names of currently loaded models."""
        return list(self._loaded.keys())
