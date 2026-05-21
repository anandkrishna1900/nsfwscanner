"""
moderation/gatekeeper.py — Stage 1: Content-type router (anime vs real/photo).

Uses deepghs/anime_real_cls ONNX model on CPU.
Returns "real", "anime", or "uncertain" based on confidence threshold.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_CONFIDENCE_THRESHOLD = 0.70  # below this → "uncertain", route to both branches
_DEFAULT_INPUT_SIZE = 224


class ContentTypeRouter:
    """
    Routes an image to the real or anime branch.

    Returns ("real"|"anime"|"uncertain", confidence).
    Confidence < 0.70 → "uncertain" (both branches used, higher severity wins).

    All inference is synchronous — wrap calls in asyncio.to_thread().
    """

    def __init__(self, cache_dir: str) -> None:
        self._cache_dir = cache_dir
        self._session = None
        self._input_name: Optional[str] = None
        self._output_name: Optional[str] = None
        self._input_size: int = _DEFAULT_INPUT_SIZE
        # Labels: index 0 = anime/illustration, index 1 = real/photo
        # (will be confirmed from model metadata if available)
        self._labels: list[str] = ["anime", "real"]
        self._load()

    def _load(self) -> None:
        """Download and load the deepghs/anime_real_cls ONNX model."""
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        logger.info("[Gatekeeper] Loading deepghs/anime_real_cls ONNX…")

        # Try common filenames
        for filename in ("model.onnx", "anime_real_cls.onnx", "classifier.onnx"):
            try:
                model_path = hf_hub_download(
                    repo_id="deepghs/anime_real_cls",
                    filename=filename,
                    cache_dir=self._cache_dir,
                    local_files_only=False,
                )
                break
            except Exception:
                continue
        else:
            raise RuntimeError(
                "Could not download deepghs/anime_real_cls model. "
                "Check MODEL_CACHE_DIR and internet connection."
            )

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self._session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

        # Infer input size from model graph
        input_shape = self._session.get_inputs()[0].shape
        if len(input_shape) == 4 and isinstance(input_shape[2], int) and input_shape[2] > 0:
            self._input_size = input_shape[2]

        logger.info("[Gatekeeper] Ready (input size: %dx%d)", self._input_size, self._input_size)

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        """Resize and normalize the image to the model's expected input."""
        img = image.convert("RGB").resize(
            (self._input_size, self._input_size), Image.BICUBIC
        )
        arr = np.array(img, dtype=np.float32) / 255.0
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        arr = arr.transpose(2, 0, 1)           # HWC → CHW
        return np.expand_dims(arr, axis=0)     # → NCHW

    def route(self, image: Image.Image) -> Tuple[str, float]:
        """
        Classify the image as "real", "anime", or "uncertain".

        Returns:
            (label, confidence) where label is "real" | "anime" | "uncertain"
        """
        if self._session is None:
            logger.error("[Gatekeeper] Session not initialized — routing uncertain")
            return ("uncertain", 0.0)

        inp = self._preprocess(image)

        try:
            outputs = self._session.run(
                [self._output_name],
                {self._input_name: inp},
            )
        except Exception as e:
            logger.warning("[Gatekeeper] Inference error: %s", e)
            return ("uncertain", 0.0)

        raw = outputs[0][0]  # shape: (N,) where N = number of classes

        # Apply softmax if needed
        def softmax(x: np.ndarray) -> np.ndarray:
            e = np.exp(x - np.max(x))
            return e / e.sum()

        probs = softmax(raw) if raw.max() > 1.0 or raw.min() < 0.0 else raw

        best_idx = int(np.argmax(probs))
        confidence = float(probs[best_idx])

        if confidence < _CONFIDENCE_THRESHOLD:
            label = "uncertain"
        else:
            # Map index to label; fall back to index comparison
            if best_idx < len(self._labels):
                label = self._labels[best_idx]
            else:
                label = "real" if best_idx == 1 else "anime"

        logger.debug(
            "[Gatekeeper] probs=%s → label=%s (confidence=%.3f)",
            {l: f"{p:.3f}" for l, p in zip(self._labels, probs)},
            label,
            confidence,
        )
        return (label, confidence)
