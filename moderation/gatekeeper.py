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
        self._labels: list[str] = ["anime", "real"]

    def _load(self) -> None:
        """Download and load the deepghs/anime_real_cls ONNX model."""
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download, list_repo_files

        logger.info("[Gatekeeper] Loading deepghs/anime_real_cls ONNX…")

        try:
            repo_files = list(list_repo_files("deepghs/anime_real_cls"))
        except Exception as e:
            raise RuntimeError(
                f"Could not list files in deepghs/anime_real_cls: {e}. "
                "Check internet connection."
            )

        onnx_files = [f for f in repo_files if f.endswith(".onnx")]
        if not onnx_files:
            raise RuntimeError(
                "No ONNX file found in deepghs/anime_real_cls. "
                "The repo structure may have changed."
            )

        # Prefer a caformer variant (best quality); fall back to first .onnx found
        preferred = next(
            (f for f in onnx_files if "caformer" in f.lower()),
            onnx_files[0],
        )
        logger.info("[Gatekeeper] Selected model file: %s", preferred)

        try:
            model_path = hf_hub_download(
                repo_id="deepghs/anime_real_cls",
                filename=preferred,
                cache_dir=self._cache_dir,
                local_files_only=False,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to download deepghs/anime_real_cls/{preferred}: {e}"
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

        input_shape = self._session.get_inputs()[0].shape
        if len(input_shape) == 4 and isinstance(input_shape[2], int) and input_shape[2] > 0:
            self._input_size = input_shape[2]

        logger.info("[Gatekeeper] Ready (input size: %dx%d)", self._input_size, self._input_size)

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        """Resize and normalize the image to the model's expected input."""
        from utils.image_utils import prepare_image_for_onnx
        return prepare_image_for_onnx(
            image,
            target_size=self._input_size,
            normalize_imagenet=True,
            to_chw=True,
        )

    def route(self, image: Image.Image) -> Tuple[str, float]:
        """
        Classify the image as "real", "anime", or "uncertain".

        Returns:
            (label, confidence) where label is "real" | "anime" | "uncertain"
        """
        if self._session is None:
            try:
                self._load()
            except Exception as e:
                logger.error("[Gatekeeper] Failed to load session dynamically: %s — routing uncertain", e)
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

        raw = outputs[0][0]

        def softmax(x: np.ndarray) -> np.ndarray:
            e = np.exp(x - np.max(x))
            return e / e.sum()

        probs = softmax(raw) if raw.max() > 1.0 or raw.min() < 0.0 else raw

        best_idx = int(np.argmax(probs))
        confidence = float(probs[best_idx])

        if confidence < _CONFIDENCE_THRESHOLD:
            label = "uncertain"
        else:
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
