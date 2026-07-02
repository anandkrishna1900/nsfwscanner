from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

class AdamCoddPrefilter:
    """
    ONNX-based NSFW pre-filter using AdamCodd/vit-base-nsfw-detector.

    This model stays loaded permanently at startup.
    All inference is synchronous — wrap calls in asyncio.to_thread().
    """

    def __init__(self, cache_dir: str) -> None:
        self._cache_dir = cache_dir
        self._session = None
        self._input_name: Optional[str] = None
        self._output_name: Optional[str] = None
        self._input_size: int = 384
        self._load()

    def _load(self) -> None:
        """Download and load the ONNX model at startup."""
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        logger.info("[Prefilter] Loading AdamCodd/vit-base-nsfw-detector ONNX…")

        try:
            model_path = hf_hub_download(
                repo_id="AdamCodd/vit-base-nsfw-detector",
                filename="model.onnx",
                cache_dir=self._cache_dir,
                local_files_only=False,
            )
        except Exception:
            # Try alternate filename
            try:
                model_path = hf_hub_download(
                    repo_id="AdamCodd/vit-base-nsfw-detector",
                    filename="onnx/model.onnx",
                    cache_dir=self._cache_dir,
                    local_files_only=False,
                )
            except Exception as e:
                logger.error("[Prefilter] Failed to download model: %s", e)
                raise

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

        logger.info("[Prefilter] Ready (input size: %dx%d)", self._input_size, self._input_size)

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        """Resize, normalize, and convert image to float32 NCHW numpy array."""
        from utils.image_utils import prepare_image_for_onnx
        return prepare_image_for_onnx(
            image,
            target_size=self._input_size,
            normalize_imagenet=True,
            to_chw=True,
        )

    def score(self, image: Image.Image) -> float:
        """
        Run ONNX inference and return the NSFW probability (0.0–1.0).

        The model outputs logits or probabilities for [safe, nsfw].
        We return the nsfw probability.
        """
        if self._session is None:
            logger.error("[Prefilter] Session not initialized")
            return 1.0  # fail open — let downstream models decide

        inp = self._preprocess(image)

        try:
            outputs = self._session.run(
                [self._output_name],
                {self._input_name: inp},
            )
        except Exception as e:
            logger.warning("[Prefilter] Inference error: %s", e)
            return 1.0  # fail open

        raw = outputs[0][0]

        if len(raw) == 2:
            total = float(raw[0]) + float(raw[1])
            if abs(total - 1.0) > 0.1:
                # Logits — apply softmax
                import math
                e0 = math.exp(float(raw[0]))
                e1 = math.exp(float(raw[1]))
                nsfw_prob = e1 / (e0 + e1)
            else:
                nsfw_prob = float(raw[1])
        elif len(raw) == 1:
            # Single sigmoid output
            import math
            nsfw_prob = 1.0 / (1.0 + math.exp(-float(raw[0])))
        else:
            nsfw_prob = float(np.max(raw))

        logger.debug("[Prefilter] NSFW score: %.4f", nsfw_prob)
        return float(nsfw_prob)

    def is_worth_checking(self, image: Image.Image, threshold: float = 0.15) -> bool:
        """
        Return True if the image should be passed to downstream models.
        Returns False (skip) only if obviously safe (score < threshold).
        """
        return self.score(image) >= threshold
