from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from PIL import Image

logger = logging.getLogger(__name__)

# Fallback defaults — actual values are loaded from sensitivity.json
_DEFAULT_MODERATION_LABELS: dict[str, float] = {
    "FEMALE_GENITALIA_EXPOSED": 0.40,
    "MALE_GENITALIA_EXPOSED": 0.40,
    "ANUS_EXPOSED": 0.45,
    "MALE_GENITALIA_COVERED": 0.65,
    "FEMALE_BREAST_EXPOSED": 0.40,
}

_DEFAULT_REVIEW_OFFSET: float = 0.10
_DEFAULT_LOW_QUALITY_RETRY_LOW: float = 0.30
_DEFAULT_LOW_QUALITY_RETRY_HIGH: float = 0.50


@dataclass
class BranchResult:
    verdict: str
    detections: List[Tuple[str, float, Optional[list]]] = field(default_factory=list)
    max_score: float = 0.0
    model: str = "nudenet"


def _to_rgb(image: Image.Image) -> Image.Image:
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        canvas = Image.new("RGBA", image.size, (255, 255, 255, 255))
        canvas.alpha_composite(image.convert("RGBA"))
        return canvas.convert("RGB")
    return image.convert("RGB")


def preprocess_real_image(image: Image.Image) -> Image.Image:
    """
    Enhance low-quality real images before NudeNet inference.

    Uses OpenCV only: preserve aspect ratio, upscale small images to a 640px
    minimum dimension, reduce JPEG artifacts, normalize local contrast with CLAHE,
    and apply mild sharpening without destroying skin detail.
    """
    import cv2
    import numpy as np

    rgb_image = _to_rgb(image)
    original_width, original_height = rgb_image.size
    rgb = np.array(rgb_image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    height, width = bgr.shape[:2]
    min_dim = min(width, height)
    if min_dim < 640:
        scale = 640.0 / float(min_dim)
        new_size = (int(round(width * scale)), int(round(height * scale)))
        bgr = cv2.resize(bgr, new_size, interpolation=cv2.INTER_CUBIC)
        logger.debug(
            "[RealBranch] Upscaled real image from %dx%d to %dx%d",
            original_width,
            original_height,
            new_size[0],
            new_size[1],
        )

    denoised = cv2.fastNlMeansDenoisingColored(bgr, None, 3, 3, 7, 21)

    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    contrast = cv2.merge((l_channel, a_channel, b_channel))
    contrast_bgr = cv2.cvtColor(contrast, cv2.COLOR_LAB2BGR)

    blurred = cv2.GaussianBlur(contrast_bgr, (0, 0), sigmaX=0.8, sigmaY=0.8)
    sharpened = cv2.addWeighted(contrast_bgr, 1.25, blurred, -0.25, 0)
    enhanced = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)

    logger.debug(
        "[RealBranch] Enhancement applied: original=%dx%d enhanced=%dx%d",
        original_width,
        original_height,
        enhanced.shape[1],
        enhanced.shape[0],
    )
    return Image.fromarray(enhanced)


def _upscale_image(image: Image.Image, scale: float) -> Image.Image:
    import cv2
    import numpy as np

    rgb_image = _to_rgb(image)
    arr = np.array(rgb_image)
    height, width = arr.shape[:2]
    new_size = (int(round(width * scale)), int(round(height * scale)))
    upscaled = cv2.resize(arr, new_size, interpolation=cv2.INTER_CUBIC)
    logger.debug(
        "[RealBranch] Created %.2fx upscale variant: %dx%d -> %dx%d",
        scale,
        width,
        height,
        new_size[0],
        new_size[1],
    )
    return Image.fromarray(upscaled)


def _center_crop(image: Image.Image, crop_ratio: float = 0.78) -> Image.Image:
    rgb_image = _to_rgb(image)
    width, height = rgb_image.size
    crop_width = max(1, int(width * crop_ratio))
    crop_height = max(1, int(height * crop_ratio))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    cropped = rgb_image.crop((left, top, left + crop_width, top + crop_height))
    logger.debug(
        "[RealBranch] Created center crop variant: %dx%d -> %dx%d",
        width,
        height,
        crop_width,
        crop_height,
    )
    return cropped


class RealBranch:
    """
    Detects exposed genitals, anus, and exposed female breasts in real images.
    Callers wrap scan() in asyncio.to_thread() to keep Discord's event loop clear.
    """

    def __init__(self, config=None) -> None:
        self._detector = None

        cfg = config.sensitivity.get("real_branch", {}) if config else {}
        labels = cfg.get("labels", _DEFAULT_MODERATION_LABELS).copy()
        if config:
            for k in labels:
                labels[k] = config.get_threshold(labels[k])
        self.MODERATION_LABELS = labels

        self.REVIEW_OFFSET = cfg.get("review_offset", _DEFAULT_REVIEW_OFFSET)
        self.LOW_QUALITY_RETRY_LOW = cfg.get("low_quality_retry_low", _DEFAULT_LOW_QUALITY_RETRY_LOW)
        self.LOW_QUALITY_RETRY_HIGH = cfg.get("low_quality_retry_high", _DEFAULT_LOW_QUALITY_RETRY_HIGH)

    def _ensure_detector(self) -> None:
        if self._detector is None:
            from nudenet import NudeDetector
            self._detector = NudeDetector()
            logger.info("[RealBranch] NudeDetector loaded")

    def _save_temp_jpeg(self, image: Image.Image, temp_paths: List[str]) -> str:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            temp_paths.append(tmp.name)
            _to_rgb(image).save(tmp.name, format="JPEG", quality=95)
            return tmp.name

    def _detect_variant(self, image: Image.Image, variant_name: str, temp_paths: List[str]) -> List[dict]:
        path = self._save_temp_jpeg(image, temp_paths)
        raw = self._detector.detect(path)
        logger.debug(
            "[RealBranch] Variant '%s' produced %d raw NudeNet detections",
            variant_name,
            len(raw),
        )
        for detection in raw:
            detection["_variant"] = variant_name
        return raw

    def _filter_moderation_detections(self, raw_detections: List[dict]) -> List[dict]:
        return [
            detection
            for detection in raw_detections
            if detection.get("class", "") in self.MODERATION_LABELS
        ]

    def _max_relevant_score(self, detections: List[dict]) -> float:
        if not detections:
            return 0.0
        return max(float(detection.get("score", 0.0)) for detection in detections)

    def _is_near_threshold(self, detections: List[dict]) -> bool:
        return any(
            self.LOW_QUALITY_RETRY_LOW <= float(detection.get("score", 0.0)) <= self.LOW_QUALITY_RETRY_HIGH
            for detection in detections
        )

    def _score_detections(self, detections: List[dict]) -> BranchResult:
        flagged: List[Tuple[str, float, Optional[list]]] = []
        max_score = 0.0
        verdict = "SAFE"

        detections.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)

        for detection in detections:
            label = detection["class"]
            score = float(detection.get("score", 0.0))
            box = detection.get("box")
            threshold = self.MODERATION_LABELS[label]
            review_low = threshold - self.REVIEW_OFFSET

            logger.debug(
                "[RealBranch] %s score=%.3f threshold=%.2f review_low=%.2f variant=%s",
                label,
                score,
                threshold,
                review_low,
                detection.get("_variant", "unknown"),
            )

            if score < review_low:
                continue

            flagged.append((label, score, box))
            max_score = max(max_score, score)
            if score >= threshold:
                verdict = "BLOCK"
            elif verdict != "BLOCK":
                verdict = "REVIEW"

        if not flagged:
            return BranchResult(verdict="SAFE", max_score=0.0)

        logger.info(
            "[RealBranch] Verdict: %s | selected_confidence=%.3f | flagged=%s",
            verdict,
            max_score,
            [(label, f"{score:.3f}") for label, score, _ in flagged],
        )
        return BranchResult(verdict=verdict, detections=flagged, max_score=max_score)

    def scan(self, image: Image.Image) -> BranchResult:
        self._ensure_detector()

        temp_paths: List[str] = []
        try:
            rgb_image = _to_rgb(image)
            logger.debug("[RealBranch] Original resolution: %dx%d", *rgb_image.size)

            raw_detections: List[dict] = []

            original_raw = self._detect_variant(rgb_image, "original", temp_paths)
            raw_detections.extend(original_raw)
            original_relevant = self._filter_moderation_detections(original_raw)
            original_max = self._max_relevant_score(original_relevant)
            logger.debug("[RealBranch] Original max moderation confidence: %.3f", original_max)

            enhanced = preprocess_real_image(rgb_image)
            enhanced_raw = self._detect_variant(enhanced, "enhanced", temp_paths)
            raw_detections.extend(enhanced_raw)
            enhanced_relevant = self._filter_moderation_detections(enhanced_raw)
            enhanced_max = self._max_relevant_score(enhanced_relevant)
            logger.debug(
                "[RealBranch] Enhancement confidence before/after: %.3f -> %.3f",
                original_max,
                enhanced_max,
            )

            upscaled = _upscale_image(rgb_image, 1.25)
            upscale_raw = self._detect_variant(upscaled, "upscale_1_25x", temp_paths)
            raw_detections.extend(upscale_raw)

            moderation_detections = self._filter_moderation_detections(raw_detections)
            retry_triggered = self._is_near_threshold(moderation_detections)
            logger.debug(
                "[RealBranch] Retry triggered=%s (near-threshold window %.2f-%.2f)",
                retry_triggered,
                self.LOW_QUALITY_RETRY_LOW,
                self.LOW_QUALITY_RETRY_HIGH,
            )

            if retry_triggered:
                center_crop = _center_crop(enhanced)
                crop_raw = self._detect_variant(center_crop, "enhanced_center_crop", temp_paths)
                raw_detections.extend(crop_raw)
                moderation_detections = self._filter_moderation_detections(raw_detections)

            logger.debug(
                "[RealBranch] raw=%d detections, moderation=%d after filter",
                len(raw_detections),
                len(moderation_detections),
            )

            if not moderation_detections:
                logger.debug("[RealBranch] No moderation labels found -> SAFE")
                return BranchResult(verdict="SAFE", max_score=0.0)

            result = self._score_detections(moderation_detections)
            logger.debug("[RealBranch] Final selected confidence: %.3f", result.max_score)
            return result

        except Exception as e:
            logger.error("[RealBranch] Scan error: %s", e, exc_info=True)
            return BranchResult(verdict="SAFE", max_score=0.0)

        finally:
            for path in temp_paths:
                if not os.path.exists(path):
                    continue
                try:
                    os.remove(path)
                except Exception as cleanup_err:
                    logger.warning(
                        "[RealBranch] Failed to clean up temp file %s: %s",
                        path,
                        cleanup_err,
                    )
