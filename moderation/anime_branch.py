"""
moderation/anime_branch.py - Stage 2B: Anime/hentai content branch.

Primary: SmilingWolf/wd-vit-large-tagger-v3 (WDv3) ONNX tagger.
Secondary: deepghs/anime_rating ONNX classifier.

Anime moderation tiers: SAFE, SUGGESTIVE, NSFW, EXPLICIT.
Ratings are broad classifiers; anatomical evidence tags drive escalation.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

GENITAL_TAGS: Dict[str, float] = {
    "penis": 0.72,
    "vagina": 0.75,
    "pussy": 0.75,
    "genitals": 0.80,
    "testicles": 0.72,
    "erection": 0.70,
    "anus": 0.72,
    "phallus": 0.72,
}

BREAST_TAGS: Dict[str, float] = {
    "nipples": 0.70,
    "bare_breasts": 0.72,
    "breasts_out": 0.75,
    "areola_slip": 0.72,
    "nipple_slip": 0.72,
    "breast_slip": 0.72,
    "one_breast_out": 0.72,
    "topless": 0.68,
    "large_areolae": 0.70,
    "dark_areolae": 0.70,
    "light_areolae": 0.70,
    "puffy_nipples": 0.70,
    "huge_nipples": 0.70,
    "dark_nipples": 0.70,
    "colored_nipples": 0.70,
}

SAFE_CONTEXT_TAGS: Dict[str, float] = {
    "bikini": 0.15,
    "swimsuit": 0.15,
    "underwear": 0.12,
    "bra": 0.10,
    "panties": 0.12,
    "lingerie": 0.10,
    "cameltoe": 0.08,
    "cleavage": 0.08,
    "sideboob": 0.10,
    "micro_bikini": 0.12,
}

SUGGESTIVE_TAGS: Dict[str, float] = {
    "lingerie": 0.62,
    "underboob": 0.60,
    "cameltoe": 0.62,
    "implied_nudity": 0.62,
    "clothes_lift": 0.65,
    "no_bra": 0.65,
}

_WDV3_INPUT_SIZE: int = 448


@dataclass
class BranchResult:
    verdict: str
    detections: List[Tuple[str, float, Optional[dict]]] = field(default_factory=list)
    max_score: float = 0.0
    model: str = "wdv3_tagger"
    rating: str = "safe"
    wdv3_explicit: float = 0.0
    anime_rating_r18: float = 0.0
    genital_score: float = 0.0
    breast_score: float = 0.0
    suggestive_score: float = 0.0
    final_score: float = 0.0


class AnimeBranch:
    """
    Anime moderation using WDv3 + deepghs/anime_rating.
    All inference is synchronous; callers wrap scan() in asyncio.to_thread().
    """

    GENITAL_TAGS = GENITAL_TAGS
    BREAST_TAGS = BREAST_TAGS
    SAFE_CONTEXT_TAGS = SAFE_CONTEXT_TAGS
    SUGGESTIVE_TAGS = SUGGESTIVE_TAGS

    def __init__(self, cache_dir: str) -> None:
        self._cache_dir = cache_dir
        self._wdv3_session = None
        self._rating_session = None
        self._tag_names: List[str] = []
        self._rating_indices: Dict[str, int] = {}
        self._wdv3_input_name: Optional[str] = None
        self._wdv3_output_name: Optional[str] = None
        self._rating_input_name: Optional[str] = None
        self._rating_output_name: Optional[str] = None

    def _load_wdv3(self) -> None:
        """Load WDv3 tagger ONNX and selected_tags.csv."""
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        logger.info("[AnimeBranch] Loading SmilingWolf/wd-vit-large-tagger-v3...")

        try:
            model_path = hf_hub_download(
                repo_id="SmilingWolf/wd-vit-large-tagger-v3",
                filename="model.onnx",
                cache_dir=self._cache_dir,
            )
            tags_path = hf_hub_download(
                repo_id="SmilingWolf/wd-vit-large-tagger-v3",
                filename="selected_tags.csv",
                cache_dir=self._cache_dir,
            )
        except Exception as e:
            logger.error("[AnimeBranch] Failed to download WDv3: %s", e)
            raise

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self._wdv3_session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._wdv3_input_name = self._wdv3_session.get_inputs()[0].name
        self._wdv3_output_name = self._wdv3_session.get_outputs()[0].name

        self._tag_names = []
        self._rating_indices = {}
        with open(tags_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tag = row.get("name", row.get("tag_id", ""))
                category = row.get("category", "0")
                self._tag_names.append(tag)
                if str(category) == "9":
                    self._rating_indices[tag] = len(self._tag_names) - 1

        logger.info(
            "[AnimeBranch] WDv3 ready - %d tags, %d rating tags",
            len(self._tag_names),
            len(self._rating_indices),
        )

    def _load_anime_rating(self) -> None:
        """Load deepghs/anime_rating ONNX model."""
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download, list_repo_files

        logger.info("[AnimeBranch] Loading deepghs/anime_rating...")

        try:
            repo_files = list(list_repo_files("deepghs/anime_rating"))
            onnx_files = [f for f in repo_files if f.endswith(".onnx")]
            if not onnx_files:
                raise RuntimeError("No ONNX files found in deepghs/anime_rating")

            preferred = next(
                (f for f in onnx_files if "caformer_s36_plus" in f.lower()),
                next((f for f in onnx_files if "caformer" in f.lower()), onnx_files[0]),
            )
            logger.info("[AnimeBranch] Selected anime rating model file: %s", preferred)
            model_path = hf_hub_download(
                repo_id="deepghs/anime_rating",
                filename=preferred,
                cache_dir=self._cache_dir,
            )
        except Exception as e:
            logger.warning(
                "[AnimeBranch] Could not load deepghs/anime_rating: %s - r18 score will be 0",
                e,
            )
            return

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self._rating_session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._rating_input_name = self._rating_session.get_inputs()[0].name
        self._rating_output_name = self._rating_session.get_outputs()[0].name
        logger.info("[AnimeBranch] deepghs/anime_rating ready")

    def _preprocess_wdv3(self, image: Image.Image) -> np.ndarray:
        """WDv3 expects padded BGR NHWC float input in the 0..255 range."""
        if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
            canvas = Image.new("RGBA", image.size, (255, 255, 255, 255))
            canvas.alpha_composite(image.convert("RGBA"))
            img = canvas.convert("RGB")
        else:
            img = image.convert("RGB")

        width, height = img.size
        max_dim = max(width, height)
        padded = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        padded.paste(img, ((max_dim - width) // 2, (max_dim - height) // 2))
        if max_dim != _WDV3_INPUT_SIZE:
            padded = padded.resize((_WDV3_INPUT_SIZE, _WDV3_INPUT_SIZE), Image.BICUBIC)

        arr = np.asarray(padded, dtype=np.float32)
        arr = arr[:, :, ::-1]
        return np.expand_dims(arr, axis=0).copy()

    def _preprocess_rating(self, image: Image.Image) -> np.ndarray:
        """Preprocess for deepghs/anime_rating."""
        from utils.image_utils import prepare_image_for_onnx

        return prepare_image_for_onnx(
            image,
            target_size=224,
            normalize_imagenet=True,
            to_chw=True,
        )

    def _run_wdv3(self, image: Image.Image) -> Tuple[str, float, Dict[str, float]]:
        """Return (rating_label, explicit_probability, all_tag_scores)."""
        inp = self._preprocess_wdv3(image)
        outputs = self._wdv3_session.run(
            [self._wdv3_output_name],
            {self._wdv3_input_name: inp},
        )
        scores = outputs[0][0]

        rating = "safe"
        best_rating_score = 0.0
        rating_scores: Dict[str, float] = {}

        if self._rating_indices:
            for r_label, r_idx in self._rating_indices.items():
                if r_idx >= len(scores):
                    continue
                clean_label = r_label.replace("rating:", "")
                rating_scores[clean_label] = float(scores[r_idx])
                if scores[r_idx] > best_rating_score:
                    best_rating_score = float(scores[r_idx])
                    rating = r_label
        else:
            rating_map = {
                "rating:general": "general",
                "rating:sensitive": "sensitive",
                "rating:questionable": "questionable",
                "rating:explicit": "explicit",
            }
            for full_tag, short_rating in rating_map.items():
                if full_tag not in self._tag_names:
                    continue
                idx = self._tag_names.index(full_tag)
                rating_scores[short_rating] = float(scores[idx])
                if scores[idx] > best_rating_score:
                    best_rating_score = float(scores[idx])
                    rating = short_rating

        tag_scores = {
            tag: float(scores[i])
            for i, tag in enumerate(self._tag_names)
            if i < len(scores)
        }
        clean_rating = rating.replace("rating:", "")
        wdv3_explicit = rating_scores.get("explicit", tag_scores.get("rating:explicit", 0.0))
        questionable = rating_scores.get("questionable", 0.0)

        logger.debug(
            "[AnimeBranch] WDv3 rating=%s (%.3f), rating_scores=%s, explicit=%.3f, questionable=%.3f",
            clean_rating,
            best_rating_score,
            {k: round(v, 3) for k, v in rating_scores.items()},
            wdv3_explicit,
            questionable,
        )
        return clean_rating, float(wdv3_explicit), tag_scores

    def _run_anime_rating(self, image: Image.Image) -> Tuple[str, float]:
        """Return (label, r18_probability)."""
        if self._rating_session is None:
            try:
                self._load_anime_rating()
            except Exception as e:
                logger.error("[AnimeBranch] Failed to load deepghs/anime_rating dynamically: %s", e)
                return "safe", 0.0

        if self._rating_session is None:
            return "safe", 0.0

        try:
            inp = self._preprocess_rating(image)
            outputs = self._rating_session.run(
                [self._rating_output_name],
                {self._rating_input_name: inp},
            )
            raw = outputs[0][0]

            def softmax(x: np.ndarray) -> np.ndarray:
                e = np.exp(x - np.max(x))
                return e / e.sum()

            probs = softmax(raw) if raw.max() > 1.0 or raw.min() < 0.0 else raw
            best_idx = int(np.argmax(probs))
            labels = ["safe", "r15", "r18"]
            label = labels[best_idx] if best_idx < len(labels) else "safe"
            r18_prob = float(probs[2]) if len(probs) > 2 else 0.0
            rating_scores = {
                labels[i]: round(float(probs[i]), 3)
                for i in range(min(len(labels), len(probs)))
            }

            logger.debug(
                "[AnimeBranch] anime_rating=%s (%.3f), scores=%s, r18=%.3f",
                label,
                float(probs[best_idx]),
                rating_scores,
                r18_prob,
            )
            return label, r18_prob
        except Exception as e:
            logger.warning("[AnimeBranch] anime_rating inference error: %s", e)
            return "safe", 0.0

    def _threshold_score(self, tag_scores: Dict[str, float], thresholds: Dict[str, float], group: str) -> float:
        score = 0.0
        debug_scores: Dict[str, float] = {}
        for tag, threshold in thresholds.items():
            tag_score = tag_scores.get(tag, 0.0)
            if tag_score > 0.0:
                debug_scores[tag] = round(tag_score, 3)
            if tag_score >= threshold:
                score = max(score, tag_score)
                logger.debug(
                    "[AnimeBranch] %s tag '%s' score %.3f exceeded threshold %.3f",
                    group,
                    tag,
                    tag_score,
                    threshold,
                )
        logger.debug("[AnimeBranch] %s tag scores=%s, group_score=%.3f", group, debug_scores, score)
        return score

    def calculate_genital_score(self, tag_scores: Dict[str, float]) -> float:
        return self._threshold_score(tag_scores, self.GENITAL_TAGS, "genital")

    def calculate_breast_score(self, tag_scores: Dict[str, float]) -> float:
        raw_score = self._threshold_score(tag_scores, self.BREAST_TAGS, "breast")
        if raw_score > 0.0:
            return max(raw_score, 0.75)
        return 0.0

    def calculate_suggestive_score(self, tag_scores: Dict[str, float]) -> float:
        return self._threshold_score(tag_scores, self.SUGGESTIVE_TAGS, "suggestive")

    def apply_negative_suppression(
        self,
        genital_score: float,
        breast_score: float,
        suggestive_score: float,
        tag_scores: Dict[str, float],
    ) -> Tuple[float, float, float]:
        """Reduce anatomical confidence when safe context tags dominate."""
        penalty = 0.0
        active_context: Dict[str, float] = {}
        for tag, value in self.SAFE_CONTEXT_TAGS.items():
            score = tag_scores.get(tag, 0.0)
            if score > 0.70:
                penalty += value
                active_context[tag] = round(score, 3)

        suppressed_genital = max(0.0, genital_score - penalty)
        suppressed_breast = max(0.0, breast_score - (penalty * 0.75))
        suppressed_suggestive = max(0.0, suggestive_score - (penalty * 0.50))

        logger.debug(
            "[AnimeBranch] Suppression adjustments: context=%s, penalty=%.3f, genital %.3f->%.3f, breast %.3f->%.3f, suggestive %.3f->%.3f",
            active_context,
            penalty,
            genital_score,
            suppressed_genital,
            breast_score,
            suppressed_breast,
            suggestive_score,
            suppressed_suggestive,
        )
        return suppressed_genital, suppressed_breast, suppressed_suggestive

    def fuse_scores(
        self,
        wdv3_explicit: float,
        anime_rating_r18: float,
        genital_score: float,
        breast_score: float,
    ) -> float:
        final_score = (
            wdv3_explicit * 0.30
            + anime_rating_r18 * 0.25
            + genital_score * 0.30
            + breast_score * 0.15
        )
        logger.debug(
            "[AnimeBranch] Score fusion: explicit=%.3f*0.30 + r18=%.3f*0.25 + genital=%.3f*0.30 + breast=%.3f*0.15 => %.3f",
            wdv3_explicit,
            anime_rating_r18,
            genital_score,
            breast_score,
            final_score,
        )
        return final_score

    def _build_detections(
        self,
        tag_scores: Dict[str, float],
        thresholds: Dict[str, float],
    ) -> List[Tuple[str, float, Optional[dict]]]:
        detections: List[Tuple[str, float, Optional[dict]]] = []
        for tag, threshold in thresholds.items():
            score = tag_scores.get(tag, 0.0)
            if score >= threshold:
                detections.append((tag, score, None))
        return detections

    def scan(self, image: Image.Image) -> BranchResult:
        """
        Return SAFE, SUGGESTIVE, NSFW, or EXPLICIT.

        Explicit/questionable ratings never block by themselves. Escalation requires
        genital or breast evidence; questionable only contributes through model scores.
        """
        if self._wdv3_session is None:
            try:
                self._load_wdv3()
            except Exception as e:
                logger.error("[AnimeBranch] Failed to load WDv3 tagger dynamically: %s", e)
                return BranchResult(verdict="SAFE")

        try:
            rating, wdv3_explicit, tag_scores = self._run_wdv3(image)
            anime_label, anime_rating_r18 = self._run_anime_rating(image)

            raw_genital_score = self.calculate_genital_score(tag_scores)
            raw_breast_score = self.calculate_breast_score(tag_scores)
            raw_suggestive_score = self.calculate_suggestive_score(tag_scores)
            genital_score, breast_score, suggestive_score = self.apply_negative_suppression(
                raw_genital_score,
                raw_breast_score,
                raw_suggestive_score,
                tag_scores,
            )
            final_score = self.fuse_scores(
                wdv3_explicit,
                anime_rating_r18,
                genital_score,
                breast_score,
            )

            if genital_score >= 0.80:
                verdict = "EXPLICIT"
            elif breast_score >= 0.75:
                verdict = "NSFW"
            elif final_score >= 0.65:
                verdict = "SUGGESTIVE"
            else:
                verdict = "SAFE"

            detections = []
            detections.extend(self._build_detections(tag_scores, self.GENITAL_TAGS))
            detections.extend(self._build_detections(tag_scores, self.BREAST_TAGS))
            if verdict == "SUGGESTIVE":
                detections.extend(self._build_detections(tag_scores, self.SUGGESTIVE_TAGS))
            detections.sort(key=lambda item: item[1], reverse=True)

            max_score = max(
                [genital_score, breast_score, suggestive_score, final_score],
                default=0.0,
            )
            logger.info(
                "[AnimeBranch] Scan result - verdict=%s, rating=%s, anime_rating=%s, final=%.3f, genital=%.3f, breast=%.3f, suggestive=%.3f",
                verdict,
                rating,
                anime_label,
                final_score,
                genital_score,
                breast_score,
                suggestive_score,
            )

            return BranchResult(
                verdict=verdict,
                detections=detections,
                max_score=max_score,
                model="anime_score_fusion" if verdict != "SAFE" else "wdv3_tagger",
                rating=rating,
                wdv3_explicit=wdv3_explicit,
                anime_rating_r18=anime_rating_r18,
                genital_score=genital_score,
                breast_score=breast_score,
                suggestive_score=suggestive_score,
                final_score=final_score,
            )
        except Exception as e:
            logger.error("[AnimeBranch] Scan error: %s", e, exc_info=True)
            return BranchResult(verdict="SAFE")
