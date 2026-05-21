"""
moderation/anime_branch.py — Stage 2B: Anime/hentai content branch.

Primary: SmilingWolf/wd-vit-large-tagger-v3 (WDv3) ONNX tagger.
Secondary: deepghs/anime_rating ONNX (consulted for borderline questionable cases).

Block policy: fused model scores only count when strong genital evidence is present.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ── Explicit tag vocabulary & Constants ───────────────────────────────────────
EXPLICIT_TAGS: Set[str] = {
    "penis",
    "vagina",
    "pussy",
    "genitals",
    "testicles",
    "erection",
    "phallus",
    "anus",
}

GENITAL_THRESHOLDS: Dict[str, float] = {
    "penis": 0.72,
    "vagina": 0.75,
    "pussy": 0.75,
    "genitals": 0.80,
    "testicles": 0.72,
    "erection": 0.70,
    "anus": 0.72,
    "phallus": 0.72,
}

SAFE_CONTEXT_TAGS: Dict[str, float] = {
    "bikini": 0.15,
    "swimsuit": 0.15,
    "underwear": 0.12,
    "bra": 0.10,
    "panties": 0.12,
    "lingerie": 0.10,
    "cameltoe": 0.08,
}

# WDv3 input size
_WDV3_INPUT_SIZE: int = 448


@dataclass
class BranchResult:
    verdict: str                                         # "BLOCK" | "REVIEW" | "SAFE"
    detections: List[Tuple[str, float, Optional[dict]]] = field(default_factory=list)
    max_score: float = 0.0
    model: str = "wdv3_tagger"
    rating: str = "safe"
    wdv3_explicit: float = 0.0
    anime_rating_r18: float = 0.0
    genital_score: float = 0.0
    final_score: float = 0.0


class AnimeBranch:
    """
    Anime/hentai content detector using WD-ViT-v3 tagger + deepghs/anime_rating.
    All inference is synchronous — wrap calls in asyncio.to_thread().
    """

    EXPLICIT_TAGS = EXPLICIT_TAGS
    GENITAL_THRESHOLDS = GENITAL_THRESHOLDS
    SAFE_CONTEXT_TAGS = SAFE_CONTEXT_TAGS

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

    # ── Loaders ───────────────────────────────────────────────────────────────

    def _load_wdv3(self) -> None:
        """Load WDv3 tagger ONNX and selected_tags.csv."""
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        logger.info("[AnimeBranch] Loading SmilingWolf/wd-vit-large-tagger-v3…")

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

        # Parse selected_tags.csv to get tag names
        self._tag_names = []
        self._rating_indices = {}
        with open(tags_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tag = row.get("name", row.get("tag_id", ""))
                category = row.get("category", "0")
                self._tag_names.append(tag)
                # Category 9 = rating tags in WDv3 schema
                if str(category) == "9":
                    idx = len(self._tag_names) - 1
                    self._rating_indices[tag] = idx

        logger.info(
            "[AnimeBranch] WDv3 ready — %d tags, %d rating tags",
            len(self._tag_names),
            len(self._rating_indices),
        )

    def _load_anime_rating(self) -> None:
        """Load deepghs/anime_rating ONNX primary model."""
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
                next(
                    (f for f in onnx_files if "caformer" in f.lower()),
                    onnx_files[0]
                )
            )
            
            logger.info("[AnimeBranch] Selected anime rating model file: %s", preferred)
            model_path = hf_hub_download(
                repo_id="deepghs/anime_rating",
                filename=preferred,
                cache_dir=self._cache_dir,
            )
        except Exception as e:
            logger.warning(
                "[AnimeBranch] Could not load deepghs/anime_rating: %s — evaluation might skip secondary probability", e
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

    # ── Preprocessing ─────────────────────────────────────────────────────────

    def _preprocess_wdv3(self, image: Image.Image) -> np.ndarray:
        """
        WDv3 expects: RGB, resized to 448×448, values in [0,1], channels-last (NHWC).
        Note: WDv3 uses BGR order internally — convert RGB→BGR before returning.
        """
        from utils.image_utils import prepare_image_for_onnx
        return prepare_image_for_onnx(
            image,
            target_size=_WDV3_INPUT_SIZE,
            to_bgr=True,
            to_chw=False,
            normalize_imagenet=False,
        )

    def _preprocess_rating(self, image: Image.Image) -> np.ndarray:
        """Preprocess for deepghs/anime_rating (standard ImageNet normalization, NCHW)."""
        from utils.image_utils import prepare_image_for_onnx
        return prepare_image_for_onnx(
            image,
            target_size=224,
            normalize_imagenet=True,
            to_chw=True,
        )

    # ── Inference ─────────────────────────────────────────────────────────────

    def _run_wdv3(self, image: Image.Image) -> Tuple[str, float, Dict[str, float]]:
        """
        Run WDv3 inference.
        Returns (rating_label, wdv3_explicit, tag_scores) where tag_scores is a Dict of all tag names and their scores.
        """
        inp = self._preprocess_wdv3(image)
        outputs = self._wdv3_session.run([self._wdv3_output_name], {self._wdv3_input_name: inp})
        scores = outputs[0][0]  # shape: (num_tags,)

        # Extract rating
        rating = "safe"
        best_rating_score = 0.0
        rating_scores: Dict[str, float] = {}

        if self._rating_indices:
            for r_label, r_idx in self._rating_indices.items():
                if r_idx < len(scores):
                    rating_scores[r_label.replace("rating:", "")] = float(scores[r_idx])
                if r_idx < len(scores) and scores[r_idx] > best_rating_score:
                    best_rating_score = scores[r_idx]
                    rating = r_label
        else:
            # Fallback: look for rating tag names in the tag list
            rating_map = {
                "rating:general": "general",
                "rating:sensitive": "sensitive",
                "rating:questionable": "questionable",
                "rating:explicit": "explicit",
            }
            for full_tag, short_rating in rating_map.items():
                if full_tag in self._tag_names:
                    idx = self._tag_names.index(full_tag)
                    rating_scores[short_rating] = float(scores[idx])
                    if scores[idx] > best_rating_score:
                        best_rating_score = scores[idx]
                        rating = short_rating

        # Normalize/clean rating label
        clean_rating = rating.replace("rating:", "")

        # Extract wdv3_explicit score
        wdv3_explicit = 0.0
        if self._rating_indices:
            for r_label, r_idx in self._rating_indices.items():
                if "explicit" in r_label.lower() and r_idx < len(scores):
                    wdv3_explicit = float(scores[r_idx])
        
        if wdv3_explicit == 0.0:
            for full_tag in ["rating:explicit", "explicit"]:
                if full_tag in self._tag_names:
                    idx = self._tag_names.index(full_tag)
                    wdv3_explicit = float(scores[idx])
                    break

        # Extract all tag scores (in a dict)
        tag_scores: Dict[str, float] = {}
        for i, tag in enumerate(self._tag_names):
            if i < len(scores):
                tag_scores[tag] = float(scores[i])

        logger.debug(
            "[AnimeBranch] WDv3 rating=%s (%.3f), rating_scores=%s, explicit_score=%.3f",
            clean_rating,
            best_rating_score,
            {k: round(v, 3) for k, v in rating_scores.items()},
            wdv3_explicit,
        )
        return clean_rating, wdv3_explicit, tag_scores

    def _run_anime_rating(self, image: Image.Image) -> Tuple[str, float]:
        """
        Run deepghs/anime_rating classifier.
        Returns (label, r18_prob).
        """
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
                [self._rating_output_name], {self._rating_input_name: inp}
            )
            raw = outputs[0][0]

            def softmax(x: np.ndarray) -> np.ndarray:
                e = np.exp(x - np.max(x))
                return e / e.sum()

            probs = softmax(raw) if raw.max() > 1.0 or raw.min() < 0.0 else raw
            best_idx = int(np.argmax(probs))

            # deepghs/anime_rating labels: [safe, r15, r18]
            rating_labels = ["safe", "r15", "r18"]
            label = rating_labels[best_idx] if best_idx < len(rating_labels) else "safe"
            
            # The r18 probability is probs[2]
            r18_prob = float(probs[2]) if len(probs) > 2 else 0.0
            rating_scores = {
                rating_labels[i]: round(float(probs[i]), 3)
                for i in range(min(len(rating_labels), len(probs)))
            }
            
            logger.debug(
                "[AnimeBranch] anime_rating -> %s (%.3f), scores=%s, r18_prob=%.3f",
                label,
                float(probs[best_idx]),
                rating_scores,
                r18_prob,
            )
            return label, r18_prob
        except Exception as e:
            logger.warning("[AnimeBranch] anime_rating inference error: %s", e)
            return "safe", 0.0

    # ── Score Fusion Helpers ──────────────────────────────────────────────────

    def calculate_genital_score(self, tag_scores: Dict[str, float]) -> float:
        """
        Calculates the raw genital score as the maximum score among genital tags that
        cross their respective GENITAL_THRESHOLDS. If no genital tag crosses its
        threshold, returns 0.0 to eliminate low-confidence noise.
        """
        genital_score = 0.0
        for tag, threshold in self.GENITAL_THRESHOLDS.items():
            score = tag_scores.get(tag, 0.0)
            if score >= threshold:
                genital_score = max(genital_score, score)
                logger.debug(
                    "[AnimeBranch] Genital tag '%s' score %.3f exceeds threshold %.3f",
                    tag,
                    score,
                    threshold,
                )

        if genital_score == 0.0:
            debug_scores = {
                tag: round(tag_scores.get(tag, 0.0), 3)
                for tag in sorted(self.GENITAL_THRESHOLDS)
                if tag_scores.get(tag, 0.0) > 0.0
            }
            logger.debug(
                "[AnimeBranch] No genital tags exceeded thresholds; genital tag scores=%s",
                debug_scores,
            )
        return genital_score

    def apply_negative_suppression(self, genital_score: float, wdv3_scores: Dict[str, float]) -> float:
        """
        If any safe context tag is strongly present (>0.70), reduce the genital score by its penalty.
        Caps the final result at 0.0.
        """
        original_score = genital_score
        penalty = 0.0
        for tag, val in self.SAFE_CONTEXT_TAGS.items():
            score = wdv3_scores.get(tag, 0.0)
            if score > 0.70:
                penalty += val
                logger.debug("[AnimeBranch] Negative suppression: tag '%s' score = %.3f, penalty = %.3f", tag, score, val)
        
        suppressed_score = max(0.0, genital_score - penalty)
        logger.debug(
            "[AnimeBranch] Suppression adjustments: raw_genital_score=%.3f, penalty=%.3f, suppressed_genital_score=%.3f",
            original_score,
            penalty,
            suppressed_score,
        )
        return suppressed_score

    def fuse_scores(self, wdv3_explicit: float, anime_rating_r18: float, genital_score: float) -> float:
        """
        Fuses the scores using the exact weighted score formula:
        final_score = wdv3_explicit * 0.35 + anime_rating_r18 * 0.30 + genital_score * 0.35
        """
        final_score = (wdv3_explicit * 0.35) + (anime_rating_r18 * 0.30) + (genital_score * 0.35)
        logger.debug("[AnimeBranch] Score Fusion: wdv3_explicit=%.3f * 0.35, anime_rating_r18=%.3f * 0.30, genital_score=%.3f * 0.35 -> final_score=%.3f",
                     wdv3_explicit, anime_rating_r18, genital_score, final_score)
        return final_score

    # ── Public scan ───────────────────────────────────────────────────────────

    def scan(self, image: Image.Image) -> BranchResult:
        """
        Scan an anime/illustration image using premium weighted score fusion.
        
        Formula:
          final_score = wdv3_explicit * 0.35 + anime_rating_r18 * 0.30 + genital_score * 0.35
          
        Decision logic:
          - BLOCK: final_score >= 0.80
          - REVIEW: final_score >= 0.65
          - SAFE: final_score < 0.65
        """
        if self._wdv3_session is None:
            try:
                self._load_wdv3()
            except Exception as e:
                logger.error("[AnimeBranch] Failed to load WDv3 tagger dynamically: %s", e)
                return BranchResult(verdict="SAFE")

        try:
            # 1. Run WDv3 Tagger
            rating, wdv3_explicit, wdv3_scores = self._run_wdv3(image)

            # 2. Run anime_rating on all scans
            anime_label, anime_rating_r18 = self._run_anime_rating(image)

            # 3. Calculate raw genital score. Ratings alone never decide a verdict.
            genital_tag_scores: Dict[str, float] = {
                tag: wdv3_scores.get(tag, 0.0)
                for tag in self.GENITAL_THRESHOLDS
            }
            raw_genital_score = self.calculate_genital_score(wdv3_scores)

            # 4. Apply negative suppression
            genital_score = self.apply_negative_suppression(raw_genital_score, wdv3_scores)

            # 5. Perform weighted score fusion
            final_score = self.fuse_scores(wdv3_explicit, anime_rating_r18, genital_score)

            # Build detections only from genital tags that cross their tag-specific threshold.
            detections = []
            for tag, score in genital_tag_scores.items():
                if score >= self.GENITAL_THRESHOLDS[tag]:
                    detections.append((tag, score, None))
            detections.sort(key=lambda x: x[1], reverse=True)

            # Determine verdict. Strong genital evidence is required for BLOCK/REVIEW;
            # WDv3 explicit/questionable ratings alone are broad classifiers, not final detectors.
            if genital_score <= 0.0:
                verdict = "SAFE"
            elif final_score >= 0.80:
                verdict = "BLOCK"
            elif final_score >= 0.65:
                verdict = "REVIEW"
            else:
                verdict = "SAFE"

            model_used = "score_fusion" if verdict != "SAFE" else "wdv3_tagger"

            logger.info(
                "[AnimeBranch] Scan Result - verdict=%s, rating=%s, anime_rating=%s, final_score=%.3f (wdv3_explicit=%.3f, r18=%.3f, genital=%.3f)",
                verdict,
                rating,
                anime_label,
                final_score,
                wdv3_explicit,
                anime_rating_r18,
                genital_score
            )

            return BranchResult(
                verdict=verdict,
                detections=detections,
                max_score=max(genital_tag_scores.values()) if genital_tag_scores else 0.0,
                model=model_used,
                rating=rating,
                wdv3_explicit=wdv3_explicit,
                anime_rating_r18=anime_rating_r18,
                genital_score=genital_score,
                final_score=final_score
            )

        except Exception as e:
            logger.error("[AnimeBranch] Scan error: %s", e, exc_info=True)
            return BranchResult(verdict="SAFE")
