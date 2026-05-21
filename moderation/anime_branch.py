"""
moderation/anime_branch.py — Stage 2B: Anime/hentai content branch.

Primary: SmilingWolf/wd-vit-large-tagger-v3 (WDv3) ONNX tagger.
Secondary: deepghs/anime_rating ONNX (consulted for borderline questionable cases).

Block policy: rating == "explicit" AND any GENITAL_TAG score > 0.50.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ── Explicit tag vocabulary ────────────────────────────────────────────────────
EXPLICIT_TAGS: Set[str] = {
    "penis",
    "vagina",
    "pussy",
    "genitals",
    "testicles",
    "erection",
    "phallus",
    "nipples",
    "bare_breasts",
    "breasts_out",
    "anus",
}

EXPLICIT_TAG_THRESHOLD: float = 0.50
EXPLICIT_TAG_REVIEW_LOW: float = 0.35  # borderline window for secondary check

# WDv3 input size
_WDV3_INPUT_SIZE: int = 448

# Rating label constants (WDv3 uses these rating indices in selected_tags.csv)
_RATING_LABELS: List[str] = ["general", "sensitive", "questionable", "explicit"]


@dataclass
class BranchResult:
    verdict: str                                         # "BLOCK" | "REVIEW" | "SAFE"
    detections: List[Tuple[str, float, Optional[dict]]] = field(default_factory=list)
    max_score: float = 0.0
    model: str = "wdv3_tagger"
    rating: str = "safe"


class AnimeBranch:
    """
    Anime/hentai content detector using WD-ViT-v3 tagger + deepghs/anime_rating.

    All inference is synchronous — wrap calls in asyncio.to_thread().
    """

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
        """Load deepghs/anime_rating ONNX secondary model."""
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
                "[AnimeBranch] Could not load deepghs/anime_rating: %s — secondary check disabled", e
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

    def _run_wdv3(self, image: Image.Image) -> Tuple[str, Dict[str, float]]:
        """
        Run WDv3 inference.
        Returns (rating_label, {genital_tag: score}) for tags above 0.35.
        """
        inp = self._preprocess_wdv3(image)
        outputs = self._wdv3_session.run([self._wdv3_output_name], {self._wdv3_input_name: inp})
        scores = outputs[0][0]  # shape: (num_tags,)

        # Extract rating
        rating = "safe"
        best_rating_score = 0.0

        if self._rating_indices:
            for r_label, r_idx in self._rating_indices.items():
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
                    if scores[idx] > best_rating_score:
                        best_rating_score = scores[idx]
                        rating = short_rating

        # Extract explicit tag scores
        explicit_scores: Dict[str, float] = {}
        for i, tag in enumerate(self._tag_names):
            if tag.lower() in EXPLICIT_TAGS and i < len(scores):
                s = float(scores[i])
                if s > EXPLICIT_TAG_REVIEW_LOW:
                    explicit_scores[tag] = s
                    logger.debug("[AnimeBranch] Tag %s = %.3f", tag, s)

        logger.debug("[AnimeBranch] WDv3 rating=%s (%.3f)", rating, best_rating_score)
        return rating, explicit_scores

    def _run_anime_rating(self, image: Image.Image) -> str:
        """
        Run deepghs/anime_rating secondary classifier.
        Returns "safe", "r15", or "r18".
        """
        if self._rating_session is None:
            try:
                self._load_anime_rating()
            except Exception as e:
                logger.error("[AnimeBranch] Failed to load deepghs/anime_rating dynamically: %s", e)
                return "safe"

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
            logger.debug("[AnimeBranch] anime_rating → %s (%.3f)", label, float(probs[best_idx]))
            return label
        except Exception as e:
            logger.warning("[AnimeBranch] anime_rating inference error: %s", e)
            return "safe"

    # ── Public scan ───────────────────────────────────────────────────────────

    def scan(self, image: Image.Image) -> BranchResult:
        """
        Scan an anime/illustration image for explicit content (genitals and exposed breasts).

        Decision logic:
        - BLOCK: WDv3 rating in ("explicit", "questionable") AND any EXPLICIT_TAGS > 0.50
        - REVIEW: WDv3 rating == "explicit" but no explicit tags > 0.50
                  OR WDv3 "questionable" with explicit tags 0.35–0.50 and anime_rating == "r18"
        - SAFE:  Anything else (sensitive/questionable alone does NOT trigger)
        """
        if self._wdv3_session is None:
            try:
                self._load_wdv3()
            except Exception as e:
                logger.error("[AnimeBranch] Failed to load WDv3 tagger dynamically: %s", e)
                return BranchResult(verdict="SAFE")

        try:
            rating, explicit_scores = self._run_wdv3(image)

            max_score = max(explicit_scores.values()) if explicit_scores else 0.0
            detections = [(tag, score, None) for tag, score in explicit_scores.items()]

            # 1. BLOCK: rating in ("explicit", "questionable") and any explicit tag > 0.50
            if rating in ("explicit", "questionable"):
                high_confidence = {t: s for t, s in explicit_scores.items() if s > EXPLICIT_TAG_THRESHOLD}
                if high_confidence:
                    logger.info(
                        "[AnimeBranch] BLOCK — rating=%s + explicit tags: %s",
                        rating,
                        {t: f"{s:.3f}" for t, s in high_confidence.items()},
                    )
                    return BranchResult(
                        verdict="BLOCK",
                        detections=detections,
                        max_score=max_score,
                        model="wdv3_tagger",
                        rating=rating,
                    )

            # 2. REVIEW / other verdicts
            if rating == "explicit":
                # Explicit rating but no strong explicit tags -> REVIEW
                logger.info("[AnimeBranch] REVIEW — explicit rating, borderline explicit tags")
                return BranchResult(
                    verdict="REVIEW",
                    detections=detections,
                    max_score=max_score,
                    model="wdv3_tagger",
                    rating=rating,
                )

            elif rating == "questionable":
                # Only consult secondary if there are borderline explicit tags
                borderline = {
                    t: s for t, s in explicit_scores.items()
                    if EXPLICIT_TAG_REVIEW_LOW <= s <= EXPLICIT_TAG_THRESHOLD
                }
                if borderline:
                    secondary = self._run_anime_rating(image)
                    if secondary == "r18":
                        logger.info(
                            "[AnimeBranch] REVIEW — questionable + borderline tags + r18 secondary"
                        )
                        return BranchResult(
                            verdict="REVIEW",
                            detections=detections,
                            max_score=max_score,
                            model="anime_rating",
                            rating=rating,
                        )

            # All other cases: SAFE
            logger.debug("[AnimeBranch] SAFE — rating=%s, max_explicit=%.3f", rating, max_score)
            return BranchResult(verdict="SAFE", max_score=max_score, rating=rating)

        except Exception as e:
            logger.error("[AnimeBranch] Scan error: %s", e, exc_info=True)
            return BranchResult(verdict="SAFE")
