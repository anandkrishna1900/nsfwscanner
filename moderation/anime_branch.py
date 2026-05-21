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

# ── Genital tag vocabulary ────────────────────────────────────────────────────
GENITAL_TAGS: Set[str] = {
    "penis",
    "vagina",
    "pussy",
    "genitals",
    "testicles",
    "erection",
    "phallus",
}

GENITAL_TAG_THRESHOLD: float = 0.50
GENITAL_TAG_REVIEW_LOW: float = 0.35  # borderline window for secondary check

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
        self._load_wdv3()
        self._load_anime_rating()

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
        img = image.convert("RGB").resize((_WDV3_INPUT_SIZE, _WDV3_INPUT_SIZE), Image.BICUBIC)
        arr = np.array(img, dtype=np.float32) / 255.0  # HWC RGB, [0,1]
        # WDv3 was trained with BGR input
        arr = arr[:, :, ::-1].copy()                   # RGB → BGR
        # Pad to square (model expects specific aspect ratio)
        arr = np.expand_dims(arr, axis=0)              # NHWC
        return arr

    def _preprocess_rating(self, image: Image.Image) -> np.ndarray:
        """Preprocess for deepghs/anime_rating (standard ImageNet normalization, NCHW)."""
        img = image.convert("RGB").resize((224, 224), Image.BICUBIC)
        arr = np.array(img, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        arr = arr.transpose(2, 0, 1)
        return np.expand_dims(arr, axis=0)

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

        # Extract genital tag scores
        genital_scores: Dict[str, float] = {}
        for i, tag in enumerate(self._tag_names):
            if tag.lower() in GENITAL_TAGS and i < len(scores):
                s = float(scores[i])
                if s > GENITAL_TAG_REVIEW_LOW:
                    genital_scores[tag] = s
                    logger.debug("[AnimeBranch] Tag %s = %.3f", tag, s)

        logger.debug("[AnimeBranch] WDv3 rating=%s (%.3f)", rating, best_rating_score)
        return rating, genital_scores

    def _run_anime_rating(self, image: Image.Image) -> str:
        """
        Run deepghs/anime_rating secondary classifier.
        Returns "safe", "r15", or "r18".
        """
        if self._rating_session is None:
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
        Scan an anime/illustration image for explicit genital content.

        Decision logic:
        - BLOCK: WDv3 rating == "explicit" AND any GENITAL_TAG > 0.50
        - REVIEW: WDv3 rating == "explicit" but no genital tags > 0.50
                  OR WDv3 "questionable" with genital tags 0.35–0.50 and anime_rating == "r18"
        - SAFE:  Anything else (sensitive/questionable alone does NOT trigger)
        """
        if self._wdv3_session is None:
            logger.error("[AnimeBranch] WDv3 session not loaded")
            return BranchResult(verdict="SAFE")

        try:
            rating, genital_scores = self._run_wdv3(image)

            max_score = max(genital_scores.values()) if genital_scores else 0.0
            detections = [(tag, score, None) for tag, score in genital_scores.items()]

            if rating == "explicit":
                high_confidence = {t: s for t, s in genital_scores.items() if s > GENITAL_TAG_THRESHOLD}
                if high_confidence:
                    # BLOCK
                    logger.info(
                        "[AnimeBranch] BLOCK — explicit + genital tags: %s",
                        {t: f"{s:.3f}" for t, s in high_confidence.items()},
                    )
                    return BranchResult(
                        verdict="BLOCK",
                        detections=detections,
                        max_score=max_score,
                        model="wdv3_tagger",
                        rating=rating,
                    )
                else:
                    # Explicit rating but no strong genital tags → REVIEW
                    logger.info("[AnimeBranch] REVIEW — explicit rating, borderline genital tags")
                    return BranchResult(
                        verdict="REVIEW",
                        detections=detections,
                        max_score=max_score,
                        model="wdv3_tagger",
                        rating=rating,
                    )

            elif rating == "questionable":
                # Only consult secondary if there are borderline genital tags
                borderline = {
                    t: s for t, s in genital_scores.items()
                    if GENITAL_TAG_REVIEW_LOW <= s <= GENITAL_TAG_THRESHOLD
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
            logger.debug("[AnimeBranch] SAFE — rating=%s, max_genital=%.3f", rating, max_score)
            return BranchResult(verdict="SAFE", max_score=max_score, rating=rating)

        except Exception as e:
            logger.error("[AnimeBranch] Scan error: %s", e, exc_info=True)
            return BranchResult(verdict="SAFE")
