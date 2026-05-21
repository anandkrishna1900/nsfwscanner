"""
scripts/download_models.py — Pre-download all AI models to MODEL_CACHE_DIR.

Run this before starting the bot for the first time:
    python scripts/download_models.py

Downloads:
  1. AdamCodd/vit-base-nsfw-detector (ONNX)         — ~330 MB
  2. deepghs/anime_real_cls (ONNX)                  — ~90 MB
  3. SmilingWolf/wd-vit-large-tagger-v3 (ONNX)      — ~650 MB
  4. deepghs/anime_rating (ONNX)                     — ~90 MB
  5. NudeNet (auto-downloads on first NudeDetector() init)

All models are cached to MODEL_CACHE_DIR (from .env, default ./models).
"""

import os
import sys
import time

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "./models")
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)


def banner(text: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")


def check_mark(text: str) -> None:
    print(f"  [OK] {text}")


def fail_mark(text: str) -> None:
    print(f"  [FAIL] {text}")


def download_onnx(repo_id: str, filename: str, label: str) -> bool:
    """Download an ONNX model file from HuggingFace Hub."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError

    print(f"\n  Downloading {label}...")
    print(f"    Repo: {repo_id}")
    print(f"    File: {filename}")
    print(f"    Cache: {MODEL_CACHE_DIR}")

    t0 = time.time()
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=MODEL_CACHE_DIR,
            local_files_only=False,
        )
        elapsed = time.time() - t0
        size_mb = os.path.getsize(path) / (1024 * 1024)
        check_mark(f"Done in {elapsed:.1f}s - {size_mb:.0f} MB -> {path}")
        return True
    except Exception as e:
        fail_mark(f"Failed: {e}")
        return False


def download_extra_files(repo_id: str, filenames: list[str], label: str) -> bool:
    """Download additional files (e.g. CSV, config) from HuggingFace Hub."""
    from huggingface_hub import hf_hub_download

    all_ok = True
    for fname in filenames:
        print(f"  Downloading {label} -> {fname}...")
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=fname,
                cache_dir=MODEL_CACHE_DIR,
                local_files_only=False,
            )
            check_mark(f"{fname} -> {path}")
        except Exception as e:
            fail_mark(f"{fname}: {e}")
            all_ok = False
    return all_ok


def init_nudenet() -> bool:
    """Trigger NudeDetector initialization so it downloads its model."""
    print("\n  Initializing NudeDetector (downloads ~5 MB model)...")
    try:
        from nudenet import NudeDetector
        _ = NudeDetector()
        check_mark("NudeDetector initialized successfully")
        return True
    except Exception as e:
        fail_mark(f"NudeDetector init failed: {e}")
        return False


def verify_onnx(repo_id: str, filename: str, label: str) -> None:
    """Quick verify that the downloaded ONNX file is a valid InferenceSession."""
    from huggingface_hub import hf_hub_download
    import onnxruntime as ort

    print(f"  Verifying {label}...")
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=MODEL_CACHE_DIR,
            local_files_only=True,
        )
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        inp_shape = sess.get_inputs()[0].shape
        check_mark(f"Valid ONNX - input shape: {inp_shape}")
    except Exception as e:
        fail_mark(f"Verification failed: {e}")


def main() -> None:
    print("\n" + "=" * 60)
    print("  NSFW Bot — Model Download Script")
    print("  Cache directory:", os.path.abspath(MODEL_CACHE_DIR))
    print("=" * 60)

    results: dict[str, bool] = {}

    # ── 1. AdamCodd/vit-base-nsfw-detector ───────────────────────────────────
    banner("Model 1/4: AdamCodd/vit-base-nsfw-detector (Pre-filter)")
    ok = download_onnx(
        repo_id="AdamCodd/vit-base-nsfw-detector",
        filename="model.onnx",
        label="AdamCodd ViT NSFW detector",
    )
    if not ok:
        # Try alternate path
        ok = download_onnx(
            repo_id="AdamCodd/vit-base-nsfw-detector",
            filename="onnx/model.onnx",
            label="AdamCodd ViT NSFW detector (alternate path)",
        )
    results["prefilter"] = ok

    # ── 2. deepghs/anime_real_cls ─────────────────────────────────────────────
    banner("Model 2/4: deepghs/anime_real_cls (Gatekeeper)")
    try:
        from huggingface_hub import list_repo_files
        repo_files = list(list_repo_files("deepghs/anime_real_cls"))
        onnx_files = [f for f in repo_files if f.endswith(".onnx")]
        if onnx_files:
            preferred = next(
                (f for f in onnx_files if "caformer" in f.lower()),
                onnx_files[0],
            )
            ok = download_onnx(
                repo_id="deepghs/anime_real_cls",
                filename=preferred,
                label=f"deepghs/anime_real_cls ({preferred})",
            )
        else:
            print("  [FAIL] No ONNX files found in deepghs/anime_real_cls repository.")
            ok = False
    except Exception as e:
        print(f"  [FAIL] Failed to list repo files for deepghs/anime_real_cls: {e}")
        ok = False
    results["gatekeeper"] = ok

    # ── 3. SmilingWolf/wd-vit-large-tagger-v3 ────────────────────────────────
    banner("Model 3/4: SmilingWolf/wd-vit-large-tagger-v3 (Anime Branch)")
    ok = download_onnx(
        repo_id="SmilingWolf/wd-vit-large-tagger-v3",
        filename="model.onnx",
        label="WD-ViT-v3 tagger",
    )
    ok_csv = download_extra_files(
        repo_id="SmilingWolf/wd-vit-large-tagger-v3",
        filenames=["selected_tags.csv"],
        label="WD-ViT-v3",
    )
    results["wdv3"] = ok and ok_csv

    # ── 4. deepghs/anime_rating ───────────────────────────────────────────────
    banner("Model 4/4: deepghs/anime_rating (Anime Rating Secondary)")
    try:
        from huggingface_hub import list_repo_files
        repo_files = list(list_repo_files("deepghs/anime_rating"))
        onnx_files = [f for f in repo_files if f.endswith(".onnx")]
        if onnx_files:
            preferred = next(
                (f for f in onnx_files if "caformer_s36_plus" in f.lower()),
                next(
                    (f for f in onnx_files if "caformer" in f.lower()),
                    onnx_files[0],
                )
            )
            ok = download_onnx(
                repo_id="deepghs/anime_rating",
                filename=preferred,
                label=f"deepghs/anime_rating ({preferred})",
            )
        else:
            print("  [FAIL] No ONNX files found in deepghs/anime_rating repository.")
            ok = False
    except Exception as e:
        print(f"  [FAIL] Failed to list repo files for deepghs/anime_rating: {e}")
        ok = False
    results["anime_rating"] = ok

    # ── 5. NudeNet ────────────────────────────────────────────────────────────
    banner("Model 5/5: NudeNet (Real Branch)")
    results["nudenet"] = init_nudenet()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Download Summary")
    print("=" * 60)
    all_ok = True
    for name, success in results.items():
        status = "[OK]  " if success else "[FAIL]"
        print(f"  {status}  {name}")
        if not success:
            all_ok = False

    if all_ok:
        print("\nAll models downloaded successfully!")
        print("   You can now start the bot with: python main.py")
    else:
        print("\nSome models failed to download.")
        print("   Check your internet connection and try again.")
        print("   The bot will attempt to download missing models on first use.")

    print()


if __name__ == "__main__":
    main()
