import asyncio
import os
import shutil
import logging
from config import config
import moderation.pipeline

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

# Setup mock download
original_download = moderation.pipeline._download_attachment

async def mock_download_attachment(url: str, dest_path: str, max_size_mb: int) -> None:
    if url.startswith("local:"):
        local_path = url[6:]
        shutil.copy(local_path, dest_path)
        print(f"[Mock Download] Copied {local_path} -> {dest_path}")
    else:
        await original_download(url, dest_path, max_size_mb)

moderation.pipeline._download_attachment = mock_download_attachment

async def main():
    print("==================================================")
    print("RUNNING PIPELINE DIAGNOSTIC VERIFICATION")
    print("==================================================")

    test_assets = {
        "test_hentai.png": "local:test_hentai.png",
        "test_real_breasts.png": "local:test_real_breasts.png",
    }

    for name, local_url in test_assets.items():
        if not os.path.exists(name):
            print(f"Skipping {name} as it does not exist locally.")
            continue

        print(f"\nScanning {name}...")
        try:
            # We bypass_prefilter=True to make sure we run both gatekeeper and stage 2 fully
            result = await moderation.pipeline.scan_attachment(
                attachment_url=local_url,
                config=config,
                bypass_prefilter=True
            )
            print(f"Result for {name}:")
            print(f"  Verdict: {result.verdict}")
            print(f"  Reason: {result.reason}")
            print(f"  Branch: {result.branch}")
            print(f"  Model: {result.model}")
            print("  Pipeline Steps Trace:")
            for step in result.pipeline_steps:
                print(f"--- Step ---\n{step}")
        except Exception as e:
            print(f"Error scanning {name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
