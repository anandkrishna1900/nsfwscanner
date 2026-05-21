# AGENTS.md — NSFW Moderation Discord Bot
# Antigravity reads this file automatically. Follow every rule here at all times.

## Project Overview
A locally-hosted Discord bot that automatically scans images, GIFs, and videos posted in
configured channels for explicit genitals and exposed breasts/nudity. It uses a council of specialized AI models
routed by a content-type classifier. No external APIs are used — all inference runs locally.

## Hardware Constraints — CRITICAL
- GPU: NVIDIA RTX 2050 (4GB GDDR6 VRAM) — Ampere architecture, CUDA 11.x+
- RAM: 16GB system RAM
- Never load more than ONE PyTorch GPU model at a time
- All ONNX models must use CPUExecutionProvider only
- PyTorch models use device="cuda" with torch.cuda.empty_cache() called after each inference
- Model loading is lazy — only load a model when needed, unload immediately after
- Never pre-load all models at startup — only load the ONNX pre-filter at startup

## Tech Stack
- Language: Python 3.11+
- Discord library: discord.py v2.x (use app_commands for slash commands)
- ML inference: torch 2.x, transformers 4.x, onnxruntime 1.17+
- Video/GIF frames: opencv-python, imageio[ffmpeg]
- Model downloads: huggingface_hub
- Config: python-dotenv (.env file for Discord token and settings)
- Dependency management: pip with requirements.txt
- No Docker, no external APIs, no cloud services

## Project Structure
```
nsfw-bot/
├── AGENTS.md
├── .env                  # Discord token, config — never commit
├── .env.example          # Committed template
├── requirements.txt
├── main.py               # Bot entry point
├── config.py             # Settings dataclass loaded from .env
├── bot/
│   ├── __init__.py
│   ├── cogs/
│   │   ├── moderation.py     # Main moderation cog — listens for messages
│   │   └── admin.py          # Slash commands for admins (/nsfw-enable, /nsfw-disable, etc)
│   └── events.py             # on_ready, on_error handlers
├── moderation/
│   ├── __init__.py
│   ├── pipeline.py           # Main orchestrator — runs the full council pipeline
│   ├── gatekeeper.py         # anime_real_cls router + CLIP fallback
│   ├── real_branch.py        # NudeNet + AdamCodd ONNX for real/photo content
│   ├── anime_branch.py       # WDv3 + deepghs/anime_rating for anime/hentai
│   ├── frame_extractor.py    # GIF and video → sampled frames
│   └── models/
│       ├── __init__.py
│       └── loader.py         # Lazy model loader with GPU memory management
└── utils/
    ├── __init__.py
    ├── image_utils.py        # Download Discord attachment → temp file
    └── logger.py             # Structured logging
```

## Model Council — Full Specification

### Stage 0: Pre-filter (CPU, always loaded)
- Model: AdamCodd/vit-base-nsfw-detector (ONNX export)
- HuggingFace: AdamCodd/vit-base-nsfw-detector
- Purpose: Fast binary safe/nsfw score. If score < 0.25, skip all further processing.
- Runtime: onnxruntime CPUExecutionProvider
- Threshold: 0.25 (very permissive — only skip obviously safe content)

### Stage 1: Gatekeeper / Router (CPU)
- Primary: deepghs/anime_real_cls
- HuggingFace: deepghs/anime_real_cls
- Purpose: Classify frame as "real/photo" or "anime/illustration"
- Runtime: onnxruntime CPUExecutionProvider
- Fallback: If confidence < 0.70, route to BOTH branches and take the higher severity result
- Output: "real" | "anime" | "uncertain"

### Stage 2A: Real/Photo Branch (GPU — load/unload per use)
- Primary detector: NudeNet (nudenet Python package, 320n model)
- Install: pip install nudenet
- Purpose: Bounding-box detection of explicit body parts
- ONLY act on these labels:
  - FEMALE_GENITALIA_EXPOSED (threshold: 0.55)
  - MALE_GENITALIA_EXPOSED   (threshold: 0.55)
  - ANUS_EXPOSED             (threshold: 0.60)
  - MALE_GENITALIA_COVERED   (threshold: 0.75 — only if erection is evident)
  - FEMALE_BREAST_EXPOSED    (threshold: 0.55 — block exposed female breasts)
- All other NudeNet labels (BUTTOCKS, etc.) are IGNORED completely
- Runtime: ONNX via nudenet's built-in runner on CPU (nudenet uses ONNX internally)

### Stage 2B: Anime/Hentai Branch (CPU via ONNX)
- Primary: SmilingWolf/wd-vit-large-tagger-v3
- HuggingFace: SmilingWolf/wd-vit-large-tagger-v3
- Purpose: Tag anime content and output safe/sensitive/questionable/explicit rating
- Runtime: onnxruntime CPUExecutionProvider (use the model.onnx file)
- Block if: (rating == "explicit" OR rating == "questionable") AND any of these explicit tags score > 0.50:
  - "penis", "vagina", "pussy", "genitals", "testicles", "erection", "phallus", "anus", "nipples", "bare_breasts", "breasts_out"
- Secondary check: deepghs/anime_rating
- HuggingFace: deepghs/anime_rating
- Purpose: R18 cross-confirmation (safe/r15/r18)
- Runtime: onnxruntime CPUExecutionProvider
- Only consulted if WDv3 rating is "questionable" (not explicit) to decide borderline cases

### Stage 3: Verdict Aggregator
Three tiers only:
- BLOCK:  High confidence explicit genital or exposed breast content detected
- REVIEW: One model flags genitals/breasts, confidence between threshold and threshold+0.15
- SAFE:   Nothing flagged or confidence below threshold

## Detection Rules — Genitals & Exposed Breasts Block Policy
This bot flags genital exposure and exposed female breasts/nipples. The following are explicitly NOT flagged (kept SAFE):
- Covered breasts, cleavage (where nipples/bare breasts are not exposed)
- Buttocks (covered or exposed)
- Lingerie, bikinis, swimwear (as long as nipples are not exposed)
- Suggestive poses without naked breast/genital exposure
- Anime content rated "sensitive" or "questionable" without explicit genital/exposed breast tags

## Video and GIF Handling
- Use imageio[ffmpeg] to extract frames
- GIFs: extract every frame (usually < 50 frames, fast)
- Videos: extract 1 frame per 2 seconds (0.5 fps)
- Process frames in batches of 4 through the pre-filter
- Stop at first BLOCK verdict — do not process remaining frames
- Maximum video size processed: 50MB (reject with message if larger)
- Maximum video duration: 5 minutes (reject with message if longer)

## Discord Bot Behaviour
- Bot monitors only channels listed in MONITORED_CHANNELS config
- On BLOCK verdict:
  1. Delete the message
  2. DM the user with a clear, non-accusatory explanation
  3. Log to a designated LOG_CHANNEL with: user, channel, timestamp, model that flagged it, confidence score, detected label
  4. Do NOT post publicly in the channel
- On REVIEW verdict:
  1. Do NOT delete the message
  2. Post in LOG_CHANNEL with [REVIEW NEEDED] tag so a human moderator can check
- Slash commands (admin only, requires Manage Messages permission):
  - /nsfw-enable #channel — add channel to monitoring
  - /nsfw-disable #channel — remove channel from monitoring
  - /nsfw-status — show which channels are monitored and model status
  - /nsfw-test [attach image] — test an image without taking action

## Error Handling Rules
- If a model fails to load, log the error and skip that model (do not crash the bot)
- If GPU OOM occurs, catch the exception, clear VRAM, and retry on CPU
- If a Discord attachment cannot be downloaded (404, timeout), log and skip silently
- All temp files must be cleaned up in a finally block — no temp file leaks
- Use a per-guild lock when running inference to prevent concurrent GPU usage

## Code Style
- All functions have type hints
- Async/await throughout — never block the event loop with sync inference
  (use asyncio.to_thread() to wrap all inference calls)
- Logging with Python's logging module, not print()
- Log level: DEBUG for inference scores, INFO for verdicts, WARNING for errors
- Config values never hardcoded — always from config.py which reads .env
- Keep inference logic out of cogs — cogs only call moderation/pipeline.py

## .env Variables Required
```
DISCORD_TOKEN=
GUILD_IDS=123456789,987654321        # comma-separated, for slash command sync
MONITORED_CHANNELS=                  # comma-separated channel IDs, or "all"
LOG_CHANNEL_ID=                      # channel ID for moderation logs
ADMIN_ROLE_ID=                       # role that can use slash commands
MODEL_CACHE_DIR=./models             # where to cache HuggingFace downloads
DEVICE=cuda                          # cuda or cpu
MAX_VIDEO_SIZE_MB=50
MAX_VIDEO_DURATION_SECS=300
REVIEW_THRESHOLD_OFFSET=0.15         # confidence window above threshold = REVIEW
```

## requirements.txt Must Include
```
discord.py>=2.3.0
torch>=2.1.0
torchvision>=0.16.0
transformers>=4.40.0
onnxruntime>=1.17.0
nudenet>=3.4.0
huggingface_hub>=0.20.0
Pillow>=10.0.0
opencv-python>=4.8.0
imageio[ffmpeg]>=2.33.0
python-dotenv>=1.0.0
aiohttp>=3.9.0
aiofiles>=23.0.0
```

## What NOT to Do
- Never hardcode the Discord token anywhere
- Never load all models simultaneously
- Never block the event loop — all inference must be in asyncio.to_thread()
- Never flag or delete messages for covered breasts, buttocks, lingerie, or suggestive content (flag exposed breasts/nipples and genitals only)
- Never use API-based model inference (all local only)
- Never store user images beyond the duration of a single inference call
- Never log the actual image content — log metadata only (user ID, channel, timestamp, verdict)
