# NSFW Moderation Discord Bot

A locally-hosted Discord bot that automatically scans images, GIFs, and videos for explicit genital content using a council of four specialized AI models. **No external APIs — all inference runs entirely on your machine.**

---

## Features

- **4-stage AI pipeline**: Pre-filter → Content router → Real/Anime branch → Verdict aggregation
- **Genital-only policy**: Only flags explicit genital exposure; ignores breasts, buttocks, lingerie, suggestive poses
- **Three-tier verdicts**: `BLOCK` (delete + DM + punish) / `REVIEW` (log for human review) / `SAFE`
- **Video & GIF support**: Extracts frames automatically
- **All existing bot features preserved**: Moderation commands, setup wizard, database logs, etc.

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.11 or higher |
| CUDA Toolkit | 11.x+ (for RTX 2050 / Ampere GPU) |
| FFmpeg | Any recent version (for video frame extraction) |
| PostgreSQL | 14+ (for moderation log database) |
| VRAM | 4 GB minimum (RTX 2050 or better) |

### Install FFmpeg

Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to your PATH.

---

## Installation

### 1. Clone / navigate to the project

```bash
cd nsfwscanner
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ **CUDA note**: If `pip install torch` installs the CPU-only version, install manually:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

---

## Creating a Discord Bot

1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Click **New Application** → name it → go to **Bot** tab
3. Click **Reset Token** and copy your token
4. Under **Privileged Gateway Intents**, enable:
   - ✅ **Server Members Intent**
   - ✅ **Message Content Intent**  ← **Required** for reading attachments
5. Under **OAuth2 → URL Generator**, select scopes:
   - `bot`, `applications.commands`
6. Bot permissions required:
   - `Read Messages/View Channels`
   - `Manage Messages` (to delete flagged content)
   - `Send Messages`
   - `Embed Links`
   - `Attach Files`
   - `Moderate Members` (for timeout punishment)

---

## Configuration

Copy `.env.example` to `.env` and fill in your values:

```bash
copy .env.example .env    # Windows
# cp .env.example .env   # Linux/Mac
```

### Required Variables

```env
# Your bot token from Discord Developer Portal
DISCORD_TOKEN=your_token_here

# Comma-separated guild IDs (for instant slash command sync)
GUILD_IDS=123456789012345678,987654321098765432

# Comma-separated channel IDs to monitor, OR "all" for every channel
MONITORED_CHANNELS=111222333444555666,999888777666555444

# Channel where mod logs are sent
LOG_CHANNEL_ID=555666777888999000

# AI model download directory
MODEL_CACHE_DIR=./models

# Use "cuda" for GPU (recommended) or "cpu"
DEVICE=cuda

# Database
POSTGRES_USER=postgres
POSTGRES_PASSWORD=yourpassword
POSTGRES_DB=nsfwbot
```

---

## First-Run: Download Models

Before starting the bot, download all AI models (~1.5 GB total):

```bash
python scripts/download_models.py
```

This downloads:
| Model | Size | Purpose |
|---|---|---|
| `AdamCodd/vit-base-nsfw-detector` | ~330 MB | Stage 0: Fast pre-filter |
| `deepghs/anime_real_cls` | ~90 MB | Stage 1: Real vs anime router |
| `SmilingWolf/wd-vit-large-tagger-v3` | ~650 MB | Stage 2B: Anime tagger |
| `deepghs/anime_rating` | ~90 MB | Stage 2B: R18 cross-check |
| NudeNet (built-in) | ~5 MB | Stage 2A: Real photo detection |

---

## Starting the Bot

```bash
python main.py
```

The bot will:
1. Connect to Discord
2. Load all cogs
3. Sync slash commands to your configured guilds
4. Initialize the pre-filter model (the only model loaded at startup)

> **Note**: The remaining models are lazy-loaded on the first scan — the first image processed may take a few extra seconds.

---

## Slash Commands

All `/nsfw` commands require **Manage Messages** permission.

| Command | Description |
|---|---|
| `/nsfw enable #channel` | Add a channel to the monitored list |
| `/nsfw disable #channel` | Remove a channel from monitoring |
| `/nsfw status` | Show monitored channels, loaded models, and VRAM usage |
| `/nsfw test [image]` | Run the full pipeline on an image — **no action taken** |

### Legacy Prefix Commands (`;`)

| Command | Description |
|---|---|
| `;scanner` | Show scanner status |
| `;scanner toggle` | Enable/disable the scanner for this server |
| `;scanner threshold [1-100]` | Set the detection confidence threshold |
| `;scanner punishment [none/kick/ban/timeout]` | Set the punishment type |
| `;scanner logchannel [#channel]` | Set the log channel |
| `;setup` | Run the interactive setup wizard |

---

## How the AI Pipeline Works

```
Message with image/video
        │
        ▼
Stage 0: Pre-filter (always loaded, CPU)
  AdamCodd ViT NSFW detector
  Score < 0.25 → SAFE (skip)
        │
        ▼
Stage 1: Gatekeeper (CPU)
  deepghs/anime_real_cls
  ┌────────┬──────────┬────────────┐
  │  real  │  anime   │ uncertain  │
  └────────┴──────────┴────────────┘
       │         │          │
       ▼         ▼       both ▼
Stage 2A: NudeNet    Stage 2B: WD-ViT-v3
Real photo branch     Anime tagger branch
(genital-only policy)  + deepghs/anime_rating
       │                    │
       └──────────┬──────────┘
                  ▼
        Stage 3: Verdict Aggregator
        BLOCK / REVIEW / SAFE
```

### Verdict Tiers

| Verdict | Action |
|---|---|
| **BLOCK** | Delete message, DM user (friendly), punish, log to mod channel |
| **REVIEW** | Log to mod channel with `[REVIEW NEEDED]` tag — no deletion |
| **SAFE** | No action taken |

### Genital-Only Policy

The bot **only** flags these detections:

**Real photos (NudeNet):**
- `FEMALE_GENITALIA_EXPOSED` (confidence > 55%)
- `MALE_GENITALIA_EXPOSED` (confidence > 55%)
- `ANUS_EXPOSED` (confidence > 60%)
- `MALE_GENITALIA_COVERED` (confidence > 75%)

**Anime/Hentai (WD-ViT-v3):**
- Explicit rating + any of: `penis`, `vagina`, `pussy`, `genitals`, `testicles`, `erection`, `phallus` > 50%

**Explicitly NOT flagged:**
- Breasts (covered or exposed)
- Buttocks (covered or exposed)
- Lingerie, bikinis, swimwear
- Suggestive poses without exposure
- Anime rated "sensitive" or "questionable" without genital tags

---

## File Structure

```
nsfwscanner/
├── AGENTS.md                 ← Project specification (do not delete)
├── .env                      ← Your secrets (never commit)
├── .env.example              ← Config template
├── requirements.txt
├── main.py                   ← Bot entry point
├── config.py                 ← BotConfig dataclass
├── database.py               ← PostgreSQL helpers
├── automod_config.json       ← Per-guild moderation settings
├── bot/
│   ├── cogs/
│   │   └── admin.py          ← /nsfw slash commands
│   └── events.py
├── cogs/
│   ├── automod.py            ← NSFW scanning (rewrtten — local AI)
│   ├── automod_setup.py      ← Interactive setup wizard
│   ├── moderation.py         ← ban/kick/mute/warn commands
│   └── ...
├── moderation/
│   ├── pipeline.py           ← Main orchestrator
│   ├── prefilter.py          ← Stage 0: AdamCodd pre-filter
│   ├── gatekeeper.py         ← Stage 1: Content router
│   ├── real_branch.py        ← Stage 2A: NudeNet
│   ├── anime_branch.py       ← Stage 2B: WD-ViT-v3 + anime_rating
│   ├── frame_extractor.py    ← GIF/video frame extraction
│   └── models/
│       └── loader.py         ← Lazy model loader
├── utils/
│   ├── image_utils.py
│   └── logger.py
└── scripts/
    └── download_models.py    ← Run before first start
```

---

## Troubleshooting

**Bot can't see message content:**
→ Enable **Message Content Intent** in the Discord Developer Portal under your bot's settings.

**Models won't download:**
→ Check internet connection. Some HuggingFace repos may require `huggingface-cli login` if they are gated.

**CUDA out of memory:**
→ The pipeline catches OOM errors automatically and retries on CPU. If this happens frequently, set `DEVICE=cpu` in `.env`.

**False positives:**
→ The confidence thresholds are intentionally conservative. Use `/nsfw test` to debug specific images. You can raise thresholds in `moderation/real_branch.py` (`GENITAL_LABELS`) and `moderation/anime_branch.py` (`GENITAL_TAG_THRESHOLD`).

**First scan is slow:**
→ Models are lazy-loaded on first use. Subsequent scans are faster. Run `python scripts/download_models.py` to pre-cache everything.
