"""
main.py — Bot entry point.

Loads BotConfig from .env, initializes the bot with all cogs,
syncs slash commands to configured guilds on startup.
"""

import os
# Silence Hugging Face Hub console warnings and verbosity
os.environ["HF_HUB_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Eagerly import torch at startup to prevent dll loading blocking the asyncio event loop thread later
try:
    import torch
except Exception:
    pass

import asyncio
import logging
import time
import warnings

import discord
from discord.ext import commands, tasks

# Suppress HF unauthenticated request warnings
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
warnings.filterwarnings("ignore", category=UserWarning)

class SilenceHFWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "unauthenticated requests" in msg:
            return False
        return True

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Apply filter to all root handlers and huggingface_hub loggers
for handler in logging.getLogger().handlers:
    handler.addFilter(SilenceHFWarningFilter())
logging.getLogger("huggingface_hub").addFilter(SilenceHFWarningFilter())
logging.getLogger("huggingface_hub.utils._http").addFilter(SilenceHFWarningFilter())

# Silence verbose warnings and requests from third-party libraries
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)

# ── Config ────────────────────────────────────────────────────────────────────
from config import config
from bot.ui.feedback_view import ModerationFeedbackView

# ── Intents ───────────────────────────────────────────────────────────────────
intents = discord.Intents.default()
intents.message_content = True   # Required for reading attachment content types
intents.messages = True
intents.guilds = True
intents.members = True

# ── Bot ───────────────────────────────────────────────────────────────────────
bot = commands.Bot(
    command_prefix=config.prefix,
    intents=intents,
    help_command=None,
)


# ── on_message passthrough (allows cog listeners + prefix commands) ───────────
@bot.event
async def on_message(message: discord.Message) -> None:
    if message.author.bot:
        return
    await bot.process_commands(message)


# ── Cog loader ────────────────────────────────────────────────────────────────
COGS = [
    "cogs.automod",
    "cogs.admin",
]


async def load_cogs() -> None:
    for cog in COGS:
        try:
            await bot.load_extension(cog)
            logger.info("✅ Loaded %s", cog)
        except Exception as e:
            logger.error("❌ Failed to load %s: %s", cog, e)


# ── on_ready ──────────────────────────────────────────────────────────────────
@bot.event
async def on_ready() -> None:
    logger.info("✅ %s connected to Discord!", bot.user.name)
    logger.info("📊 Serving %d guild(s)", len(bot.guilds))
    logger.info("🔐 Message Content Intent: %s", intents.message_content)

    # Register persistent views so buttons survive restarts
    bot.add_view(ModerationFeedbackView())
    logger.info("🔘 Registered persistent feedback view")

    # Sync slash commands to configured guilds
    if config.guild_ids:
        # First, clear global commands to prevent duplicate entries between global and guild scopes
        try:
            global_cmds = bot.tree.get_commands(guild=None)
            bot.tree.clear_commands(guild=None)
            await bot.tree.sync()
            
            # Restore global commands in memory
            for cmd in global_cmds:
                bot.tree.add_command(cmd)
                
            logger.info("🗑️ Cleared global slash commands to prevent duplicates")
        except Exception as e:
            logger.warning("Could not clear global commands: %s", e)

        for guild_id in config.guild_ids:
            guild_obj = discord.Object(id=guild_id)
            try:
                bot.tree.copy_global_to(guild=guild_obj)
                synced = await bot.tree.sync(guild=guild_obj)
                logger.info("🔄 Synced %d slash commands to guild %s", len(synced), guild_id)
            except Exception as e:
                logger.warning("Could not sync commands to guild %s: %s", guild_id, e)
    else:
        # Global sync (takes up to 1 hour to propagate)
        try:
            synced = await bot.tree.sync()
            logger.info("🔄 Synced %d slash commands globally", len(synced))
        except Exception as e:
            logger.warning("Global command sync failed: %s", e)

    # Log monitored channels
    if config.monitor_all:
        logger.info("👁️  Monitoring: ALL channels")
    elif config.monitored_channels:
        logger.info("👁️  Monitoring channels: %s", config.monitored_channels)
    else:
        logger.info("👁️  No channels pre-configured — use /nsfw enable #channel")

    # Start daily database cleanup
    if not daily_cleanup.is_running():
        daily_cleanup.start()
        logger.info("🧹 Daily cleanup task started (runs every 24h)")


# ── Daily database cleanup task ───────────────────────────────────────────────────
@tasks.loop(hours=24)
async def daily_cleanup() -> None:
    """Prune old records from the SQLite database once every 24 hours."""
    try:
        from utils.database import cleanup_old_records
        from config import config
        deleted = await cleanup_old_records(
            db_path=config.sqlite_db_path,
            feedback_days=90,
            cache_days=30,
            scan_log_days=60,
        )
        logger.info(
            "🧹 Daily cleanup done — feedback: %d, cache: %d, scan_log: %d rows deleted",
            deleted["moderation_feedback"],
            deleted["image_hash_cache"],
            deleted["scan_log"],
        )
    except Exception as e:
        logger.error("Daily cleanup failed: %s", e)


@daily_cleanup.before_loop
async def before_daily_cleanup() -> None:
    await bot.wait_until_ready()


# ── on_error ──────────────────────────────────────────────────────────────────
@bot.event
async def on_error(event: str, *args, **kwargs) -> None:
    logger.exception("Unhandled error in event %s", event)


# ── Persisted channel loader ──────────────────────────────────────────────────
def _load_persisted_channels(cfg) -> None:
    """
    Merge channel IDs saved by /nsfw enable (in automod_config.json) into the
    live config.monitored_channels list so they survive a bot restart.
    """
    import json as _json
    import os as _os
    path = "automod_config.json"
    if not _os.path.exists(path):
        return
    try:
        with open(path) as f:
            data = _json.load(f)
        added = 0
        for guild_data in data.values():
            for cid in guild_data.get("channels", []):
                if cid not in cfg.monitored_channels:
                    cfg.monitored_channels.append(cid)
                    added += 1
        if added:
            logger.info(
                "Loaded %d persisted channel(s) from automod_config.json",
                added,
            )
    except Exception as e:
        logger.warning("Could not load persisted channels: %s", e)


# ── Main ──────────────────────────────────────────────────────────────────────
async def main() -> None:
    bot.start_time = time.time()

    # Initialize SQLite database for moderator feedback
    from utils.database import init_db
    await init_db(config.sqlite_db_path)

    # Merge channels persisted by /nsfw enable into live config
    _load_persisted_channels(config)

    async with bot:
        await load_cogs()
        try:
            await bot.start(config.discord_token)
        except KeyboardInterrupt:
            pass
        finally:
            logger.info("Bot shutdown complete.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
