"""
main.py — Bot entry point.

Loads BotConfig from .env, initializes the bot with all cogs,
syncs slash commands to configured guilds on startup.
"""

import asyncio
import logging
import time

import discord
from discord.ext import commands

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
from config import config

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

    # Sync slash commands to configured guilds
    if config.guild_ids:
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

    # Initialize database (Removed, relying only on log channels per AGENTS.md)

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
