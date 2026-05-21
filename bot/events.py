"""
bot/events.py — Global bot event handlers.

on_ready: log bot name, guild count, and monitored channels.
on_error: log full traceback.
"""

from __future__ import annotations

import logging

import discord
from discord.ext import commands

logger = logging.getLogger(__name__)


class BotEvents(commands.Cog):
    """Global event handlers registered as a cog."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    @commands.Cog.listener()
    async def on_ready(self) -> None:
        logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info("✅  Bot online: %s (ID: %s)", self.bot.user.name, self.bot.user.id)
        logger.info("📊  Guilds: %d", len(self.bot.guilds))

        try:
            from config import config as bot_config
            if bot_config.monitor_all:
                logger.info("👁️   Monitoring: ALL channels")
            elif bot_config.monitored_channels:
                logger.info("👁️   Monitored channels: %s", bot_config.monitored_channels)
            else:
                logger.info("👁️   No channels configured yet — use /nsfw enable")
        except Exception:
            pass

        logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    @commands.Cog.listener()
    async def on_error(self, event: str, *args, **kwargs) -> None:
        logger.exception("Unhandled exception in event: %s", event)


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(BotEvents(bot))
