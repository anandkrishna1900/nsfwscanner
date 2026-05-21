"""
bot/cogs/admin.py — Admin slash command group for NSFW scanner control.

All commands require Manage Messages permission.
Channel monitoring config is persisted to automod_config.json (same file
that automod.py uses) under each guild's "channels" list — no separate
nsfw_channels.json.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import discord
from discord import app_commands
from discord.ext import commands

logger = logging.getLogger(__name__)

_CONFIG_FILE = "automod_config.json"

# Defaults mirror get_server_config() in automod.py exactly
_DEFAULTS = {
    "enabled": True,
    "channels": [],
    "punishment": "timeout",
    "timeout_duration": 10,
    "ban_duration": None,
    "log_channel": None,
    "whitelisted_roles": [],
    "whitelisted_users": [],
}


def _load_config() -> dict:
    try:
        with open(_CONFIG_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def _save_config(data: dict) -> None:
    with open(_CONFIG_FILE, "w") as f:
        json.dump(data, f, indent=4)


def _get_guild(data: dict, guild_id: str) -> dict:
    """Return the guild sub-dict, creating it with defaults if missing."""
    if guild_id not in data:
        data[guild_id] = dict(_DEFAULTS)
    # Back-fill any missing keys
    for k, v in _DEFAULTS.items():
        if k not in data[guild_id]:
            data[guild_id][k] = v
    return data[guild_id]


def _has_manage_messages():
    """Check decorator: requires Manage Messages."""
    async def predicate(interaction: discord.Interaction) -> bool:
        if not interaction.user.guild_permissions.manage_messages:
            await interaction.response.send_message(
                "❌ You need **Manage Messages** permission to use this command.",
                ephemeral=True,
            )
            return False
        return True
    return app_commands.check(predicate)


class NSFWAdminCog(commands.Cog):
    """Admin slash commands for the local NSFW AI scanner."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    nsfw_group = app_commands.Group(
        name="nsfw",
        description="NSFW scanner admin commands (requires Manage Messages)",
    )

    # ── /nsfw enable ──────────────────────────────────────────────────────────

    @nsfw_group.command(name="enable", description="Add a channel to NSFW monitoring")
    @_has_manage_messages()
    async def nsfw_enable(
        self,
        interaction: discord.Interaction,
        channel: discord.TextChannel,
    ) -> None:
        """Enable NSFW scanning for a specific channel."""
        guild_id = str(interaction.guild_id)
        data = _load_config()
        guild_cfg = _get_guild(data, guild_id)

        if channel.id in guild_cfg["channels"]:
            await interaction.response.send_message(
                f"ℹ️ {channel.mention} is already being monitored.", ephemeral=True
            )
            return

        guild_cfg["channels"].append(channel.id)
        _save_config(data)

        embed = discord.Embed(
            title="✅ Channel Added to Monitoring",
            description=f"{channel.mention} will now be scanned for explicit content.",
            color=0x22C55E,
        )
        embed.set_footer(text="Use /nsfw status to see all monitored channels")
        await interaction.response.send_message(embed=embed)
        logger.info("Guild %s: added #%s to NSFW monitoring", guild_id, channel.name)

    # ── /nsfw disable ─────────────────────────────────────────────────────────

    @nsfw_group.command(name="disable", description="Remove a channel from NSFW monitoring")
    @_has_manage_messages()
    async def nsfw_disable(
        self,
        interaction: discord.Interaction,
        channel: discord.TextChannel,
    ) -> None:
        """Disable NSFW scanning for a specific channel."""
        guild_id = str(interaction.guild_id)
        data = _load_config()
        guild_cfg = _get_guild(data, guild_id)

        if channel.id not in guild_cfg["channels"]:
            await interaction.response.send_message(
                f"ℹ️ {channel.mention} is not currently being monitored.", ephemeral=True
            )
            return

        guild_cfg["channels"].remove(channel.id)
        _save_config(data)

        embed = discord.Embed(
            title="🔕 Channel Removed from Monitoring",
            description=f"{channel.mention} will no longer be scanned.",
            color=0xEF4444,
        )
        await interaction.response.send_message(embed=embed)
        logger.info("Guild %s: removed #%s from NSFW monitoring", guild_id, channel.name)

    # ── /nsfw status ──────────────────────────────────────────────────────────

    @nsfw_group.command(name="status", description="Show NSFW scanner status and monitored channels")
    @_has_manage_messages()
    async def nsfw_status(self, interaction: discord.Interaction) -> None:
        """Display current scanner status, monitored channels, and model info."""
        await interaction.response.defer(ephemeral=True)

        try:
            from config import config as bot_config
            from moderation import pipeline as pl

            guild_id = str(interaction.guild_id)
            data = _load_config()
            guild_cfg = _get_guild(data, guild_id)
            guild_channels: list[int] = guild_cfg.get("channels", [])

            # Resolve channel mentions
            if guild_channels:
                mentions = []
                for cid in guild_channels:
                    ch = interaction.guild.get_channel(cid)
                    mentions.append(ch.mention if ch else f"`{cid}`")
                channels_text = "\n".join(mentions)
            else:
                channels_text = "All channels (no specific channels configured)"

            # Scanner enabled/disabled
            enabled_text = "🟢 Enabled" if guild_cfg.get("enabled", True) else "🔴 Disabled"

            # Loaded models info
            loaded = pl._initialized
            model_lines = []
            if loaded:
                if pl._prefilter:
                    model_lines.append("✅ `prefilter` (AdamCodd ViT)")
                if pl._gatekeeper:
                    model_lines.append("✅ `gatekeeper` (deepghs/anime_real_cls)")
                if pl._real_branch:
                    model_lines.append("✅ `real_branch` (NudeNet)")
                if pl._anime_branch:
                    model_lines.append("✅ `anime_branch` (WDv3 + deepghs/anime_rating)")
            else:
                model_lines = ["⏳ Models not yet initialized (loads on first scan)"]

            # VRAM info
            vram_text = "N/A"
            try:
                import torch
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1e6
                    reserved = torch.cuda.memory_reserved() / 1e6
                    total = torch.cuda.get_device_properties(0).total_memory / 1e6
                    vram_text = (
                        f"{allocated:.0f} MB allocated / "
                        f"{reserved:.0f} MB reserved / "
                        f"{total:.0f} MB total"
                    )
            except Exception:
                pass

            embed = discord.Embed(
                title="🤖 NSFW Scanner Status",
                color=0x6366F1,
                timestamp=discord.utils.utcnow(),
            )
            embed.add_field(name="⚙️ Scanner", value=enabled_text, inline=True)
            embed.add_field(name="🔨 Punishment", value=guild_cfg.get("punishment", "timeout").title(), inline=True)
            embed.add_field(name="📡 Monitored Channels", value=channels_text, inline=False)
            embed.add_field(name="🧠 AI Models", value="\n".join(model_lines), inline=False)
            embed.add_field(name="💾 VRAM Usage", value=vram_text, inline=False)
            embed.add_field(name="⚙️ Device", value=bot_config.device.upper(), inline=True)
            embed.add_field(name="📁 Model Cache", value=f"`{bot_config.model_cache_dir}`", inline=True)
            embed.set_footer(text="Local AI Pipeline — No external APIs")
            await interaction.followup.send(embed=embed, ephemeral=True)

        except Exception as e:
            logger.error("nsfw status error: %s", e, exc_info=True)
            await interaction.followup.send(f"❌ Error: {e}", ephemeral=True)

    # ── /nsfw test ────────────────────────────────────────────────────────────

    @nsfw_group.command(name="test", description="Test the NSFW scanner on an attached image (no action taken)")
    @_has_manage_messages()
    async def nsfw_test(
        self,
        interaction: discord.Interaction,
        attachment: discord.Attachment,
    ) -> None:
        """Run the full pipeline on an image and report back — no deletion or punishment."""
        await interaction.response.defer(ephemeral=True)

        try:
            from config import config as bot_config
            from moderation.pipeline import scan_attachment

            result = await scan_attachment(attachment.url, bot_config, bypass_prefilter=True)

            verdict_emoji = {"BLOCK": "🚨", "REVIEW": "⚠️", "SAFE": "✅"}.get(result.verdict, "❓")
            color = {"BLOCK": 0xFF4444, "REVIEW": 0xFFA500, "SAFE": 0x22C55E}.get(result.verdict, 0x888888)

            embed = discord.Embed(
                title=f"{verdict_emoji} Scan Result: {result.verdict}",
                color=color,
                timestamp=discord.utils.utcnow(),
            )
            embed.add_field(name="File", value=attachment.filename, inline=True)
            embed.add_field(name="Verdict", value=f"`{result.verdict}`", inline=True)
            embed.add_field(name="Branch", value=result.branch, inline=True)
            embed.add_field(name="Model", value=result.model, inline=True)
            embed.add_field(name="Reason", value=result.reason or "—", inline=False)
            embed.add_field(name="Processing Time", value=f"{result.processing_time_ms:.0f}ms", inline=True)
            if result.frame_index is not None:
                embed.add_field(name="Triggered Frame", value=f"#{result.frame_index}", inline=True)

            if getattr(result, "pipeline_steps", None):
                steps_text = "\n\n".join(result.pipeline_steps)
                if len(steps_text) > 980:
                    steps_text = steps_text[:980] + "\n... [Trace truncated due to character limit]"
                embed.add_field(
                    name="📋 AI Model Council Verification Trace",
                    value=f"```yaml\n{steps_text}\n```",
                    inline=False,
                )

            embed.set_footer(text="Test mode — no action was taken regardless of verdict")

            await interaction.followup.send(embed=embed, ephemeral=True)

        except Exception as e:
            logger.error("nsfw test error: %s", e, exc_info=True)
            await interaction.followup.send(f"❌ Scan failed: {e}", ephemeral=True)


async def setup(bot: commands.Bot) -> None:
    cog = NSFWAdminCog(bot)
    await bot.add_cog(cog)
