

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
    import tempfile
    dir_name = os.path.dirname(os.path.abspath(_CONFIG_FILE)) or "."
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", dir=dir_name, suffix=".tmp", delete=False
        ) as tmp:
            json.dump(data, tmp, indent=4)
            tmp_path = tmp.name
        os.replace(tmp_path, _CONFIG_FILE)
    except Exception:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        raise


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

            if guild_channels:
                mentions = []
                for cid in guild_channels:
                    ch = interaction.guild.get_channel(cid)
                    mentions.append(ch.mention if ch else f"`{cid}`")
                channels_text = "\n".join(mentions)
            else:
                channels_text = "All channels (no specific channels configured)"

            enabled_text = "🟢 Enabled" if guild_cfg.get("enabled", True) else "🔴 Disabled"

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

            verdict_emoji = {
                "EXPLICIT": "🚨",
                "NSFW": "🚨",
                "BLOCK": "🚨",
                "SUGGESTIVE": "⚠️",
                "REVIEW": "⚠️",
                "SAFE": "✅",
            }.get(result.verdict, "❓")
            color = {
                "EXPLICIT": 0xDC2626,
                "NSFW": 0xEF4444,
                "BLOCK": 0xFF4444,
                "SUGGESTIVE": 0xFFA500,
                "REVIEW": 0xFFA500,
                "SAFE": 0x22C55E,
            }.get(result.verdict, 0x888888)

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

    @nsfw_group.command(
        name="feedback-stats",
        description="Show moderator feedback statistics and model accuracy estimates",
    )
    @_has_manage_messages()
    async def nsfw_feedback_stats(self, interaction: discord.Interaction) -> None:
        """Display active-learning feedback statistics for this guild."""
        await interaction.response.defer(ephemeral=True)

        try:
            from utils.database import get_feedback_stats

            guild_id = str(interaction.guild_id)
            stats = await get_feedback_stats(guild_id)

            total = stats["total_logged"]

            if total == 0:
                embed = discord.Embed(
                    title="📊 Moderator Feedback Stats",
                    description=(
                        "No feedback has been submitted yet for this server.\n\n"
                        "Use the **✅ Correct Detection**, **❌ False Positive**, "
                        "and **⚠️ False Negative** buttons on moderation log embeds to collect data."
                    ),
                    color=0x6366F1,
                    timestamp=discord.utils.utcnow(),
                )
                embed.set_footer(text="Active Learning Feedback System")
                await interaction.followup.send(embed=embed, ephemeral=True)
                return

            accuracy = stats["accuracy"]
            color = (
                0x22C55E if accuracy >= 80
                else 0xF59E0B if accuracy >= 60
                else 0xEF4444
            )

            embed = discord.Embed(
                title="📊 Moderator Feedback Stats",
                description=(
                    f"Based on **{total}** moderator feedback submissions for this server.\n"
                    f"This data is used to calibrate future ML model improvements."
                ),
                color=color,
                timestamp=discord.utils.utcnow(),
            )

            embed.add_field(
                name="📈 Feedback Summary",
                value=(
                    f"**Total Logged:** `{total}`\n"
                    f"**✅ Correct:** `{stats['correct']}`\n"
                    f"**❌ False Positives:** `{stats['false_positives']}`\n"
                    f"**⚠️ False Negatives:** `{stats['false_negatives']}`"
                ),
                inline=True,
            )

            embed.add_field(
                name="📉 Error Rates",
                value=(
                    f"**FP Rate:** `{stats['fp_rate']}%`\n"
                    f"**FN Rate:** `{stats['fn_rate']}%`\n"
                    f"**Accuracy:** `{accuracy}%`\n"
                ),
                inline=True,
            )

            filled = int(accuracy / 10)
            bar = "🟩" * filled + "⬜" * (10 - filled)
            embed.add_field(
                name="🎯 Accuracy Bar",
                value=f"{bar}\n`{accuracy}%` model precision",
                inline=False,
            )

            top_tags = stats.get("top_failed_tags", [])
            if top_tags:
                tag_lines = [f"`{tag}` — **{count}** failures" for tag, count in top_tags[:8]]
                embed.add_field(
                    name="🔍 Most Commonly Misidentified Tags",
                    value="\n".join(tag_lines),
                    inline=False,
                )

            embed.set_footer(
                text="Active Learning Feedback System • Data used for future ML calibration only"
            )

            await interaction.followup.send(embed=embed, ephemeral=True)

        except Exception as e:
            logger.error("nsfw feedback-stats error: %s", e, exc_info=True)
            await interaction.followup.send(f"❌ Error fetching stats: {e}", ephemeral=True)

    @nsfw_group.command(
        name="stats",
        description="Show overall scan volume, verdict breakdown, and model accuracy",
    )
    @_has_manage_messages()
    async def nsfw_stats(self, interaction: discord.Interaction) -> None:
        """Display combined scan stats and feedback accuracy for this guild."""
        await interaction.response.defer(ephemeral=True)

        try:
            from utils.database import get_scan_stats, get_feedback_stats
            from utils.hash_cache import _DB_PATH as _hc_db

            guild_id = str(interaction.guild_id)
            scan_stats = await get_scan_stats(guild_id)
            fb_stats = await get_feedback_stats(guild_id)

            total = scan_stats["total_scans"]

            embed = discord.Embed(
                title="📊 NSFW Scanner — Full Statistics",
                color=0x6366F1,
                timestamp=discord.utils.utcnow(),
            )

            if total == 0:
                embed.description = (
                    "No scans have been logged yet for this server.\n"
                    "Stats populate automatically as media is posted in monitored channels."
                )
                embed.set_footer(text="NSFW Bot Analytics")
                await interaction.followup.send(embed=embed, ephemeral=True)
                return

            cache_pct = round(scan_stats["cache_hits"] / total * 100, 1) if total else 0.0
            embed.add_field(
                name="📈 Scan Volume",
                value=(
                    f"**Total Scans:** `{total:,}`\n"
                    f"**🚨 Blocked:** `{scan_stats['blocked']:,}`\n"
                    f"**⚠️ Reviewed:** `{scan_stats['reviewed']:,}`\n"
                    f"**✅ Safe:** `{scan_stats['safe']:,}`\n"
                    f"**⚡ Cache Hits:** `{scan_stats['cache_hits']:,}` (`{cache_pct}%`)"
                ),
                inline=True,
            )

            bd = scan_stats["verdict_breakdown"]
            bd_lines = "\n".join(
                f"`{v}`: **{c:,}**" for v, c in sorted(bd.items(), key=lambda x: -x[1])
            )
            embed.add_field(
                name="🗂️ Verdict Breakdown",
                value=bd_lines or "No data",
                inline=True,
            )

            # ── Performance ───────────────────────────────────────────────
            embed.add_field(
                name="⏱️ Avg Processing Time",
                value=f"`{scan_stats['avg_processing_ms']:.0f} ms` per scan",
                inline=True,
            )

            # ── Feedback accuracy (if any) ────────────────────────────────
            fb_total = fb_stats["total_logged"]
            if fb_total > 0:
                accuracy = fb_stats["accuracy"]
                color_bar = "🟩" * int(accuracy / 10) + "⬜" * (10 - int(accuracy / 10))
                embed.add_field(
                    name="🎯 Moderator-Verified Accuracy",
                    value=(
                        f"{color_bar}\n"
                        f"`{accuracy}%` on {fb_total} reviewed cases\n"
                        f"FP rate: `{fb_stats['fp_rate']}%` • FN rate: `{fb_stats['fn_rate']}%`"
                    ),
                    inline=False,
                )

                top_tags = fb_stats.get("top_failed_tags", [])
                if top_tags:
                    tag_lines = [f"`{tag}` — {count} misses" for tag, count in top_tags[:6]]
                    embed.add_field(
                        name="🔍 Top Misidentified Tags",
                        value="\n".join(tag_lines),
                        inline=False,
                    )
            else:
                embed.add_field(
                    name="🎯 Moderator Feedback",
                    value=(
                        "No feedback submitted yet.\n"
                        "Use the **✅ / ❌ / ⚠️** buttons on log embeds to track accuracy."
                    ),
                    inline=False,
                )

            embed.set_footer(text="NSFW Bot Analytics • /nsfw export to download raw CSV")
            await interaction.followup.send(embed=embed, ephemeral=True)

        except Exception as e:
            logger.error("nsfw stats error: %s", e, exc_info=True)
            await interaction.followup.send(f"❌ Error fetching stats: {e}", ephemeral=True)


    @nsfw_group.command(
        name="export",
        description="Download all moderator feedback data as a CSV file",
    )
    @_has_manage_messages()
    async def nsfw_export(self, interaction: discord.Interaction) -> None:
        """Export moderation_feedback table for this guild as a .csv attachment."""
        await interaction.response.defer(ephemeral=True)

        try:
            from utils.database import export_feedback_csv
            import io

            guild_id = str(interaction.guild_id)
            csv_text = await export_feedback_csv(guild_id)

            if not csv_text.strip() or csv_text.count("\n") <= 1:
                await interaction.followup.send(
                    "ℹ️ No feedback data has been recorded for this server yet.",
                    ephemeral=True,
                )
                return

            row_count = csv_text.count("\n") - 1  # subtract header row
            csv_bytes = io.BytesIO(csv_text.encode("utf-8"))
            filename = f"nsfw_feedback_{interaction.guild_id}.csv"
            file = discord.File(fp=csv_bytes, filename=filename)

            embed = discord.Embed(
                title="📥 Feedback Export Ready",
                description=(
                    f"Exported **{row_count:,}** feedback record(s) for this server.\n"
                    "Columns: `id`, `message_id`, `user_id`, `moderator_id`, "
                    "`predicted_verdict`, `moderator_verdict`, `branch`, `model`, "
                    "`processing_time_ms`, `detected_tags`, `created_at`"
                ),
                color=0x22C55E,
                timestamp=discord.utils.utcnow(),
            )
            embed.set_footer(text="Use this data to calibrate future ML model improvements")
            await interaction.followup.send(embed=embed, file=file, ephemeral=True)

        except Exception as e:
            logger.error("nsfw export error: %s", e, exc_info=True)
            await interaction.followup.send(f"❌ Export failed: {e}", ephemeral=True)


async def setup(bot: commands.Bot) -> None:
    cog = NSFWAdminCog(bot)
    await bot.add_cog(cog)
