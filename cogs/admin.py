"""
cogs/admin.py — Compact slash commands for NSFW scanner administration.

All commands require Manage Messages permission.
New commands: set-timeout, whitelist-role/user, unwhitelist-role/user,
              clear-cache, user-stats, set-punishment.
Standalone:   /ping (latency + uptime)
"""

from __future__ import annotations

import json
import logging
import os
import time

import discord
from discord import app_commands
from discord.ext import commands

logger = logging.getLogger(__name__)

_CONFIG_FILE = "automod_config.json"

_DEFAULTS: dict = {
    "enabled": True,
    "channels": [],
    "punishment": "timeout",
    "timeout_duration": 10,
    "ban_duration": None,
    "log_channel": None,
    "whitelisted_roles": [],
    "whitelisted_users": [],
}


# ── Config helpers ─────────────────────────────────────────────────────────────

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
        with tempfile.NamedTemporaryFile("w", dir=dir_name, suffix=".tmp", delete=False) as tmp:
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
    """Return the guild config sub-dict, creating and back-filling defaults if missing."""
    if guild_id not in data:
        data[guild_id] = dict(_DEFAULTS)
    else:
        for k, v in _DEFAULTS.items():
            if k not in data[guild_id]:
                data[guild_id][k] = v
    return data[guild_id]


def _requires_manage_messages() -> app_commands.check:
    async def predicate(interaction: discord.Interaction) -> bool:
        if not interaction.user.guild_permissions.manage_messages:
            await interaction.response.send_message(
                "\u274c You need **Manage Messages** permission.", ephemeral=True
            )
            return False
        return True
    return app_commands.check(predicate)


class NSFWAdminCog(commands.Cog):
    """Admin slash commands for the NSFW AI scanner."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    nsfw = app_commands.Group(
        name="nsfw",
        description="NSFW scanner admin commands (requires Manage Messages)",
    )

    # ── Channel management ────────────────────────────────────────────────────

    @nsfw.command(name="enable", description="Add a channel to NSFW monitoring")
    @_requires_manage_messages()
    async def nsfw_enable(
        self, interaction: discord.Interaction, channel: discord.TextChannel
    ) -> None:
        guild_id = str(interaction.guild_id)
        data = _load_config()
        cfg = _get_guild(data, guild_id)
        if channel.id in cfg["channels"]:
            await interaction.response.send_message(
                f"\u2139\ufe0f {channel.mention} is already monitored.", ephemeral=True
            )
            return
        cfg["channels"].append(channel.id)
        _save_config(data)
        await interaction.response.send_message(
            f"\u2705 Now scanning {channel.mention} for NSFW content.", ephemeral=True
        )

    @nsfw.command(name="disable", description="Remove a channel from NSFW monitoring")
    @_requires_manage_messages()
    async def nsfw_disable(
        self, interaction: discord.Interaction, channel: discord.TextChannel
    ) -> None:
        guild_id = str(interaction.guild_id)
        data = _load_config()
        cfg = _get_guild(data, guild_id)
        if channel.id not in cfg["channels"]:
            await interaction.response.send_message(
                f"\u2139\ufe0f {channel.mention} is not being monitored.", ephemeral=True
            )
            return
        cfg["channels"].remove(channel.id)
        _save_config(data)
        await interaction.response.send_message(
            f"\U0001f515 Stopped scanning {channel.mention}.", ephemeral=True
        )

    # ── Status ────────────────────────────────────────────────────────────────

    @nsfw.command(name="status", description="Show scanner status for this server")
    @_requires_manage_messages()
    async def nsfw_status(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True)
        try:
            from config import config as bot_config
            from moderation import pipeline as pl
            guild_id = str(interaction.guild_id)
            data = _load_config()
            cfg = _get_guild(data, guild_id)
            channels = cfg.get("channels", [])
            ch_parts = []
            for cid in channels:
                ch = interaction.guild.get_channel(cid)
                ch_parts.append(ch.mention if ch else f"`{cid}`")
            ch_str = " ".join(ch_parts) or "All channels"

            if pl._initialized:
                loaded = [n for n, m in [
                    ("prefilter", pl._prefilter), ("gatekeeper", pl._gatekeeper),
                    ("real", pl._real_branch), ("anime", pl._anime_branch),
                ] if m]
                model_str = "\u2705 " + " \u00b7 ".join(loaded) if loaded else "\u26a0\ufe0f None loaded"
            else:
                model_str = "\u23f3 Not loaded yet"

            vram_str = "N/A"
            try:
                import torch
                if torch.cuda.is_available():
                    alloc = torch.cuda.memory_allocated() / 1e6
                    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e6
                    vram_str = f"{alloc:.0f}/{total_mem:.0f} MB"
            except Exception:
                pass

            on_off = "\U0001f7e2 On" if cfg.get("enabled", True) else "\U0001f534 Off"
            punishment = cfg.get("punishment", "timeout")
            timeout_min = cfg.get("timeout_duration", 10)
            punish_str = punishment + (f" ({timeout_min}m)" if punishment == "timeout" else "")
            wl_roles = len(cfg.get("whitelisted_roles", []))
            wl_users = len(cfg.get("whitelisted_users", []))

            embed = discord.Embed(
                title="\U0001f916 NSFW Scanner Status", color=0x6366F1,
                timestamp=discord.utils.utcnow()
            )
            embed.add_field(name="State", value=on_off, inline=True)
            embed.add_field(name="Punishment", value=punish_str, inline=True)
            embed.add_field(name="Device", value=bot_config.device.upper(), inline=True)
            embed.add_field(name="Channels", value=ch_str, inline=False)
            embed.add_field(name="Models", value=model_str, inline=True)
            embed.add_field(name="VRAM", value=vram_str, inline=True)
            if wl_roles or wl_users:
                embed.add_field(
                    name="Whitelisted",
                    value=f"Roles: `{wl_roles}` \u00b7 Users: `{wl_users}`",
                    inline=True,
                )
            await interaction.followup.send(embed=embed, ephemeral=True)
        except Exception as e:
            logger.error("nsfw status error: %s", e, exc_info=True)
            await interaction.followup.send(f"\u274c Error: {e}", ephemeral=True)

    # ── Test ──────────────────────────────────────────────────────────────────

    @nsfw.command(name="test", description="Scan an attachment (no moderation action taken)")
    @_requires_manage_messages()
    async def nsfw_test(
        self, interaction: discord.Interaction, attachment: discord.Attachment
    ) -> None:
        await interaction.response.defer(ephemeral=True)
        try:
            from config import config as bot_config
            from moderation.pipeline import scan_attachment
            result = await scan_attachment(attachment.url, bot_config, bypass_prefilter=True)
            EMOJI = {
                "EXPLICIT": "\U0001f534", "NSFW": "\U0001f7e0",
                "BLOCK": "\U0001f7e0", "SUGGESTIVE": "\U0001f7e1",
                "REVIEW": "\U0001f7e1", "SAFE": "\U0001f7e2",
            }
            COLOR = {
                "EXPLICIT": 0xDC2626, "NSFW": 0xEF4444, "BLOCK": 0xFF4444,
                "SUGGESTIVE": 0xFFA500, "REVIEW": 0xFFA500, "SAFE": 0x22C55E,
            }
            embed = discord.Embed(
                title=f"{EMOJI.get(result.verdict, '?')} {result.verdict} \u2014 {attachment.filename}",
                color=COLOR.get(result.verdict, 0x888888),
                timestamp=discord.utils.utcnow(),
            )
            embed.add_field(name="Branch", value=result.branch or "\u2014", inline=True)
            embed.add_field(name="Model", value=result.model or "\u2014", inline=True)
            embed.add_field(name="Time", value=f"{result.processing_time_ms:.0f}ms", inline=True)
            embed.add_field(name="Reason", value=(result.reason or "\u2014")[:1024], inline=False)
            if getattr(result, "pipeline_steps", None):
                steps = "\n\n".join(result.pipeline_steps)
                if len(steps) > 900:
                    steps = steps[:900] + "\n\u2026[truncated]"
                embed.add_field(name="Trace", value=f"```yaml\n{steps}\n```", inline=False)
            embed.set_footer(text="Test mode \u2014 no action taken")
            await interaction.followup.send(embed=embed, ephemeral=True)
        except Exception as e:
            logger.error("nsfw test error: %s", e, exc_info=True)
            await interaction.followup.send(f"\u274c Scan failed: {e}", ephemeral=True)

    # ── Stats ─────────────────────────────────────────────────────────────────

    @nsfw.command(name="stats", description="Compact scan volume and accuracy overview")
    @_requires_manage_messages()
    async def nsfw_stats(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True)
        try:
            from utils.database import get_scan_stats, get_feedback_stats
            scan = await get_scan_stats(str(interaction.guild_id))
            fb = await get_feedback_stats(str(interaction.guild_id))
            total = scan["total_scans"]
            if total == 0:
                await interaction.followup.send(
                    "\U0001f4ca No scans yet \u2014 stats will appear after media is posted.",
                    ephemeral=True,
                )
                return
            cache_pct = round(scan["cache_hits"] / total * 100, 1) if total else 0.0
            bd = scan["verdict_breakdown"]
            bd_str = " \u00b7 ".join(
                f"`{v}:{c}`" for v, c in sorted(bd.items(), key=lambda x: -x[1])
            )
            if fb["total_logged"] > 0:
                acc = fb["accuracy"]
                bar = "\U0001f7e9" * int(acc / 10) + "\u2b1c" * (10 - int(acc / 10))
                fp = fb["fp_rate"]
                fn = fb["fn_rate"]
                tot_fb = fb["total_logged"]
                acc_str = f"{bar} `{acc}%` \u00b7 FP:{fp}% FN:{fn}% ({tot_fb} reviewed)"
            else:
                acc_str = "No feedback yet"
            blocked = scan["blocked"]
            reviewed = scan["reviewed"]
            safe = scan["safe"]
            avg_ms = scan["avg_processing_ms"]
            embed = discord.Embed(title="\U0001f4ca NSFW Stats", color=0x6366F1, timestamp=discord.utils.utcnow())
            embed.add_field(
                name="Scans",
                value=f"**{total:,}** \u00b7 \U0001f6a8 {blocked:,} blocked \u00b7 \u26a0\ufe0f {reviewed:,} reviewed \u00b7 \u2705 {safe:,} safe",
                inline=False,
            )
            embed.add_field(
                name="Performance",
                value=f"\u26a1 Cache `{cache_pct}%` \u00b7 \u23f1 `{avg_ms:.0f}ms` avg",
                inline=False,
            )
            embed.add_field(name="Verdicts", value=bd_str or "\u2014", inline=False)
            embed.add_field(name="Accuracy", value=acc_str, inline=False)
            await interaction.followup.send(embed=embed, ephemeral=True)
        except Exception as e:
            logger.error("nsfw stats error: %s", e, exc_info=True)
            await interaction.followup.send(f"\u274c Error: {e}", ephemeral=True)

    # ── Feedback stats ────────────────────────────────────────────────────────

    @nsfw.command(name="feedback-stats", description="Compact moderator feedback accuracy")
    @_requires_manage_messages()
    async def nsfw_feedback_stats(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True)
        try:
            from utils.database import get_feedback_stats
            stats = await get_feedback_stats(str(interaction.guild_id))
            total = stats["total_logged"]
            if total == 0:
                await interaction.followup.send(
                    "\U0001f4ca No feedback yet \u2014 use the \u2705 / \u274c / \u26a0\ufe0f buttons on log embeds.",
                    ephemeral=True,
                )
                return
            acc = stats["accuracy"]
            bar = "\U0001f7e9" * int(acc / 10) + "\u2b1c" * (10 - int(acc / 10))
            color = 0x22C55E if acc >= 80 else 0xF59E0B if acc >= 60 else 0xEF4444
            correct = stats["correct"]
            fps = stats["false_positives"]
            fns = stats["false_negatives"]
            fp_r = stats["fp_rate"]
            fn_r = stats["fn_rate"]
            lines = [
                f"{bar} **{acc}%** accuracy",
                f"\u2705 `{correct}` correct \u00b7 \u274c `{fps}` FP \u00b7 \u26a0\ufe0f `{fns}` FN",
                f"FP: `{fp_r}%` \u00b7 FN: `{fn_r}%` \u00b7 Total: `{total}`",
            ]
            top = stats.get("top_failed_tags", [])
            if top:
                lines.append("Top missed: " + ", ".join(f"`{t}` ({c})" for t, c in top[:5]))
            embed = discord.Embed(
                title="\U0001f4ca Feedback Stats",
                description="\n".join(lines),
                color=color,
                timestamp=discord.utils.utcnow(),
            )
            await interaction.followup.send(embed=embed, ephemeral=True)
        except Exception as e:
            logger.error("nsfw feedback-stats error: %s", e, exc_info=True)
            await interaction.followup.send(f"\u274c Error: {e}", ephemeral=True)

    # ── Export ────────────────────────────────────────────────────────────────

    @nsfw.command(name="export", description="Download feedback data as CSV")
    @_requires_manage_messages()
    async def nsfw_export(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True)
        try:
            import io
            from utils.database import export_feedback_csv
            csv_text = await export_feedback_csv(str(interaction.guild_id))
            if not csv_text.strip() or csv_text.count("\n") <= 1:
                await interaction.followup.send("\u2139\ufe0f No feedback data to export yet.", ephemeral=True)
                return
            rows = csv_text.count("\n") - 1
            file = discord.File(
                fp=io.BytesIO(csv_text.encode("utf-8")),
                filename=f"nsfw_feedback_{interaction.guild_id}.csv",
            )
            await interaction.followup.send(
                f"\U0001f4e5 **{rows:,}** feedback records.", file=file, ephemeral=True
            )
        except Exception as e:
            logger.error("nsfw export error: %s", e, exc_info=True)
            await interaction.followup.send(f"\u274c Export failed: {e}", ephemeral=True)

    # ── Set punishment ────────────────────────────────────────────────────────

    @nsfw.command(name="set-punishment", description="Set punishment: none / timeout / kick / ban")
    @_requires_manage_messages()
    async def nsfw_set_punishment(self, interaction: discord.Interaction, punishment: str) -> None:
        if punishment.lower() not in ("none", "timeout", "kick", "ban"):
            await interaction.response.send_message(
                "\u274c Valid options: `none` \u00b7 `timeout` \u00b7 `kick` \u00b7 `ban`",
                ephemeral=True,
            )
            return
        data = _load_config()
        cfg = _get_guild(data, str(interaction.guild_id))
        cfg["punishment"] = punishment.lower()
        _save_config(data)
        await interaction.response.send_message(
            f"\u2705 Punishment set to **{punishment.lower()}**.", ephemeral=True
        )

    # ── Set timeout duration ──────────────────────────────────────────────────

    @nsfw.command(name="set-timeout", description="Set timeout duration in minutes (1-10080)")
    @_requires_manage_messages()
    async def nsfw_set_timeout(self, interaction: discord.Interaction, minutes: int) -> None:
        if not 1 <= minutes <= 10080:
            await interaction.response.send_message(
                "\u274c Must be 1\u201310\u00a0080 minutes (max 7 days).", ephemeral=True
            )
            return
        data = _load_config()
        cfg = _get_guild(data, str(interaction.guild_id))
        cfg["timeout_duration"] = minutes
        cfg["punishment"] = "timeout"
        _save_config(data)
        days, rem = divmod(minutes, 1440)
        hours, mins = divmod(rem, 60)
        parts = (
            ([f"{days}d"] if days else [])
            + ([f"{hours}h"] if hours else [])
            + ([f"{mins}m"] if mins else [])
        )
        dur = " ".join(parts) or f"{minutes}m"
        await interaction.response.send_message(
            f"\u2705 Timeout set to **{dur}** (punishment \u2192 `timeout`).", ephemeral=True
        )

    # ── Whitelist: roles ──────────────────────────────────────────────────────

    @nsfw.command(name="whitelist-role", description="Exempt a role from NSFW scanning")
    @_requires_manage_messages()
    async def nsfw_whitelist_role(self, interaction: discord.Interaction, role: discord.Role) -> None:
        data = _load_config()
        cfg = _get_guild(data, str(interaction.guild_id))
        if role.id in cfg["whitelisted_roles"]:
            await interaction.response.send_message(
                f"\u2139\ufe0f {role.mention} is already whitelisted.", ephemeral=True
            )
            return
        cfg["whitelisted_roles"].append(role.id)
        _save_config(data)
        await interaction.response.send_message(
            f"\u2705 {role.mention} is now whitelisted.", ephemeral=True
        )

    @nsfw.command(name="unwhitelist-role", description="Remove a role from the scan whitelist")
    @_requires_manage_messages()
    async def nsfw_unwhitelist_role(self, interaction: discord.Interaction, role: discord.Role) -> None:
        data = _load_config()
        cfg = _get_guild(data, str(interaction.guild_id))
        if role.id not in cfg["whitelisted_roles"]:
            await interaction.response.send_message(
                f"\u2139\ufe0f {role.mention} is not whitelisted.", ephemeral=True
            )
            return
        cfg["whitelisted_roles"].remove(role.id)
        _save_config(data)
        await interaction.response.send_message(
            f"\u2705 {role.mention} removed from whitelist.", ephemeral=True
        )

    # ── Whitelist: users ──────────────────────────────────────────────────────

    @nsfw.command(name="whitelist-user", description="Exempt a user from NSFW scanning")
    @_requires_manage_messages()
    async def nsfw_whitelist_user(self, interaction: discord.Interaction, user: discord.Member) -> None:
        data = _load_config()
        cfg = _get_guild(data, str(interaction.guild_id))
        if user.id in cfg["whitelisted_users"]:
            await interaction.response.send_message(
                f"\u2139\ufe0f {user.mention} is already whitelisted.", ephemeral=True
            )
            return
        cfg["whitelisted_users"].append(user.id)
        _save_config(data)
        await interaction.response.send_message(
            f"\u2705 {user.mention} whitelisted \u2014 their media won't be scanned.", ephemeral=True
        )

    @nsfw.command(name="unwhitelist-user", description="Remove a user from the scan whitelist")
    @_requires_manage_messages()
    async def nsfw_unwhitelist_user(self, interaction: discord.Interaction, user: discord.Member) -> None:
        data = _load_config()
        cfg = _get_guild(data, str(interaction.guild_id))
        if user.id not in cfg["whitelisted_users"]:
            await interaction.response.send_message(
                f"\u2139\ufe0f {user.mention} is not whitelisted.", ephemeral=True
            )
            return
        cfg["whitelisted_users"].remove(user.id)
        _save_config(data)
        await interaction.response.send_message(
            f"\u2705 {user.mention} removed from whitelist.", ephemeral=True
        )

    # ── Clear cache ───────────────────────────────────────────────────────────

    @nsfw.command(name="clear-cache", description="Wipe the image hash cache (forces full rescans)")
    @_requires_manage_messages()
    async def nsfw_clear_cache(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True)
        try:
            from utils.database import clear_hash_cache
            from config import config as bot_config
            deleted = await clear_hash_cache(bot_config.sqlite_db_path)
            await interaction.followup.send(
                f"\U0001f5d1\ufe0f Cleared **{deleted:,}** cached hash(es). All new media will be fully rescanned.",
                ephemeral=True,
            )
        except Exception as e:
            logger.error("nsfw clear-cache error: %s", e, exc_info=True)
            await interaction.followup.send(f"\u274c Failed: {e}", ephemeral=True)

    # ── Per-user stats ────────────────────────────────────────────────────────

    @nsfw.command(name="user-stats", description="Compact scan history for a specific user")
    @_requires_manage_messages()
    async def nsfw_user_stats(self, interaction: discord.Interaction, user: discord.Member) -> None:
        await interaction.response.defer(ephemeral=True)
        try:
            from utils.database import get_scan_stats_by_user
            stats = await get_scan_stats_by_user(str(interaction.guild_id), str(user.id))
            total = stats["total_scans"]
            if total == 0:
                await interaction.followup.send(
                    f"\U0001f4ca No scans logged for {user.mention}.", ephemeral=True
                )
                return
            bd = stats["verdict_breakdown"]
            bd_str = " \u00b7 ".join(
                f"`{v}:{c}`" for v, c in sorted(bd.items(), key=lambda x: -x[1])
            )
            last_str = f"`{stats['last_scan_at']}`" if stats.get("last_scan_at") else "unknown"
            blocked = stats["blocked"]
            reviewed = stats["reviewed"]
            safe = stats["safe"]
            embed = discord.Embed(
                title=f"\U0001f464 {user.display_name}",
                color=0x6366F1,
                timestamp=discord.utils.utcnow(),
            )
            embed.set_thumbnail(url=user.display_avatar.url)
            embed.add_field(
                name="Scans",
                value=f"**{total}** \u00b7 \U0001f6a8 {blocked} \u00b7 \u26a0\ufe0f {reviewed} \u00b7 \u2705 {safe}",
                inline=False,
            )
            embed.add_field(name="Verdicts", value=bd_str or "\u2014", inline=False)
            embed.add_field(name="Last Seen", value=last_str, inline=True)
            await interaction.followup.send(embed=embed, ephemeral=True)
        except Exception as e:
            logger.error("nsfw user-stats error: %s", e, exc_info=True)
            await interaction.followup.send(f"\u274c Error: {e}", ephemeral=True)


# ── Standalone ping command ────────────────────────────────────────────────────

class PingCog(commands.Cog):
    """Lightweight latency/uptime check."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    @app_commands.command(name="ping", description="Bot latency and uptime")
    async def ping(self, interaction: discord.Interaction) -> None:
        latency_ms = round(self.bot.latency * 1000)
        uptime_str = "unknown"
        if hasattr(self.bot, "start_time"):
            elapsed = int(time.time() - self.bot.start_time)
            days, rem = divmod(elapsed, 86400)
            hours, rem = divmod(rem, 3600)
            mins, secs = divmod(rem, 60)
            parts = (
                ([f"{days}d"] if days else [])
                + ([f"{hours}h"] if hours else [])
                + ([f"{mins}m"] if mins else [])
                + [f"{secs}s"]
            )
            uptime_str = " ".join(parts)
        color = 0x22C55E if latency_ms < 100 else 0xF59E0B if latency_ms < 250 else 0xEF4444
        embed = discord.Embed(
            title="\U0001f3d3 Pong!",
            description=f"**Latency:** `{latency_ms}ms`\n**Uptime:** `{uptime_str}`",
            color=color,
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)


# ── Setup ─────────────────────────────────────────────────────────────────────

async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(NSFWAdminCog(bot))
    await bot.add_cog(PingCog(bot))
