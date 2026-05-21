"""
bot/cogs/automod.py — NSFW Moderation cog.

Listens for messages in monitored channels, runs the local AI pipeline,
and takes action based on SAFE / SUGGESTIVE / REVIEW / NSFW / BLOCK / EXPLICIT verdicts.

Per-guild config (punishment, channels, whitelist, log_channel) is stored
in automod_config.json.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta
from io import BytesIO
from typing import Optional

import aiohttp
import discord
from discord.ext import commands
from config import config as bot_config
from bot.ui.feedback_view import ModerationFeedbackView
from moderation.feedback_manager import _extract_model_scores, _extract_detected_tags

logger = logging.getLogger(__name__)

_EMBED_FIELD_LIMIT = 1024


# ── Verdict severity helper ────────────────────────────────────────────────────
_SEVERITY = {
    "SAFE": 0,
    "SUGGESTIVE": 1,
    "REVIEW": 1,
    "NSFW": 2,
    "BLOCK": 2,
    "EXPLICIT": 3,
}
_BLOCKING_VERDICTS = {"BLOCK", "NSFW", "EXPLICIT"}
_REVIEW_VERDICTS = {"REVIEW", "SUGGESTIVE"}


def _trim_embed_value(value: str, limit: int = _EMBED_FIELD_LIMIT) -> str:
    """Trim a Discord embed field value to the 1024-character hard limit."""
    if len(value) <= limit:
        return value
    suffix = "\n... [truncated]"
    return value[: max(0, limit - len(suffix))] + suffix


def _code_field_value(text: str, lang: str = "") -> str:
    """Wrap text in a code block while reserving room for Discord's field limit."""
    open_fence = f"```{lang}\n"
    close_fence = "\n```"
    budget = _EMBED_FIELD_LIMIT - len(open_fence) - len(close_fence)
    return f"{open_fence}{_trim_embed_value(text, budget)}{close_fence}"


def _find_trace_value(lines: list[str], label: str) -> Optional[str]:
    prefix = f"{label}:"
    for line in lines:
        clean = line.strip()
        if clean.startswith(prefix):
            return clean[len(prefix):].strip()
    return None


def _human_trace_value(pipeline_steps: list[str]) -> str:
    """Convert verbose model trace strings into a short moderator-readable summary."""
    sections: list[str] = []

    for step in pipeline_steps:
        lines = [line.strip() for line in step.splitlines() if line.strip()]
        if not lines:
            continue

        title = lines[0].replace("Stage 0: ", "").replace("Stage 1: ", "").replace("Stage 2B: ", "").replace("Stage 2A: ", "").replace("Stage 2: ", "")

        if "Pre-filter" in title:
            score = _find_trace_value(lines, "NSFW Score") or "unknown"
            verdict = _find_trace_value(lines, "Verdict") or "unknown"
            sections.append(f"**Pre-filter:** NSFW score `{score}`. {verdict}.")
            continue

        if "Gatekeeper" in title:
            route = _find_trace_value(lines, "Classification") or "unknown"
            confidence = _find_trace_value(lines, "Confidence") or "unknown"
            sections.append(f"**Content type:** `{route}` with `{confidence}` confidence.")
            continue

        if "Anime" in title:
            wdv3 = _find_trace_value(lines, "- wdv3_explicit") or "0"
            r18 = _find_trace_value(lines, "- anime_rating_r18") or "0"
            genital = _find_trace_value(lines, "- genital_score") or "0"
            breast = _find_trace_value(lines, "- breast_score") or "0"
            rating = _find_trace_value(lines, "Tagger Rating") or "unknown"
            verdict = _find_trace_value(lines, "Verdict") or "unknown"
            sections.append(
                "**Anime detector:** "
                f"rating `{rating}`, explicit `{wdv3.split()[0]}`, r18 `{r18.split()[0]}`, "
                f"genitals `{genital.split()[0]}`, breasts `{breast.split()[0]}`. Verdict: `{verdict}`."
            )
            continue

        if "Real/Photo" in title:
            detections = _find_trace_value(lines, "Detected Explicit Labels") or "see log details"
            verdict = _find_trace_value(lines, "Verdict") or "unknown"
            sections.append(f"**Real-photo detector:** {detections}. Verdict: `{verdict}`.")
            continue

        verdict = _find_trace_value(lines, "Verdict")
        if verdict:
            sections.append(f"**{title}:** Verdict `{verdict}`.")

    if not sections:
        return "No detailed model trace was recorded."

    return _trim_embed_value("\n".join(sections))


class RemoveTimeoutView(discord.ui.View):
    """Button view to remove timeout from the log embed."""

    def __init__(self, user_id: int) -> None:
        super().__init__(timeout=None)
        self.user_id = user_id

    @discord.ui.button(label="Remove Timeout", style=discord.ButtonStyle.green, emoji="✅")
    async def remove_timeout_button(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ) -> None:
        try:
            if not interaction.user.guild_permissions.moderate_members:
                return await interaction.response.send_message(
                    "❌ You don't have permission to do this!", ephemeral=True
                )
            member = interaction.guild.get_member(self.user_id)
            if not member:
                return await interaction.response.send_message(
                    "❌ User not found in server!", ephemeral=True
                )
            await member.timeout(None, reason=f"Timeout removed by {interaction.user.name}")

            original_embed = interaction.message.embeds[0]
            original_embed.add_field(
                name="⚠️ Timeout Status",
                value=f"**Removed by:** {interaction.user.mention}\n"
                      f"**At:** <t:{int(discord.utils.utcnow().timestamp())}:F>",
                inline=False,
            )
            original_embed.color = 0x808080
            button.disabled = True
            button.label = "Timeout Removed"
            button.style = discord.ButtonStyle.gray
            await interaction.response.edit_message(embed=original_embed, view=self)
            await interaction.followup.send(
                f"✅ Removed timeout from {member.mention}", ephemeral=True
            )
        except Exception as e:
            logger.error("Failed to remove timeout: %s", e)
            await interaction.response.send_message(
                f"❌ Failed to remove timeout: {e}", ephemeral=True
            )


class AutoMod(commands.Cog):
    """
    NSFW auto-moderation cog.

    Uses the local AI pipeline (moderation/pipeline.py) instead of the FastAPI endpoint.
    Per-guild config (punishment, threshold, whitelist, log_channel) is preserved from
    automod_config.json for backwards compatibility.
    """

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self.config_file = "automod_config.json"
        self._guild_locks: dict[int, asyncio.Lock] = {}
        self.load_config()
        logger.info("🤖 AutoMod initialized with local AI pipeline")

    # ── Config helpers ────────────────────────────────────────────────────────

    def load_config(self) -> None:
        try:
            with open(self.config_file, "r") as f:
                self.config = json.load(f)
        except FileNotFoundError:
            self.config = {}

    def save_config(self) -> None:
        with open(self.config_file, "w") as f:
            json.dump(self.config, f, indent=4)

    def get_server_config(self, guild_id: int) -> dict:
        gid = str(guild_id)
        if gid not in self.config:
            self.config[gid] = {
                "enabled": True,
                "channels": [],
                "punishment": "timeout",
                "timeout_duration": 10,
                "ban_duration": None,
                "log_channel": None,
                "whitelisted_roles": [],
                "whitelisted_users": [],
            }
            self.save_config()

        defaults = {
            "enabled": True,
            "channels": [],
            "punishment": "timeout",
            "timeout_duration": 10,
            "ban_duration": None,
            "log_channel": None,
            "whitelisted_roles": [],
            "whitelisted_users": [],
        }
        for k, v in defaults.items():
            if k not in self.config[gid]:
                self.config[gid][k] = v
        self.save_config()
        return self.config[gid]

    def _guild_lock(self, guild_id: int) -> asyncio.Lock:
        if guild_id not in self._guild_locks:
            self._guild_locks[guild_id] = asyncio.Lock()
        return self._guild_locks[guild_id]

    # ── Image URL extraction ──────────────────────────────────────────────────

    def extract_media_urls(self, message: discord.Message) -> list[tuple[str, str, int]]:
        """
        Extract media URLs from attachments, embeds, and message content.
        Returns list of (url, filename, size_bytes).
        """
        results: list[tuple[str, str, int]] = []

        for att in message.attachments:
            # Only process image/gif/video content
            ct = att.content_type or ""
            if any(ct.startswith(prefix) for prefix in ("image/", "video/")):
                results.append((att.url, att.filename, att.size))
            elif att.filename.lower().split(".")[-1] in (
                "jpg", "jpeg", "png", "gif", "webp", "mp4", "webm", "mov", "avi"
            ):
                results.append((att.url, att.filename, att.size))

        for embed in message.embeds:
            if embed.image and embed.image.url:
                results.append((embed.image.url, "embed_image", 0))

        if message.content:
            patterns = [
                r"https?://[^\s]+\.(?:jpg|jpeg|png|gif|webp|mp4|webm|mov)",
                r"https?://cdn\.discordapp\.com/attachments/[^\s]+",
                r"https?://media\.discordapp\.net/attachments/[^\s]+",
                r"https?://i\.imgur\.com/[^\s]+",
            ]
            seen = {u for u, _, _ in results}
            for pat in patterns:
                for match in re.findall(pat, message.content, re.IGNORECASE):
                    clean = match.split("?")[0]
                    if clean not in seen:
                        results.append((clean, "linked_image", 0))
                        seen.add(clean)

        return results

    # ── Action helpers ────────────────────────────────────────────────────────

    async def _punish_user(
        self, member: discord.Member, reason: str, cfg: dict
    ) -> None:
        punishment = cfg.get("punishment", "timeout")
        duration = cfg.get("timeout_duration", 10)
        ban_duration = cfg.get("ban_duration")

        try:
            if punishment == "kick":
                await member.kick(reason=f"AI: {reason}")
                await self._log_action(member, "Kick", reason)
            elif punishment == "ban":
                ban_reason = f"AI: {reason}" + (f" (Temp: {ban_duration}d)" if ban_duration else "")
                await member.ban(reason=ban_reason, delete_message_days=0)
                await self._log_action(member, "Ban", reason)
            elif punishment == "timeout":
                until = discord.utils.utcnow() + timedelta(minutes=duration)
                await member.timeout(until, reason=f"AI: {reason}")
                await self._log_action(member, "Timeout", reason)
            elif punishment == "none":
                await self._log_action(member, "Warning", reason)
        except Exception as e:
            logger.error("Punishment failed for %s: %s", member, e)

    async def _log_action(self, user: discord.User, action: str, reason: str) -> None:
        logger.info("Action taken on %s: %s - %s", user, action, reason)

    async def _send_dm(
        self, user: discord.User, guild_name: str
    ) -> None:
        try:
            embed = discord.Embed(
                title="📋 Content Removed",
                description=(
                    f"Hey! A file you posted in **{guild_name}** was automatically removed "
                    f"because it appears to contain content that violates the server's "
                    f"explicit content policy.\n\n"
                    f"If you believe this was a mistake, please contact a server moderator."
                ),
                color=0xF59E0B,
            )
            embed.set_footer(text="This is an automated message from the moderation system.")
            await user.send(embed=embed)
        except discord.Forbidden:
            pass  # DMs disabled
        except Exception as e:
            logger.warning("Failed to DM user %s: %s", user, e)

    async def _send_log_embed(
        self,
        message: discord.Message,
        scan_result,
        file_info: list[tuple[str, str, int]],
        verdict: str,
        cfg: dict,
    ) -> None:
        """Send a detailed log embed to the configured log channel with feedback buttons."""
        log_channel_id = bot_config.log_channel_id or cfg.get("log_channel")
        if not log_channel_id:
            logger.warning("NSFW log channel is not configured; cannot send moderation log")
            return

        log_channel = message.guild.get_channel(int(log_channel_id))
        if not log_channel:
            try:
                log_channel = await message.guild.fetch_channel(int(log_channel_id))
            except Exception as e:
                logger.warning("Could not fetch NSFW log channel %s: %s", log_channel_id, e)
        if not log_channel:
            logger.warning("NSFW log channel %s was not found in guild %s", log_channel_id, message.guild.id)
            return

        member = message.author
        now = discord.utils.utcnow()

        # ── Enhanced color coding ──────────────────────────────────────────────
        _COLOR_MAP = {
            "SAFE":      0x22C55E,  # green
            "SUGGESTIVE": 0xF59E0B, # yellow
            "REVIEW":    0xF59E0B,  # yellow
            "NSFW":      0xEF4444,  # orange-red
            "BLOCK":     0xEF4444,  # orange-red
            "EXPLICIT":  0xDC2626,  # deep red
        }
        color = _COLOR_MAP.get(scan_result.verdict, 0xFF4444)

        title = (
            "🚨 NSFW Content Detected — BLOCKED"
            if verdict == "BLOCK"
            else "⚠️ [REVIEW NEEDED] Possible NSFW Content"
        )

        embed = discord.Embed(title=title, color=color, timestamp=now)
        embed.set_thumbnail(url=member.display_avatar.url)

        # ── User & context fields ──────────────────────────────────────────────
        embed.add_field(name="👤 User", value=f"{member.mention} (`{member.id}`)", inline=True)
        embed.add_field(name="📺 Channel", value=message.channel.mention, inline=True)
        embed.add_field(name="💬 Message ID", value=f"`{message.id}`", inline=True)

        acct_ts = int(member.created_at.timestamp())
        embed.add_field(
            name="🗓️ Account",
            value=f"**Created:** <t:{acct_ts}:R>\n**Name:** {member.name}",
            inline=True,
        )

        if member.joined_at:
            join_ts = int(member.joined_at.timestamp())
            embed.add_field(
                name="🏠 Server",
                value=f"**Joined:** <t:{join_ts}:R>",
                inline=True,
            )

        # ── AI Detection Summary ───────────────────────────────────────────────
        # Verdict badge
        verdict_badge = {
            "EXPLICIT": "🔴 EXPLICIT",
            "NSFW":     "🟠 NSFW",
            "BLOCK":    "🟠 BLOCKED",
            "REVIEW":   "🟡 REVIEW",
            "SUGGESTIVE": "🟡 SUGGESTIVE",
            "SAFE":     "🟢 SAFE",
        }.get(scan_result.verdict, scan_result.verdict)

        # Content type / branch label
        branch_label = {
            "real":  "📷 Real/Photo",
            "anime": "🎨 Anime/Illustration",
            "both":  "🔀 Both (Uncertain)",
        }.get(scan_result.branch, scan_result.branch)

        # Confidence from reason string (best effort)
        confidence_str = ""
        if scan_result.reason and "score=" in scan_result.reason:
            try:
                confidence_str = f"\n**Confidence:** `{scan_result.reason.split('score=')[1].split()[0].rstrip(',)')}`"
            except Exception:
                pass

        embed.add_field(
            name="🤖 AI Detection Summary",
            value=_trim_embed_value(
                f"**Verdict:** {verdict_badge}\n"
                f"**Content Type:** {branch_label}\n"
                f"**Model:** `{scan_result.model}`\n"
                f"**Processing Time:** `{scan_result.processing_time_ms:.0f}ms`"
                f"{confidence_str}"
                + (f"\n**Triggered Frame:** `#{scan_result.frame_index}`" if scan_result.frame_index is not None else "")
            ),
            inline=False,
        )

        # ── Detected tags / labels ─────────────────────────────────────────────
        detected_tags = _extract_detected_tags(scan_result)
        if detected_tags:
            tag_lines = [f"`{tag}` — `{score:.2f}`" for tag, score in detected_tags[:8]]
            embed.add_field(
                name="🔍 Detected Tags / Labels",
                value=_trim_embed_value("\n".join(tag_lines)),
                inline=False,
            )

        # ── Reason / model trace ───────────────────────────────────────────────
        if scan_result.reason:
            embed.add_field(
                name="📊 Reason",
                value=_trim_embed_value(scan_result.reason),
                inline=False,
            )

        if getattr(scan_result, "pipeline_steps", None):
            embed.add_field(
                name="📋 Model Decision Summary",
                value=_human_trace_value(scan_result.pipeline_steps),
                inline=False,
            )

        if message.content:
            embed.add_field(
                name="💬 Message Content",
                value=_trim_embed_value(message.content),
                inline=False,
            )

        if file_info:
            lines = []
            for url, fname, size in file_info[:5]:
                size_str = f" `({size:,} bytes)`" if size else ""
                lines.append(f"[{fname}]({url}){size_str}")
            embed.add_field(name="📎 Attachments", value=_trim_embed_value("\n".join(lines)), inline=False)

        embed.set_footer(text="NSFW Detection System • Local AI Pipeline | Use buttons below to submit feedback")

        # ── Build combined view: RemoveTimeout + Feedback buttons ─────────────
        # Serialize scan data for the feedback view to store on click
        try:
            model_scores = _extract_model_scores(scan_result)
            scan_result_payload = json.dumps({
                "branch": scan_result.branch,
                "model": scan_result.model,
                "processing_time_ms": scan_result.processing_time_ms,
                "model_scores": model_scores,
                "detected_tags": _extract_detected_tags(scan_result),
            })
        except Exception:
            scan_result_payload = "{}"

        feedback_view = ModerationFeedbackView(
            message_id=str(message.id),
            user_id=str(member.id),
            channel_id=str(message.channel.id),
            guild_id=str(message.guild.id),
            predicted_verdict=scan_result.verdict,
            scan_result_json=scan_result_payload,
        )

        # If there's also a timeout-removal button, we inject it into the feedback view
        if verdict == "BLOCK" and cfg.get("punishment") == "timeout":
            # Add the remove-timeout button to the feedback view
            remove_btn = discord.ui.Button(
                label="Remove Timeout",
                style=discord.ButtonStyle.success,
                emoji="✅",
                custom_id=f"remove_timeout_{member.id}",
                row=1,
            )

            async def _remove_timeout_callback(interaction: discord.Interaction) -> None:
                try:
                    if not interaction.user.guild_permissions.moderate_members:
                        return await interaction.response.send_message(
                            "❌ You don't have permission to do this!", ephemeral=True
                        )
                    target_member = interaction.guild.get_member(member.id)
                    if not target_member:
                        return await interaction.response.send_message(
                            "❌ User not found in server!", ephemeral=True
                        )
                    await target_member.timeout(None, reason=f"Timeout removed by {interaction.user.name}")
                    original_embed = interaction.message.embeds[0]
                    original_embed.add_field(
                        name="⚠️ Timeout Status",
                        value=f"**Removed by:** {interaction.user.mention}\n"
                              f"**At:** <t:{int(discord.utils.utcnow().timestamp())}:F>",
                        inline=False,
                    )
                    original_embed.color = 0x808080
                    remove_btn.disabled = True
                    remove_btn.label = "Timeout Removed"
                    remove_btn.style = discord.ButtonStyle.gray
                    await interaction.response.edit_message(embed=original_embed, view=feedback_view)
                    await interaction.followup.send(
                        f"✅ Removed timeout from {target_member.mention}", ephemeral=True
                    )
                except Exception as e:
                    logger.error("Failed to remove timeout: %s", e)
                    await interaction.response.send_message(
                        f"❌ Failed to remove timeout: {e}", ephemeral=True
                    )

            remove_btn.callback = _remove_timeout_callback
            feedback_view.add_item(remove_btn)

        try:
            log_msg = await log_channel.send(embed=embed, view=feedback_view)
        except discord.HTTPException as e:
            logger.error("Failed to send rich NSFW log embed; sending compact fallback: %s", e)
            fallback = discord.Embed(
                title=title,
                color=color,
                timestamp=now,
                description=_trim_embed_value(
                    f"**User:** {member.mention} (`{member.id}`)\n"
                    f"**Channel:** {message.channel.mention}\n"
                    f"**Verdict:** `{scan_result.verdict}`\n"
                    f"**Reason:** {scan_result.reason}\n"
                    f"**Branch:** `{scan_result.branch}`\n"
                    f"**Model:** `{scan_result.model}`"
                ),
            )
            log_msg = await log_channel.send(embed=fallback, view=feedback_view)

        # Send image preview as spoiler
        if file_info:
            try:
                first_url, first_fname, _ = file_info[0]
                async with aiohttp.ClientSession() as session:
                    async with session.get(first_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        if resp.status == 200:
                            data = await resp.read()
                            preview_file = discord.File(
                                fp=BytesIO(data),
                                filename=f"SPOILER_{first_fname}",
                                spoiler=True,
                            )
                            preview_embed = discord.Embed(
                                description="⚠️ **Flagged Content Preview** (Click to reveal)",
                                color=0xFF0000,
                            )
                            await log_channel.send(embed=preview_embed, file=preview_file)
            except Exception as e:
                logger.debug("Could not send image preview: %s", e)

    async def _send_debug_log_embed(
        self,
        message: discord.Message,
        scan_result,
        url: str,
        fname: str,
        size: int,
    ) -> None:
        """Send a detailed debug log embed with statistics for EVERY scan."""
        try:
            debug_channel_id = bot_config.debug_log_channel_id
            if not debug_channel_id:
                return

            debug_channel = self.bot.get_channel(debug_channel_id)
            if not debug_channel:
                try:
                    debug_channel = await self.bot.fetch_channel(debug_channel_id)
                except Exception:
                    pass
            if not debug_channel:
                return

            member = message.author
            now = discord.utils.utcnow()

            # Choose color based on verdict
            color_map = {
                "SAFE": 0x22C55E,
                "SUGGESTIVE": 0xF59E0B,
                "REVIEW": 0xF59E0B,
                "NSFW": 0xEF4444,
                "BLOCK": 0xEF4444,
                "EXPLICIT": 0xDC2626,
            }
            color = color_map.get(scan_result.verdict, 0x888888)

            verdict_emoji = {
                "SAFE": "✅ SAFE",
                "SUGGESTIVE": "⚠️ SUGGESTIVE",
                "REVIEW": "⚠️ REVIEW NEEDED",
                "NSFW": "🚨 NSFW",
                "BLOCK": "🚨 BLOCKED",
                "EXPLICIT": "🚨 EXPLICIT",
            }.get(scan_result.verdict, "❓ UNKNOWN")

            embed = discord.Embed(
                title=f"🔍 Scan Debug Stats — {verdict_emoji}",
                color=color,
                timestamp=now
            )
            embed.set_thumbnail(url=member.display_avatar.url)

            # Basic metadata
            embed.add_field(name="👤 User", value=f"{member.mention}\n`{member.id}`", inline=True)
            embed.add_field(name="📺 Channel", value=message.channel.mention, inline=True)
            embed.add_field(name="💬 Message ID", value=f"`{message.id}`", inline=True)

            # AI Stats
            embed.add_field(
                name="🧠 AI Council Engine",
                value=(
                    f"**Verdict:** `{scan_result.verdict}`\n"
                    f"**Engine/Model:** `{scan_result.model}`\n"
                    f"**Active Branch:** `{scan_result.branch}`\n"
                    f"**Processing Time:** `{scan_result.processing_time_ms:.1f}ms`"
                ),
                inline=False
            )

            embed.add_field(
                name="📊 Details & Confidence",
                value=_code_field_value(str(scan_result.reason or "")),
                inline=False
            )

            if getattr(scan_result, "pipeline_steps", None):
                embed.add_field(
                    name="📋 Model Decision Summary",
                    value=_human_trace_value(scan_result.pipeline_steps),
                    inline=False,
                )

            # File Info
            size_str = f"{size:,} bytes" if size else "Unknown"
            embed.add_field(
                name="📎 Media Info",
                value=_trim_embed_value(f"**Name:** `{fname}`\n**Size:** `{size_str}`\n**Link:** [Open original file]({url})"),
                inline=False
            )

            # Set image preview
            preview_file = None
            if any(fname.lower().endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp", ".gif")):
                if scan_result.verdict == "SAFE":
                    embed.set_image(url=url)
                else:
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                                if resp.status == 200:
                                    data = await resp.read()
                                    safe_fname = fname if fname.upper().startswith("SPOILER_") else f"SPOILER_{fname}"
                                    preview_file = discord.File(
                                        fp=BytesIO(data),
                                        filename=safe_fname,
                                        spoiler=True,
                                    )
                                    embed.set_image(url=f"attachment://{safe_fname}")
                    except Exception as e:
                        logger.debug("Could not download debug image for spoiler: %s", e)

            # ── Build feedback buttons view ───────────────────────────────────
            try:
                model_scores = _extract_model_scores(scan_result)
                scan_result_payload = json.dumps({
                    "branch": scan_result.branch,
                    "model": scan_result.model,
                    "processing_time_ms": scan_result.processing_time_ms,
                    "model_scores": model_scores,
                    "detected_tags": _extract_detected_tags(scan_result),
                })
            except Exception:
                scan_result_payload = "{}"

            feedback_view = ModerationFeedbackView(
                message_id=str(message.id),
                user_id=str(member.id),
                channel_id=str(message.channel.id),
                guild_id=str(message.guild.id),
                predicted_verdict=scan_result.verdict,
                scan_result_json=scan_result_payload,
            )

            embed.set_footer(text="NSFW Bot Debugging Logger • Local Inference | Use buttons below to submit feedback")
            if preview_file:
                await debug_channel.send(embed=embed, file=preview_file, view=feedback_view)
            else:
                await debug_channel.send(embed=embed, view=feedback_view)
        except Exception as e:
            logger.error("Failed to send debug log embed: %s", e, exc_info=True)

    # ── Main listener ─────────────────────────────────────────────────────────

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        """Scan incoming messages for NSFW media."""
        try:
            if message.author.bot or not message.guild:
                return

            cfg = self.get_server_config(message.guild.id)
            if not cfg.get("enabled", True):
                return

            # Channel monitoring: if cfg["channels"] is non-empty, only scan those.
            # If empty, scan all channels in this guild (global enable behaviour).
            guild_channels = cfg.get("channels", [])
            if guild_channels and message.channel.id not in guild_channels:
                return

            # Check whitelist — roles
            member_role_ids = [r.id for r in message.author.roles]
            if any(rid in cfg.get("whitelisted_roles", []) for rid in member_role_ids):
                return

            # Check whitelist — users
            if message.author.id in cfg.get("whitelisted_users", []):
                return

            # Extract media
            media_list = self.extract_media_urls(message)
            if not media_list:
                return

            logger.info(
                "📎 %s posted %d media item(s) in #%s",
                message.author,
                len(media_list),
                message.channel.name,
            )

            # Per-guild lock to prevent concurrent GPU inference
            async with self._guild_lock(message.guild.id):
                from moderation.pipeline import scan_attachment

                best_result = None
                best_file_info = None

                for url, fname, size in media_list:
                    try:
                        result = await scan_attachment(url, bot_config)
                    except Exception as e:
                        logger.error("Pipeline error for %s: %s", url, e, exc_info=True)
                        continue

                    logger.info(
                        "🔍 [%s] %s → %s (%s)",
                        message.author.name,
                        fname,
                        result.verdict,
                        result.reason,
                    )

                    # Send to debug channel if configured, even if SAFE
                    if bot_config.debug_log_channel_id:
                        await self._send_debug_log_embed(message, result, url, fname, size)

                    if best_result is None or _SEVERITY.get(result.verdict, 0) > _SEVERITY.get(best_result.verdict, 0):
                        best_result = result
                        best_file_info = [(url, fname, size)]

                    if result.verdict in _BLOCKING_VERDICTS:
                        break  # Stop processing on first block

                if best_result is None or best_result.verdict == "SAFE":
                    return

                # Take action based on verdict
                file_info = best_file_info or []

                if best_result.verdict in _BLOCKING_VERDICTS:
                    # Delete message
                    try:
                        await message.delete()
                        logger.info("🗑️ Deleted message from %s", message.author)
                    except Exception as e:
                        logger.warning("Could not delete message: %s", e)

                    # DM user
                    await self._send_dm(message.author, message.guild.name)

                    # Punish
                    try:
                        member = message.guild.get_member(message.author.id)
                        if member:
                            await self._punish_user(member, best_result.reason, cfg)
                    except Exception as e:
                        logger.error("Punishment error: %s", e)

                    # Log
                    await self._send_log_embed(message, best_result, file_info, "BLOCK", cfg)

                elif best_result.verdict in _REVIEW_VERDICTS:
                    # Do NOT delete — just log for human review
                    await self._send_log_embed(message, best_result, file_info, "REVIEW", cfg)
                    logger.info(
                        "⚠️ REVIEW flagged for %s in #%s",
                        message.author,
                        message.channel.name,
                    )

        except Exception as e:
            logger.error("AutoMod on_message error: %s", e, exc_info=True)

    # ── Legacy scanner commands (preserved for backwards compatibility) ────────

    @commands.hybrid_group(invoke_without_command=True, name="scanner", description="NSFW Scanner configuration")
    @commands.has_permissions(manage_messages=True)
    async def scanner(self, ctx: commands.Context) -> None:
        if ctx.invoked_subcommand is None:
            cfg = self.get_server_config(ctx.guild.id)
            status = "🟢 Enabled" if cfg["enabled"] else "🔴 Disabled"

            log_channel_id = bot_config.log_channel_id or cfg.get("log_channel")
            log_channel = None
            if log_channel_id:
                log_channel = ctx.guild.get_channel(int(log_channel_id))
                if not log_channel:
                    try:
                        log_channel = await ctx.guild.fetch_channel(int(log_channel_id))
                    except Exception:
                        pass

            channels = cfg.get("channels", [])
            if channels:
                ch_list = ", ".join(
                    (ctx.guild.get_channel(c).mention if ctx.guild.get_channel(c) else str(c))
                    for c in channels
                )
            else:
                ch_list = "All channels"

            em = discord.Embed(title="🤖 NSFW Scanner (Local AI)", color=discord.Color.blue())
            em.add_field(name="Status", value=status, inline=True)
            em.add_field(name="Punishment", value=cfg["punishment"].title(), inline=True)
            em.add_field(name="Engine", value="Local AI Pipeline (4-model council)", inline=True)
            em.add_field(name="Monitoring", value=ch_list, inline=False)
            em.add_field(name="Log Channel", value=log_channel.mention if log_channel else "Not set", inline=True)
            em.set_footer(text="Use /scanner commands to configure | /nsfw for slash commands")
            await ctx.send(embed=em)

    @scanner.command(description="Enable/disable scanner")
    @commands.has_permissions(manage_messages=True)
    async def toggle(self, ctx: commands.Context) -> None:
        cfg = self.get_server_config(ctx.guild.id)
        cfg["enabled"] = not cfg["enabled"]
        self.save_config()
        await ctx.send(f"✅ Scanner {'enabled' if cfg['enabled'] else 'disabled'}")

    @scanner.command(description="Set punishment type (none/kick/ban/timeout)")
    @commands.has_permissions(manage_messages=True)
    async def punishment(self, ctx: commands.Context, ptype: str) -> None:
        if ptype.lower() not in ["none", "kick", "ban", "timeout"]:
            return await ctx.send("❌ Use: none, kick, ban, timeout", ephemeral=True)
        cfg = self.get_server_config(ctx.guild.id)
        cfg["punishment"] = ptype.lower()
        self.save_config()
        await ctx.send(f"✅ Punishment: {ptype.lower()}")

    @scanner.command(name="logchannel", description="Set log channel for NSFW detections")
    @commands.has_permissions(manage_messages=True)
    async def log_channel(self, ctx: commands.Context, channel: discord.TextChannel = None) -> None:
        cfg = self.get_server_config(ctx.guild.id)
        if channel is None:
            cfg["log_channel"] = None
            self.save_config()
            await ctx.send("✅ Log channel disabled")
        else:
            cfg["log_channel"] = channel.id
            self.save_config()
            em = discord.Embed(
                title="✅ Log Channel Set",
                description=f"NSFW detection logs will be sent to {channel.mention}",
                color=discord.Color.green(),
            )
            await ctx.send(embed=em)


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(AutoMod(bot))
