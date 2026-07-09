

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
from utils.constants import SEVERITY as _SEVERITY, BLOCKING_VERDICTS as _BLOCKING_VERDICTS, REVIEW_VERDICTS as _REVIEW_VERDICTS
import contextlib

logger = logging.getLogger(__name__)
_EMBED_FIELD_LIMIT: int = 1024  # Discord embed field character hard limit


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
        self._user_scan_counts: dict[int, int] = {}
        self._user_rate_lock: asyncio.Lock = asyncio.Lock()
        self._http_session: Optional[aiohttp.ClientSession] = None
        self.load_config()
        logger.info("🤖 AutoMod initialized with local AI pipeline")

    async def cog_unload(self) -> None:
        """Close the shared HTTP session on cog unload."""
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()

    async def _get_http_session(self) -> aiohttp.ClientSession:
        """Return (or create) a persistent aiohttp session for preview downloads."""
        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession()
        return self._http_session


    def load_config(self) -> None:
        try:
            with open(self.config_file, "r") as f:
                self.config = json.load(f)
        except FileNotFoundError:
            self.config = {}

    def save_config(self) -> None:
        import os, tempfile
        dir_name = os.path.dirname(os.path.abspath(self.config_file)) or "."
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", dir=dir_name, suffix=".tmp", delete=False
            ) as tmp:
                json.dump(self.config, tmp, indent=4)
                tmp_path = tmp.name
            os.replace(tmp_path, self.config_file)
        except Exception:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
            raise

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
        changed = False
        for k, v in defaults.items():
            if k not in self.config[gid]:
                self.config[gid][k] = v
                changed = True
        if changed:
            self.save_config()
        return self.config[gid]

    def _guild_lock(self, guild_id: int) -> asyncio.Lock:
        if guild_id not in self._guild_locks:
            self._guild_locks[guild_id] = asyncio.Lock()
        return self._guild_locks[guild_id]

    @contextlib.asynccontextmanager
    async def _scan_slot(self, user_id: int):
        MAX_CONCURRENT_SCANS = 3
        async with self._user_rate_lock:
            active = self._user_scan_counts.get(user_id, 0)
            if active >= MAX_CONCURRENT_SCANS:
                raise ValueError("Rate limit exceeded")
            self._user_scan_counts[user_id] = active + 1
        
        try:
            yield
        finally:
            async with self._user_rate_lock:
                self._user_scan_counts[user_id] -= 1
                if self._user_scan_counts[user_id] <= 0:
                    self._user_scan_counts.pop(user_id, None)


    @staticmethod
    def _is_discord_cdn_url(url: str) -> bool:
        """True if url is a Discord attachment CDN/media URL."""
        return (
            "cdn.discordapp.com/attachments/" in url
            or "media.discordapp.net/attachments/" in url
        )

    @staticmethod
    def _discord_url_has_auth(url: str) -> bool:
        """True if the Discord CDN URL contains the required expiring auth params."""
        return "ex=" in url and "is=" in url

    def extract_media_urls(self, message: discord.Message) -> list[tuple[str, str, int]]:
        """
        Extract scannable media URLs from attachments, embeds, and message text.
        Returns list of (url, filename, size_bytes).

        Discord CDN URLs require expiring auth params (ex=, is=, hm=).
        Bare CDN links pasted as text almost never have them, so we:
          1. Prefer direct message.attachments (always signed by discord.py).
          2. Use embed.image.proxy_url if it has auth params.
          3. For text-pasted CDN links: only accept them if they carry auth params.
             Otherwise skip — they will 404 on download.
          4. Non-Discord external URLs (imgur, etc.) are always accepted as-is.
        """
        results: list[tuple[str, str, int]] = []

        for att in message.attachments:
            ct = att.content_type or ""
            if any(ct.startswith(prefix) for prefix in ("image/", "video/")):
                results.append((att.url, att.filename, att.size))
            elif att.filename.lower().split(".")[-1] in (
                "jpg", "jpeg", "png", "gif", "webp", "mp4", "webm", "mov", "avi"
            ):
                results.append((att.url, att.filename, att.size))

        # proxy_url is Discord's own cached copy; it may or may not carry auth.
        # Build a lookup so step 3 can upgrade bare text links to proxied URLs.
        embed_proxy_lookup: dict[str, str] = {}
        for embed in message.embeds:
            for img_obj in (embed.image, embed.thumbnail):
                if not img_obj or not img_obj.url:
                    continue
                clean_key = img_obj.url.split("?")[0]
                # Only use Discord CDN candidates that have auth params
                for candidate in (img_obj.proxy_url, img_obj.url):
                    if not candidate:
                        continue
                    if self._is_discord_cdn_url(candidate) and not self._discord_url_has_auth(candidate):
                        continue
                    embed_proxy_lookup[clean_key] = candidate
                    fname = "embed_gif" if (embed.type == "gifv" or "tenor.com" in candidate or "giphy.com" in candidate) else "embed_image"
                    results.append((candidate, fname, 0))
                    break

        # URLs found in message text
        if message.content:
            patterns = [
                r"https?://[^\s]+?\.(?:jpg|jpeg|png|gif|webp|mp4|webm|mov)(?:\?[^\s]*)?",
                r"https?://cdn\.discordapp\.com/attachments/[^\s]+",
                r"https?://media\.discordapp\.net/attachments/[^\s]+",
                r"https?://i\.imgur\.com/[^\s]+",
            ]
            seen_clean = {u.split("?")[0] for u, _, _ in results}
            for pat in patterns:
                for match in re.findall(pat, message.content, re.IGNORECASE):
                    clean = match.split("?")[0]
                    if clean in seen_clean:
                        continue

                    fname = "linked_image"
                    try:
                        path_part = clean.split("/")[-1]
                        if path_part and "." in path_part:
                            fname = path_part
                    except Exception:
                        pass

                    if clean in embed_proxy_lookup:
                        resolved = embed_proxy_lookup[clean]
                    elif self._is_discord_cdn_url(clean):
                        # Bare Discord CDN — only usable if auth params are present (without them CDN returns 404)
                        if self._discord_url_has_auth(match):
                            resolved = match
                        else:
                            logger.debug(
                                "⏩ Skipping bare Discord CDN link (no auth params): %s", clean
                            )
                            seen_clean.add(clean)
                            continue
                    else:
                        resolved = match

                    results.append((resolved, fname, 0))
                    seen_clean.add(clean)

        return results



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
        file_info: list[tuple],
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

        _COLOR_MAP = {
            "SAFE":      0x22C55E,  # green
            "SUGGESTIVE": 0xF59E0B, # yellow
            "REVIEW":    0xF59E0B,  # yellow
            "NSFW":      0xEF4444,  # orange-red
            "BLOCK":     0xEF4444,  # orange-red
            "EXPLICIT":  0xDC2626,  # deep red
        }
        color = _COLOR_MAP.get(scan_result.verdict, 0xFF4444)

        if verdict == "BLOCK":
            title = "🚨 NSFW Content Detected — BLOCKED"
        elif verdict == "REVIEW":
            title = "⚠️ [REVIEW NEEDED] Possible NSFW Content"
        else:
            title = "✅ Content Approved — SAFE"

        embed = discord.Embed(title=title, color=color, timestamp=now)
        embed.set_thumbnail(url=member.display_avatar.url)

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

        verdict_badge = {
            "EXPLICIT": "🔴 EXPLICIT",
            "NSFW":     "🟠 NSFW",
            "BLOCK":    "🟠 BLOCKED",
            "REVIEW":   "🟡 REVIEW",
            "SUGGESTIVE": "🟡 SUGGESTIVE",
            "SAFE":     "🟢 SAFE",
        }.get(scan_result.verdict, scan_result.verdict)

        branch_label = {
            "real":  "📷 Real/Photo",
            "anime": "🎨 Anime/Illustration",
            "both":  "🔀 Both (Uncertain)",
        }.get(scan_result.branch, scan_result.branch)

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

        detected_tags = _extract_detected_tags(scan_result)
        if detected_tags:
            tag_lines = [f"`{tag}` — `{score:.2f}`" for tag, score in detected_tags[:8]]
            embed.add_field(
                name="🔍 Detected Tags / Labels",
                value=_trim_embed_value("\n".join(tag_lines)),
                inline=False,
            )

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
            for item in file_info[:5]:
                url, fname, size = item[0], item[1], item[2]
                size_str = f" `({size:,} bytes)`" if size else ""
                lines.append(f"[{fname}]({url}){size_str}")
            embed.add_field(name="📎 Attachments", value=_trim_embed_value("\n".join(lines)), inline=False)

        embed.set_footer(text="NSFW Detection System • Local AI Pipeline | Use buttons below to submit feedback")

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

        if verdict == "BLOCK" and cfg.get("punishment") == "timeout":
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
            await log_channel.send(embed=embed, view=feedback_view)
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
            await log_channel.send(embed=fallback, view=feedback_view)

        if file_info:
            for item in file_info[:5]:
                url, fname, size, res = item[0], item[1], item[2], item[3]
                is_flagged = res.verdict in ("BLOCK", "REVIEW", "NSFW", "EXPLICIT", "SUGGESTIVE")
                if verdict == "SAFE" or is_flagged:
                    try:
                        _session = await self._get_http_session()
                        async with _session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                                if resp.status == 200:
                                    data = await resp.read()
                                    safe_fname = fname if fname.upper().startswith("SPOILER_") else f"SPOILER_{fname}"
                                    preview_file = discord.File(
                                        fp=BytesIO(data),
                                        filename=safe_fname,
                                        spoiler=True,
                                    )
                                    preview_embed = discord.Embed(
                                        description="⚠️ **Scanned Media Preview** (Click to reveal)" if verdict == "SAFE" else "⚠️ **Flagged Content Preview** (Click to reveal)",
                                        color=0x22C55E if verdict == "SAFE" else 0xFF0000,
                                    )
                                    await log_channel.send(embed=preview_embed, file=preview_file)
                                else:
                                    raise IOError(f"HTTP Status {resp.status}")
                    except Exception as e:
                        logger.warning("Could not download image preview for %s, sending placeholder: %s", fname, e)
                        try:
                            from utils.image_utils import generate_placeholder_image
                            placeholder_bytes = generate_placeholder_image()
                            placeholder_file = discord.File(
                                fp=BytesIO(placeholder_bytes),
                                filename=f"SPOILER_preview_unavailable_{fname}.png",
                                spoiler=True,
                            )
                            preview_embed = discord.Embed(
                                description="⚠️ **Flagged Content Preview** (Download Failed)",
                                color=0xFF0000,
                            )
                            await log_channel.send(embed=preview_embed, file=placeholder_file)
                        except Exception as placeholder_err:
                            logger.error("Failed to send placeholder image: %s", placeholder_err)

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
                except Exception as e:
                    logger.warning(
                        "⚠️ Configured DEBUG_LOG_CHANNEL_ID %s could not be fetched: %s. "
                        "Please verify that the ID is correct and the bot has permission to view and send messages in it.",
                        debug_channel_id,
                        e,
                    )
            if not debug_channel:
                logger.warning(
                    "⚠️ Configured DEBUG_LOG_CHANNEL_ID %s was not found in cache or fetched. "
                    "Debugging scans will not be logged to Discord.",
                    debug_channel_id,
                )
                return

            member = message.author
            now = discord.utils.utcnow()

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

            embed.add_field(name="👤 User", value=f"{member.mention}\n`{member.id}`", inline=True)
            embed.add_field(name="📺 Channel", value=message.channel.mention, inline=True)
            embed.add_field(name="💬 Message ID", value=f"`{message.id}`", inline=True)

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

            size_str = f"{size:,} bytes" if size else "Unknown"
            is_nsfw_verdict = scan_result.verdict != "SAFE"

            _url_path = url.split("?")[0].lower()
            _img_exts = (".jpg", ".jpeg", ".png", ".webp", ".gif")
            _vid_exts = (".mp4", ".webm", ".mov", ".avi")
            _is_image = any(_url_path.endswith(e) for e in _img_exts)
            _is_video = any(_url_path.endswith(e) for e in _vid_exts)

            if is_nsfw_verdict:
                # Redact raw URL for NSFW verdicts — moderators can find the original via Message ID
                media_info_value = _trim_embed_value(
                    f"**Name:** `{fname}`\n"
                    f"**Size:** `{size_str}`\n"
                    f"**Link:** `[redacted — see spoiler preview below]`"
                )
            else:
                media_info_value = _trim_embed_value(
                    f"**Name:** `{fname}`\n**Size:** `{size_str}`\n**Link:** [Open original file]({url})"
                )

            embed.add_field(name="📎 Media Info", value=media_info_value, inline=False)

            if not is_nsfw_verdict and _is_image:
                embed.set_image(url=url)

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
            
            await debug_channel.send(embed=embed, view=feedback_view)

            if is_nsfw_verdict:
                try:
                    _session = await self._get_http_session()
                    async with _session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                            if resp.status == 200:
                                data = await resp.read()
                                base_fname = fname
                                if not any(fname.lower().endswith(e) for e in _img_exts + _vid_exts):
                                    try:
                                        base_fname = _url_path.split("/")[-1] or fname
                                    except Exception:
                                        pass
                                safe_fname = (
                                    base_fname if base_fname.upper().startswith("SPOILER_")
                                    else f"SPOILER_{base_fname}"
                                )
                                preview_file = discord.File(
                                    fp=BytesIO(data),
                                    filename=safe_fname,
                                    spoiler=True,
                                )
                                preview_embed = discord.Embed(
                                    description="⚠️ **Flagged Content Preview** (Click to reveal)",
                                    color=0xFF0000,
                                )
                                await debug_channel.send(embed=preview_embed, file=preview_file)
                            else:
                                raise IOError(f"HTTP Status {resp.status}")
                except Exception as e:
                    logger.warning("Could not download debug image/video for spoiler, sending placeholder: %s", e)
                    try:
                        from utils.image_utils import generate_placeholder_image
                        placeholder_bytes = generate_placeholder_image()
                        placeholder_file = discord.File(
                            fp=BytesIO(placeholder_bytes),
                            filename=f"SPOILER_preview_unavailable_{fname}.png",
                            spoiler=True,
                        )
                        preview_embed = discord.Embed(
                            description="⚠️ **Flagged Content Preview** (Download Failed)",
                            color=0xFF0000,
                        )
                        await debug_channel.send(embed=preview_embed, file=placeholder_file)
                    except Exception as placeholder_err:
                        logger.error("Failed to send debug placeholder image: %s", placeholder_err)
        except Exception as e:
            logger.error("Failed to send debug log embed: %s", e, exc_info=True)


    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        """Scan incoming messages for NSFW media."""
        try:
            if message.author.bot or not message.guild:
                return

            cfg = self.get_server_config(message.guild.id)
            if not cfg.get("enabled", True):
                return

            # If cfg["channels"] is non-empty, only scan those; empty = all channels
            guild_channels = cfg.get("channels", [])
            if guild_channels and message.channel.id not in guild_channels:
                return

            member_role_ids = [r.id for r in message.author.roles]
            if any(rid in cfg.get("whitelisted_roles", []) for rid in member_role_ids):
                return

            if message.author.id in cfg.get("whitelisted_users", []):
                return

            media_list = self.extract_media_urls(message)
            if not media_list:
                return

            logger.info(
                "📎 %s posted %d media item(s) in #%s",
                message.author,
                len(media_list),
                message.channel.name,
            )

            uid = message.author.id
            try:
                async with self._scan_slot(uid):
                    async with self._guild_lock(message.guild.id):
                        from moderation.pipeline import scan_attachment
                        from utils.database import record_scan

                        best_result = None
                        scanned_files = []

                        for url, fname, size in media_list[:5]:
                            try:
                                result = await scan_attachment(url, bot_config)
                            except IOError as e:
                                # Download failed — skip silently
                                logger.warning("⏭️ Skipping undownloadable URL [%s]: %s", fname, e)
                                continue
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

                            if bot_config.debug_log_channel_id:
                                await self._send_debug_log_embed(message, result, url, fname, size)

                            cache_hit = result.reason and result.reason.startswith("[Cache HIT]")
                            await record_scan(
                                message_id=str(message.id),
                                guild_id=str(message.guild.id),
                                channel_id=str(message.channel.id),
                                user_id=str(message.author.id),
                                filename=fname,
                                verdict=result.verdict,
                                branch=result.branch,
                                model=result.model,
                                reason=result.reason,
                                processing_time_ms=result.processing_time_ms,
                                cache_hit=cache_hit,
                            )

                            scanned_files.append((url, fname, size, result))

                            if best_result is None or _SEVERITY.get(result.verdict, 0) > _SEVERITY.get(best_result.verdict, 0):
                                best_result = result

                        if best_result is None:
                            return

                        file_info = scanned_files

                        if best_result.verdict in _BLOCKING_VERDICTS:
                            try:
                                await message.delete()
                                logger.info("🗑️ Deleted message from %s", message.author)
                            except Exception as e:
                                logger.warning("Could not delete message: %s", e)

                            await self._send_dm(message.author, message.guild.name)
                            await self._send_log_embed(message, best_result, file_info, "BLOCK", cfg)
                            logger.info(
                                "🚨 BLOCKED content from %s in #%s",
                                message.author,
                                message.channel.name,
                            )

                        elif best_result.verdict in _REVIEW_VERDICTS:
                            await self._send_log_embed(message, best_result, file_info, "REVIEW", cfg)
                            logger.info(
                                "⚠️ REVIEW flagged for %s in #%s",
                                message.author,
                                message.channel.name,
                            )

                        elif best_result.verdict == "SAFE":
                            # Log safe static images only (not GIFs/videos)
                            gif_vid_exts = (".gif", ".gifv", ".mp4", ".webm", ".mov", ".avi")
                            has_gif_or_video = any(
                                any(ext in m_fname.lower() for ext in gif_vid_exts)
                                or any(ext in m_url.split("?")[0].lower() for ext in gif_vid_exts)
                                or "tenor.com" in m_url.lower()
                                or "giphy.com" in m_url.lower()
                                or m_fname.lower() == "embed_gif"
                                for m_url, m_fname, _ in media_list
                            )
                            if has_gif_or_video or any(emb.type == "gifv" for emb in message.embeds):
                                logger.info(
                                    "✅ SAFE GIF/video — skipping moderation log for %s in #%s",
                                    message.author, message.channel.name,
                                )
                            else:
                                await self._send_log_embed(message, best_result, file_info, "SAFE", cfg)
                                logger.info(
                                    "✅ SAFE image logged for %s in #%s",
                                    message.author, message.channel.name,
                                )

            except ValueError:
                pass  # Rate limit exceeded — silently drop extra scans

        except Exception as e:
            logger.error("AutoMod on_message error: %s", e, exc_info=True)

    @commands.Cog.listener()
    async def on_message_edit(self, before: discord.Message, after: discord.Message) -> None:
        """Scan edited messages for new or changed NSFW media to prevent bypasses."""
        if after.author.bot or not after.guild:
            return

        before_content = before.content or ""
        after_content = after.content or ""
        before_att = {a.id for a in before.attachments}
        after_att = {a.id for a in after.attachments}
        
        if before_content == after_content and before_att == after_att:
            return
            
        logger.debug("Message %s edited by %s. Re-running moderation.", after.id, after.author)
        await self.on_message(after)


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
