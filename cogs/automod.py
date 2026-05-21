"""
cogs/automod.py — NSFW Moderation cog.

Listens for messages in monitored channels, runs the local AI pipeline,
and takes action based on the BLOCK / REVIEW / SAFE verdict.

Preserves all existing per-guild config (punishment, threshold, whitelist, log_channel)
from automod_config.json.
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

logger = logging.getLogger(__name__)


# ── Verdict severity helper ────────────────────────────────────────────────────
_SEVERITY = {"SAFE": 0, "REVIEW": 1, "BLOCK": 2}


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
                "punishment": "timeout",
                "timeout_duration": 10,
                "ban_duration": None,
                "nsfw_threshold": 50,
                "log_channel": None,
                "whitelisted_roles": [],
                "whitelisted_users": [],
            }
            self.save_config()

        defaults = {
            "enabled": True,
            "punishment": "timeout",
            "timeout_duration": 10,
            "ban_duration": None,
            "nsfw_threshold": 50,
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
        try:
            from database import add_modlog
            await add_modlog(user.id, action, reason, self.bot.user.id)
        except Exception as e:
            logger.warning("DB log failed: %s", e)

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
        """Send a detailed log embed to the configured log channel."""
        log_channel_id = cfg.get("log_channel")
        if not log_channel_id:
            return

        log_channel = message.guild.get_channel(int(log_channel_id))
        if not log_channel:
            return

        member = message.author
        now = discord.utils.utcnow()
        color = 0xFF4444 if verdict == "BLOCK" else 0xFFA500

        title = (
            "🚨 NSFW Content Detected — BLOCKED"
            if verdict == "BLOCK"
            else "⚠️ [REVIEW NEEDED] Possible NSFW Content"
        )

        embed = discord.Embed(title=title, color=color, timestamp=now)
        embed.set_thumbnail(url=member.display_avatar.url)

        embed.add_field(name="User", value=f"{member.mention} (`{member.id}`)", inline=True)
        embed.add_field(name="Channel", value=message.channel.mention, inline=True)
        embed.add_field(name="Message ID", value=f"`{message.id}`", inline=True)

        acct_ts = int(member.created_at.timestamp())
        embed.add_field(
            name="👤 Account",
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

        # AI detection info
        embed.add_field(
            name="🤖 AI Detection",
            value=(
                f"**Verdict:** `{scan_result.verdict}`\n"
                f"**Reason:** {scan_result.reason}\n"
                f"**Branch:** {scan_result.branch}\n"
                f"**Model:** {scan_result.model}\n"
                f"**Time:** {scan_result.processing_time_ms:.0f}ms"
                + (f"\n**Frame:** #{scan_result.frame_index}" if scan_result.frame_index is not None else "")
            ),
            inline=False,
        )

        if message.content:
            embed.add_field(
                name="💬 Message",
                value=message.content[:512],
                inline=False,
            )

        if file_info:
            lines = []
            for url, fname, size in file_info[:5]:
                size_str = f" `({size:,} bytes)`" if size else ""
                lines.append(f"[{fname}]({url}){size_str}")
            embed.add_field(name="📎 Attachments", value="\n".join(lines), inline=False)

        embed.set_footer(text="NSFW Detection System • Local AI Pipeline")

        view = None
        if verdict == "BLOCK" and cfg.get("punishment") == "timeout":
            view = RemoveTimeoutView(member.id)

        log_msg = await log_channel.send(embed=embed, view=view)

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

            # Check channel monitoring
            from config import config as bot_config

            # Use the bot_config monitored_channels if set, otherwise use per-guild config
            if not bot_config.monitor_all and bot_config.monitored_channels:
                if message.channel.id not in bot_config.monitored_channels:
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

                    if best_result is None or _SEVERITY.get(result.verdict, 0) > _SEVERITY.get(best_result.verdict, 0):
                        best_result = result
                        best_file_info = [(url, fname, size)]

                    if result.verdict == "BLOCK":
                        break  # Stop processing on first block

                if best_result is None or best_result.verdict == "SAFE":
                    return

                # Take action based on verdict
                file_info = best_file_info or []

                if best_result.verdict == "BLOCK":
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

                elif best_result.verdict == "REVIEW":
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

            log_channel = None
            if cfg["log_channel"]:
                log_channel = ctx.guild.get_channel(int(cfg["log_channel"]))

            em = discord.Embed(title="🤖 NSFW Scanner (Local AI)", color=discord.Color.blue())
            em.add_field(name="Status", value=status, inline=True)
            em.add_field(name="Threshold", value=f"{cfg['nsfw_threshold']}%", inline=True)
            em.add_field(name="Punishment", value=cfg["punishment"].title(), inline=True)
            em.add_field(name="Engine", value="Local AI Pipeline (4-model council)", inline=True)
            em.add_field(name="Log Channel", value=log_channel.mention if log_channel else "Not set", inline=True)
            em.set_footer(text="Use /scanner commands to configure | /nsfw for new slash commands")
            await ctx.send(embed=em)

    @scanner.command(description="Enable/disable scanner")
    @commands.has_permissions(manage_messages=True)
    async def toggle(self, ctx: commands.Context) -> None:
        cfg = self.get_server_config(ctx.guild.id)
        cfg["enabled"] = not cfg["enabled"]
        self.save_config()
        await ctx.send(f"✅ Scanner {'enabled' if cfg['enabled'] else 'disabled'}")

    @scanner.command(description="Set detection threshold (1-100)")
    @commands.has_permissions(manage_messages=True)
    async def threshold(self, ctx: commands.Context, percentage: int) -> None:
        if not 1 <= percentage <= 100:
            return await ctx.send("❌ Use 1-100", ephemeral=True)
        cfg = self.get_server_config(ctx.guild.id)
        cfg["nsfw_threshold"] = percentage
        self.save_config()
        await ctx.send(f"✅ Threshold: {percentage}%")

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
