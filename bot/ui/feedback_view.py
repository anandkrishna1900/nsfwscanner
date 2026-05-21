"""
bot/ui/feedback_view.py — Persistent Discord UI view for moderator feedback buttons.

Attached to every moderation log embed. Only users with Manage Messages permission
OR the configured ADMIN_ROLE_ID can interact with the buttons.

Buttons disable after first click (per-message). Survives bot restarts via
persistent views (timeout=None + stable custom_id).
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import discord

logger = logging.getLogger(__name__)

# ── Permission helper ──────────────────────────────────────────────────────────

async def _check_mod_permission(interaction: discord.Interaction) -> bool:
    """
    Return True if the interacting user has permission to submit feedback.
    Sends an ephemeral error response if they don't.
    """
    from config import config as bot_config

    has_perm = interaction.user.guild_permissions.manage_messages
    if not has_perm and bot_config.admin_role_id:
        role_ids = [r.id for r in interaction.user.roles]
        has_perm = bot_config.admin_role_id in role_ids

    if not has_perm:
        await interaction.response.send_message(
            "❌ You need **Manage Messages** permission to submit feedback.",
            ephemeral=True,
        )
    return has_perm


# ── False Negative Severity Modal ──────────────────────────────────────────────

class FalseNegativeModal(discord.ui.Modal, title="False Negative — Set Severity"):
    """Modal that collects severity from a moderator marking a False Negative."""

    severity: discord.ui.TextInput = discord.ui.TextInput(
        label="Severity",
        placeholder="Enter: NSFW or EXPLICIT",
        required=True,
        max_length=10,
        style=discord.TextStyle.short,
    )

    def __init__(self, view: "ModerationFeedbackView") -> None:
        super().__init__()
        self._view = view

    async def on_submit(self, interaction: discord.Interaction) -> None:
        raw = self.severity.value.strip().upper()
        if raw not in ("NSFW", "EXPLICIT"):
            await interaction.response.send_message(
                "❌ Invalid severity. Please enter `NSFW` or `EXPLICIT`.",
                ephemeral=True,
            )
            return

        moderator_verdict = f"FALSE_NEGATIVE_{raw}"
        await self._view._handle_feedback(interaction, moderator_verdict, label=f"⚠️ FN ({raw})")


# ── Persistent Feedback View ───────────────────────────────────────────────────

class ModerationFeedbackView(discord.ui.View):
    """
    Persistent feedback view attached to moderation log embeds.

    Stores the original scan result metadata (serialized as JSON) to allow
    the button callbacks to persist feedback to the database.
    """

    def __init__(
        self,
        *,
        message_id: str = "",
        user_id: str = "",
        channel_id: str = "",
        guild_id: str = "",
        predicted_verdict: str = "",
        scan_result_json: str = "{}",
    ) -> None:
        super().__init__(timeout=None)
        self.message_id = message_id
        self.user_id = user_id
        self.channel_id = channel_id
        self.guild_id = guild_id
        self.predicted_verdict = predicted_verdict
        self.scan_result_json = scan_result_json

        # Assign stable custom_ids so buttons survive restarts
        mid = message_id or "0"
        self.btn_correct.custom_id = f"fb_correct_{mid}"
        self.btn_fp.custom_id = f"fb_fp_{mid}"
        self.btn_fn.custom_id = f"fb_fn_{mid}"

    # ── Buttons ───────────────────────────────────────────────────────────────

    @discord.ui.button(
        label="✅ Correct Detection",
        style=discord.ButtonStyle.success,
        custom_id="fb_correct_0",
    )
    async def btn_correct(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ) -> None:
        if not await _check_mod_permission(interaction):
            return
        await self._handle_feedback(interaction, "CORRECT", label="✅ Correct")

    @discord.ui.button(
        label="❌ False Positive",
        style=discord.ButtonStyle.danger,
        custom_id="fb_fp_0",
    )
    async def btn_fp(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ) -> None:
        if not await _check_mod_permission(interaction):
            return
        await self._handle_feedback(interaction, "FALSE_POSITIVE", label="❌ False Positive")

    @discord.ui.button(
        label="⚠️ False Negative",
        style=discord.ButtonStyle.secondary,
        custom_id="fb_fn_0",
    )
    async def btn_fn(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ) -> None:
        if not await _check_mod_permission(interaction):
            return
        modal = FalseNegativeModal(view=self)
        await interaction.response.send_modal(modal)

    # ── Shared handler ────────────────────────────────────────────────────────

    async def _handle_feedback(
        self,
        interaction: discord.Interaction,
        moderator_verdict: str,
        label: str,
    ) -> None:
        """Persist feedback, disable buttons, and update embed footer."""
        from utils.database import has_feedback

        # Check duplicate submission
        already = await has_feedback(self.message_id, str(interaction.user.id))
        if already:
            await interaction.response.send_message(
                "ℹ️ You have already submitted feedback on this moderation event.",
                ephemeral=True,
            )
            return

        # Persist to database
        try:
            await self._store_feedback(
                moderator_id=str(interaction.user.id),
                moderator_verdict=moderator_verdict,
            )
        except Exception as e:
            logger.error("[FeedbackView] DB error storing feedback: %s", e, exc_info=True)
            await interaction.response.send_message(
                f"❌ Failed to save feedback: {e}", ephemeral=True
            )
            return

        # Disable all buttons
        for item in self.children:
            if isinstance(item, discord.ui.Button):
                item.disabled = True

        # Update embed footer to reflect feedback
        footer_map = {
            "CORRECT": f"✅ Verified correct by {interaction.user.display_name}",
            "FALSE_POSITIVE": f"❌ Marked False Positive by {interaction.user.display_name}",
            "FALSE_NEGATIVE_NSFW": f"⚠️ False Negative (NSFW) marked by {interaction.user.display_name}",
            "FALSE_NEGATIVE_EXPLICIT": f"⚠️ False Negative (EXPLICIT) marked by {interaction.user.display_name}",
        }
        footer_text = footer_map.get(moderator_verdict, f"Feedback submitted by {interaction.user.display_name}")

        try:
            embed = interaction.message.embeds[0]
            embed.set_footer(text=footer_text)
            # For false positives, dim the embed color
            if moderator_verdict == "FALSE_POSITIVE":
                embed.color = discord.Color.from_rgb(128, 128, 128)
            elif moderator_verdict == "CORRECT":
                embed.color = discord.Color.from_rgb(34, 197, 94)  # green
            await interaction.response.edit_message(embed=embed, view=self)
        except discord.NotFound:
            if not interaction.response.is_done():
                await interaction.response.send_message(
                    "✅ Feedback recorded, but the log embed could not be updated.",
                    ephemeral=True,
                )
        except Exception as e:
            logger.error("[FeedbackView] Failed to edit embed: %s", e, exc_info=True)
            if not interaction.response.is_done():
                await interaction.response.send_message(
                    "✅ Feedback recorded.", ephemeral=True
                )

        logger.info(
            "[FeedbackView] Moderator %s (%s) submitted %s for message %s",
            interaction.user.name,
            interaction.user.id,
            moderator_verdict,
            self.message_id,
        )

    async def _store_feedback(self, moderator_id: str, moderator_verdict: str) -> None:
        """Reconstruct a minimal ScanResult-like object and store feedback."""
        from utils.database import record_feedback
        import json as _json

        try:
            scan_data = _json.loads(self.scan_result_json)
        except Exception:
            scan_data = {}

        await record_feedback(
            message_id=self.message_id,
            guild_id=self.guild_id,
            channel_id=self.channel_id,
            user_id=self.user_id,
            moderator_id=moderator_id,
            content_type=scan_data.get("branch"),
            predicted_verdict=self.predicted_verdict,
            moderator_verdict=moderator_verdict,
            model_scores=scan_data.get("model_scores", {}),
            detected_tags=scan_data.get("detected_tags", []),
            branch=scan_data.get("branch"),
            model=scan_data.get("model"),
            processing_time_ms=scan_data.get("processing_time_ms"),
        )
