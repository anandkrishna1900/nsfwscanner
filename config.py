"""
config.py — BotConfig dataclass loaded from .env

Reads all environment variables using python-dotenv.
Raises ValueError at startup if DISCORD_TOKEN is missing.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv

load_dotenv()


def _parse_int_list(raw: Optional[str]) -> List[int]:
    """Parse a comma-separated string of integers into a list."""
    if not raw or raw.strip().lower() in ("", "all"):
        return []
    return [int(x.strip()) for x in raw.split(",") if x.strip().isdigit()]


@dataclass
class BotConfig:
    # ── Discord ──────────────────────────────────────────────────────────────
    discord_token: str
    prefix: str
    guild_ids: List[int]

    # ── Channel Config ────────────────────────────────────────────────────────
    monitored_channels: List[int]          # empty list → monitor all channels
    monitor_all: bool                      # True if MONITORED_CHANNELS="all"
    log_channel_id: Optional[int]
    admin_role_id: Optional[int]
    debug_log_channel_id: Optional[int]

    # ── AI Model Config ───────────────────────────────────────────────────────
    model_cache_dir: str
    device: str                            # "cuda" or "cpu"

    # ── Video/GIF Limits ─────────────────────────────────────────────────────
    max_video_size_mb: int
    max_video_duration_secs: int
    review_threshold_offset: float

    # ── Database ──────────────────────────────────────────────────────────────
    sqlite_db_path: str                    # path to the SQLite .db file

    # ── Sensitivity ───────────────────────────────────────────────────────────
    sensitivity: Dict[str, Any] = field(default_factory=dict)

    def get_threshold(self, base_value: float) -> float:
        """Override base_value with global_threshold if it is > 0."""
        global_t = self.sensitivity.get("global_threshold", 0)
        if global_t > 0:
            return float(global_t)
        return float(base_value)

    @classmethod
    def from_env(cls) -> "BotConfig":
        """Load and validate config from environment variables."""
        # Support both DISCORD_TOKEN and legacy TOKEN
        token = os.getenv("DISCORD_TOKEN") or os.getenv("TOKEN")
        if not token or not token.strip():
            raise ValueError(
                "DISCORD_TOKEN (or TOKEN) is missing from the .env file. "
                "Please set it before starting the bot."
            )

        monitored_raw = os.getenv("MONITORED_CHANNELS", "").strip()
        monitor_all = monitored_raw.lower() == "all"
        monitored_channels = [] if monitor_all else _parse_int_list(monitored_raw)

        log_channel_raw = os.getenv("LOG_CHANNEL_ID", "").strip()
        log_channel_id = int(log_channel_raw) if log_channel_raw.isdigit() else None

        admin_role_raw = os.getenv("ADMIN_ROLE_ID", "").strip()
        admin_role_id = int(admin_role_raw) if admin_role_raw.isdigit() else None

        debug_log_channel_raw = os.getenv("DEBUG_LOG_CHANNEL_ID", "").strip()
        debug_log_channel_id = int(debug_log_channel_raw) if debug_log_channel_raw.isdigit() else None

        # Load sensitivity config
        sensitivity_config = {}
        try:
            with open("sensitivity.json", "r") as f:
                sensitivity_config = json.load(f)
        except Exception as e:
            pass # Fallback to defaults if missing or invalid

        return cls(
            discord_token=token.strip(),
            prefix=os.getenv("PREFIX", ";"),
            guild_ids=_parse_int_list(os.getenv("GUILD_IDS", "")),
            monitored_channels=monitored_channels,
            monitor_all=monitor_all,
            log_channel_id=log_channel_id,
            admin_role_id=admin_role_id,
            debug_log_channel_id=debug_log_channel_id,
            model_cache_dir=os.getenv("MODEL_CACHE_DIR", "./models"),
            device=os.getenv("DEVICE", "cuda").lower(),
            max_video_size_mb=int(os.getenv("MAX_VIDEO_SIZE_MB", "50")),
            max_video_duration_secs=int(os.getenv("MAX_VIDEO_DURATION_SECS", "300")),
            review_threshold_offset=float(os.getenv("REVIEW_THRESHOLD_OFFSET", "0.15")),
            sqlite_db_path=os.getenv("SQLITE_DB_PATH", "./bot.db"),
            sensitivity=sensitivity_config,
        )

    def is_monitored(self, channel_id: int) -> bool:
        """Return True if this channel should be scanned."""
        if self.monitor_all:
            return True
        return channel_id in self.monitored_channels

    def add_monitored_channel(self, channel_id: int) -> None:
        """Add a channel to the monitored list."""
        if channel_id not in self.monitored_channels:
            self.monitored_channels.append(channel_id)
            self.monitor_all = False

    def remove_monitored_channel(self, channel_id: int) -> None:
        """Remove a channel from the monitored list."""
        if channel_id in self.monitored_channels:
            self.monitored_channels.remove(channel_id)


# Singleton instance — import this everywhere
config: BotConfig = BotConfig.from_env()
