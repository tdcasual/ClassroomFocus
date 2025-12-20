"""
Centralized configuration helpers.

Secrets (API keys, etc.) should be stored in a local `.env` file which is
already git-ignored. Example:
    DASH_SCOPE_API_KEY=sk-...
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional
from dotenv import load_dotenv


# Load environment variables from .env if present.
load_dotenv()


def env_str(name: str, default: str = "") -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw)


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def env_int(name: str, default: int = 0, min_value: Optional[int] = None, max_value: Optional[int] = None) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        val = int(str(raw).strip())
    except Exception:
        return default
    if min_value is not None:
        val = max(min_value, val)
    if max_value is not None:
        val = min(max_value, val)
    return val


def env_float(name: str, default: float = 0.0, min_value: Optional[float] = None, max_value: Optional[float] = None) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        val = float(str(raw).strip())
    except Exception:
        return default
    if min_value is not None:
        val = max(min_value, val)
    if max_value is not None:
        val = min(max_value, val)
    return val


def get_dashscope_api_key(default: str = "") -> str:
    """Fetch DashScope API key from environment."""
    return os.getenv("DASH_SCOPE_API_KEY", default)


@dataclass
class SessionConfig:
    preview_enabled: bool
    preview_interval_sec: float
    preview_max_width: int
    preview_max_height: int
    video_write_enabled: bool
    audio_enabled: bool
    faces_write_enabled: bool
    faces_sample_sec: float
    stop_soft_timeout_sec: float
    stop_hard_timeout_sec: float
    preview_thumbs_enabled: bool
    preview_thumb_size: int
    preview_thumb_quality: int
    preview_thumb_pad: float
    preview_thumb_refresh_sec: float
    preview_thumb_refresh_frames: int


def load_session_config() -> SessionConfig:
    soft_default = env_float("WEB_STOP_TIMEOUT_SEC", 2.0, min_value=0.1)
    return SessionConfig(
        preview_enabled=env_bool("WEB_PREVIEW_ENABLED", default=True),
        preview_interval_sec=env_float("WEB_PREVIEW_INTERVAL_SEC", 0.15, min_value=0.01),
        preview_max_width=env_int("WEB_PREVIEW_MAX_WIDTH", 320, min_value=64),
        preview_max_height=env_int("WEB_PREVIEW_MAX_HEIGHT", 240, min_value=64),
        video_write_enabled=env_bool("WEB_RECORD_VIDEO", default=True),
        audio_enabled=env_bool("WEB_RECORD_AUDIO", default=True),
        faces_write_enabled=env_bool("WEB_WRITE_FACES", default=True),
        faces_sample_sec=env_float("WEB_FACES_SAMPLE_SEC", 0.0, min_value=0.0),
        stop_soft_timeout_sec=env_float("WEB_STOP_SOFT_TIMEOUT_SEC", soft_default, min_value=0.1),
        stop_hard_timeout_sec=env_float("WEB_STOP_HARD_TIMEOUT_SEC", 1.0, min_value=0.1),
        preview_thumbs_enabled=env_bool("WEB_PREVIEW_THUMBS", default=False),
        preview_thumb_size=env_int("WEB_PREVIEW_THUMB_SIZE", 64, min_value=16),
        preview_thumb_quality=env_int("WEB_PREVIEW_THUMB_QUALITY", 70, min_value=30, max_value=95),
        preview_thumb_pad=env_float("WEB_PREVIEW_THUMB_PAD", 0.18, min_value=0.0, max_value=1.0),
        preview_thumb_refresh_sec=env_float("WEB_PREVIEW_THUMB_REFRESH_SEC", 60.0, min_value=0.0),
        preview_thumb_refresh_frames=env_int("WEB_PREVIEW_THUMB_REFRESH_FRAMES", 60, min_value=0),
    )


@dataclass
class LiteConfig:
    enabled: bool
    status_interval_sec: float
    auto_start: bool
    auto_report: bool
    disable_video_write: bool
    disable_audio: bool
    write_faces: bool
    faces_sample_sec: float
    auto_stop_sec: float
    auto_upload: bool
    adaptive_scheduler: bool
    recovery_backoff_sec: float
    recovery_max_fails: int
    heartbeat_sec: float
    thumbnails: bool
    thumb_interval_sec: float
    thumb_size: int
    thumb_quality: int
    thumb_pad: float
    thumb_refresh_sec: float
    thumb_refresh_frames: int
    max_faces: int
    process_every_n: int
    target_fps: float
    input_scale: float
    disable_camera_similarity: bool
    adaptive_cpu: float
    adaptive_latency_ms: float
    adaptive_max_skip: int


@dataclass
class WebConfig:
    ui_mode: str
    event_queue_maxsize: int
    idle_throttle_sec: float
    idle_preview_interval_sec: float
    idle_stats_interval_sec: float
    lite: LiteConfig


def resolve_ui_mode(profile: Any) -> str:
    raw = env_str("WEB_UI_MODE", "").strip().lower()
    if raw in ("lite", "full"):
        return raw
    if raw in ("auto", "", None):
        return "lite" if bool(getattr(profile, "is_constrained", False)) else "full"
    return "full"


def load_web_config(profile: Any) -> WebConfig:
    ui_mode = resolve_ui_mode(profile)
    lite_enabled = ui_mode == "lite"
    lite = LiteConfig(
        enabled=lite_enabled,
        status_interval_sec=env_float("LITE_STATUS_INTERVAL_SEC", 5.0, min_value=0.5),
        auto_start=env_bool("LITE_AUTO_START", default=True),
        auto_report=env_bool("LITE_AUTO_REPORT", default=False),
        disable_video_write=env_bool("LITE_DISABLE_VIDEO_WRITE", default=True),
        disable_audio=env_bool("LITE_DISABLE_AUDIO", default=True),
        write_faces=env_bool("LITE_WRITE_FACES", default=True),
        faces_sample_sec=env_float("LITE_FACES_SAMPLE_SEC", 1.0, min_value=0.0),
        auto_stop_sec=env_float("LITE_AUTO_STOP_SEC", 0.0, min_value=0.0),
        auto_upload=env_bool("LITE_AUTO_UPLOAD", default=False),
        adaptive_scheduler=env_bool("LITE_ADAPTIVE_SCHEDULER", default=True),
        recovery_backoff_sec=env_float("LITE_RECOVERY_BACKOFF_SEC", 3.0, min_value=0.1),
        recovery_max_fails=env_int("LITE_RECOVERY_MAX_FAILS", 3, min_value=1),
        heartbeat_sec=env_float("LITE_HEARTBEAT_SEC", 60.0, min_value=0.0),
        thumbnails=env_bool("LITE_THUMBNAILS", default=True),
        thumb_interval_sec=env_float("LITE_THUMB_INTERVAL_SEC", 1.2, min_value=0.05),
        thumb_size=env_int("LITE_THUMB_SIZE", 64, min_value=16),
        thumb_quality=env_int("LITE_THUMB_QUALITY", 70, min_value=30, max_value=95),
        thumb_pad=env_float("LITE_THUMB_PAD", 0.18, min_value=0.0, max_value=1.0),
        thumb_refresh_sec=env_float("LITE_THUMB_REFRESH_SEC", 60.0, min_value=0.0),
        thumb_refresh_frames=env_int("LITE_THUMB_REFRESH_FRAMES", 60, min_value=0),
        max_faces=env_int("LITE_MAX_FACES", 1, min_value=1),
        process_every_n=env_int("LITE_PROCESS_EVERY_N", 3, min_value=1),
        target_fps=env_float("LITE_TARGET_FPS", 8.0, min_value=1.0),
        input_scale=env_float("LITE_INPUT_SCALE", 0.5, min_value=0.1, max_value=1.0),
        disable_camera_similarity=env_bool("LITE_DISABLE_CAMERA_SIMILARITY", default=True),
        adaptive_cpu=env_float("LITE_ADAPTIVE_CPU", 65.0, min_value=10.0, max_value=95.0),
        adaptive_latency_ms=env_float("LITE_ADAPTIVE_LATENCY_MS", 120.0, min_value=10.0),
        adaptive_max_skip=env_int("LITE_ADAPTIVE_MAX_SKIP", 6, min_value=1),
    )
    return WebConfig(
        ui_mode=ui_mode,
        event_queue_maxsize=env_int("WEB_EVENT_QUEUE_MAXSIZE", 2000, min_value=0),
        idle_throttle_sec=env_float("WEB_IDLE_THROTTLE_SEC", 15.0, min_value=0.0),
        idle_preview_interval_sec=env_float("WEB_IDLE_PREVIEW_INTERVAL_SEC", 1.5, min_value=0.0),
        idle_stats_interval_sec=env_float("WEB_IDLE_STATS_INTERVAL_SEC", 10.0, min_value=0.5),
        lite=lite,
    )
