from typing import Optional


def effective_refresh_sec(
    base_refresh_sec: Optional[float],
    preview_interval_sec: Optional[float],
    refresh_frames: Optional[int],
) -> float:
    """Compute refresh interval linked to preview cadence."""
    try:
        refresh = float(base_refresh_sec or 0.0)
    except Exception:
        refresh = 0.0
    try:
        interval = float(preview_interval_sec or 0.0)
    except Exception:
        interval = 0.0
    try:
        frames = int(refresh_frames or 0)
    except Exception:
        frames = 0
    if frames > 0 and interval > 0:
        refresh = max(refresh, interval * frames)
    return refresh
