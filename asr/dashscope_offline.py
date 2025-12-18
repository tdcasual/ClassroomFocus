from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class DashScopeOfflineConfig:
    api_key: str
    model: str = "fun-asr-realtime"
    sample_rate: int = 16000
    chunk_ms: int = 100


def _extract_ms(raw: Dict[str, Any], keys: List[str]) -> Optional[Tuple[float, float]]:
    pairs = [
        ("begin_time", "end_time"),
        ("begin", "end"),
        ("start_time", "end_time"),
        ("start", "end"),
        ("bg", "ed"),
    ]
    # keep backward compatibility for callers passing legacy keys list
    if keys:
        for k in keys:
            kk = str(k)
            if kk.startswith("begin"):
                pairs.insert(0, (kk, kk.replace("begin", "end", 1)))

    for s_key, e_key in pairs:
        if s_key not in raw or e_key not in raw:
            continue
        try:
            bg = float(raw[s_key])
            ed = float(raw[e_key])
        except Exception:
            continue
        if ed <= bg:
            continue
        return bg, ed
    return None


def transcribe_wav_to_segments(wav_path: str, cfg: DashScopeOfflineConfig) -> List[Dict[str, Any]]:
    """Offline transcription using DashScope Recognition on a WAV file.

    Notes:
    - DashScope's streaming Recognition callback does not always expose per-sentence
      timestamps. We use best-effort extraction from the raw payload when present.
    - If timestamps are not available, we approximate segment boundaries by audio
      progress when the sentence arrives.
    """
    from asr.asr_client import transcribe_file  # local import (dashscope optional)

    progress_sec = 0.0

    def time_base() -> float:
        return float(progress_sec)

    segments: List[Dict[str, Any]] = []
    last_end = 0.0

    def on_sentence(ev: Dict[str, Any]):
        nonlocal last_end, segments
        raw = ev.get("raw") if isinstance(ev, dict) else None
        if not isinstance(raw, dict):
            raw = {}
        txt = str(ev.get("text", "")).strip() if isinstance(ev, dict) else ""
        if not txt:
            return

        # Try extract begin/end in ms from raw payload.
        ms = _extract_ms(raw, keys=["begin_time", "begin", "start_time", "start"])
        if ms is not None:
            s = float(ms[0] / 1000.0)
            e = float(ms[1] / 1000.0)
        else:
            # Approx: end at current progress, start at last_end.
            e = max(float(time_base()), last_end + 0.2)
            s = last_end

        if e <= s:
            e = s + 0.2
        # keep monotonic
        if s < last_end:
            s = last_end
        if e < s:
            e = s + 0.2
        last_end = e
        segments.append({"start": s, "end": e, "text": txt, "raw": raw})

    def progress_hook(sec: float):
        nonlocal progress_sec
        try:
            progress_sec = float(sec)
        except Exception:
            pass

    transcribe_file(
        api_key=cfg.api_key,
        wav_path=wav_path,
        time_base=time_base,
        on_sentence=on_sentence,
        model=cfg.model,
        sample_rate=int(cfg.sample_rate),
        chunk_ms=int(cfg.chunk_ms),
        progress_hook=progress_hook,
    )

    segments.sort(key=lambda s: float(s.get("start", 0.0)))
    return segments


def config_from_env() -> Optional[DashScopeOfflineConfig]:
    key = os.getenv("DASH_SCOPE_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or ""
    if not key:
        return None
    model = os.getenv("ALI_ASR_MODEL") or os.getenv("DASHSCOPE_ASR_MODEL") or "fun-asr-realtime"
    try:
        sr = int(float(os.getenv("ALI_ASR_SAMPLE_RATE", "16000")))
    except Exception:
        sr = 16000
    try:
        chunk_ms = int(float(os.getenv("ALI_ASR_CHUNK_MS", "100")))
    except Exception:
        chunk_ms = 100
    return DashScopeOfflineConfig(api_key=key, model=model, sample_rate=sr, chunk_ms=chunk_ms)
