"""Record classroom audio and run DashScope ASR.

This repo's recommended path is `tools/web_viz_server.py` (record → report).
This script is a standalone utility for DashScope ASR and audio capture.

Modes:
  - file: record WAV (sounddevice) then transcribe the WAV by sending chunks to DashScope.
  - stream: stream microphone to DashScope in real time (pyaudio) while also recording WAV (sounddevice).

Examples:
  - Record then transcribe (offline):
      ./.venv/bin/python tools/record_classroom_asr.py file --out-prefix logs/demo --duration 60

  - Streaming ASR + record WAV, and push events to web server:
      ./.venv/bin/python tools/record_classroom_asr.py stream --duration 60 --push-url http://localhost:8000/push
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv


PROJ_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJ_ROOT / ".env")

if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))


def _default_dashscope_key() -> str:
    return os.getenv("DASH_SCOPE_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or ""


def list_audio_devices() -> List[Dict[str, Any]]:
    try:
        import sounddevice as sd  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"sounddevice not available: {exc}") from exc
    devs = sd.query_devices()
    out: List[Dict[str, Any]] = []
    for i, d in enumerate(devs):
        out.append(
            {
                "index": i,
                "name": d.get("name"),
                "max_input_channels": d.get("max_input_channels"),
                "default_samplerate": d.get("default_samplerate"),
            }
        )
    return out


def record_wav(path: Path, duration: float, samplerate: int = 16000, channels: int = 1, device: Optional[int] = None) -> None:
    try:
        import numpy as np  # type: ignore
        import sounddevice as sd  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"recording requires sounddevice+numpy: {exc}") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Recording {duration:.1f}s -> {path} @ {samplerate}Hz (device={device})")

    kwargs: Dict[str, Any] = {}
    if device is not None:
        kwargs["device"] = device

    frames = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels, dtype="int16", **kwargs)
    sd.wait()
    data = np.asarray(frames, dtype="int16")

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(data.tobytes())


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def transcribe_wav_with_dashscope(wav_path: Path, out_jsonl: Path, api_key: str, sample_rate: int = 16000, chunk_ms: int = 100) -> None:
    """Transcribe WAV by sending frames to DashScope Recognition."""
    try:
        import dashscope  # type: ignore
        from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"dashscope not available: {exc}") from exc

    if api_key:
        dashscope.api_key = api_key

    events: List[Dict[str, Any]] = []

    class _CB(RecognitionCallback):
        def on_open(self) -> None:
            return

        def on_close(self) -> None:
            return

        def on_event(self, result: RecognitionResult) -> None:
            try:
                txt = result.get_sentence()
            except Exception:
                txt = None
            if not txt:
                return
            events.append({"ts": time.time(), "type": "ASR_SENTENCE", "text": txt, "raw": result.__dict__})

    recog = Recognition(model="fun-asr-realtime", format="pcm", sample_rate=sample_rate, callback=_CB())
    recog.start()
    try:
        with wave.open(str(wav_path), "rb") as wf:
            frames_per_chunk = int(sample_rate * (chunk_ms / 1000.0))
            while True:
                chunk = wf.readframes(frames_per_chunk)
                if not chunk:
                    break
                recog.send_audio_frame(chunk)
                time.sleep(chunk_ms / 1000.0)
    finally:
        try:
            recog.stop()
        except Exception:
            pass

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")


class _Poster:
    """Best-effort event poster to the web viz `/push` endpoint."""

    def __init__(self, url: str):
        self.url = url

    def post_many(self, events: List[Dict[str, Any]]) -> None:
        if not self.url or not events:
            return
        try:
            import requests  # type: ignore
        except Exception:
            return
        try:
            requests.post(self.url, json=events, timeout=3)
        except Exception:
            pass


def _cmd_file(args: argparse.Namespace) -> int:
    if args.list_devices:
        try:
            devs = list_audio_devices()
        except Exception as exc:
            print(str(exc))
            return 1
        print("Available audio devices:")
        for d in devs:
            print(f"  {d['index']}: {d['name']} (inputs={d['max_input_channels']}, sr={d['default_samplerate']})")
        return 0

    out_prefix = Path(args.out_prefix)
    wav_path = out_prefix.with_suffix(".wav")
    jsonl_path = out_prefix.with_suffix(".asr.jsonl")
    api_key = args.api_key or _default_dashscope_key()
    if not api_key:
        print("Missing DashScope key. Set DASH_SCOPE_API_KEY (or DASHSCOPE_API_KEY) or pass --api-key.")
        return 2

    try:
        record_wav(wav_path, args.duration, samplerate=args.sample_rate, channels=1, device=args.device)
    except Exception as exc:
        print("Recording failed:", exc)
        return 1

    print("Recorded WAV ->", wav_path)
    print("Starting transcription (may take a while)...")
    try:
        transcribe_wav_with_dashscope(wav_path, jsonl_path, api_key, sample_rate=args.sample_rate, chunk_ms=args.chunk_ms)
    except Exception as exc:
        print("Transcription failed:", exc)
        with open(jsonl_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), "type": "ASR_SENTENCE", "text": "(transcription failed)"}, ensure_ascii=False) + "\n")
        return 1

    print("Wrote ASR JSONL ->", jsonl_path)
    return 0


def _cmd_stream(args: argparse.Namespace) -> int:
    api_key = args.api_key or _default_dashscope_key()
    if not api_key:
        print("Missing DashScope key. Set DASH_SCOPE_API_KEY (or DASHSCOPE_API_KEY) or pass --api-key.")
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "asr_events.jsonl"
    wav_path = out_dir / "classroom_audio.wav"

    try:
        from asr.asr_client import AliASRClient  # local import (may require pyaudio)
    except Exception as exc:
        print(f"AliASRClient is not available: {exc}")
        return 2

    poster = _Poster(args.push_url) if args.push_url else None

    def on_sentence(ev: Dict[str, Any]):
        _append_jsonl(jsonl_path, ev)
        if poster:
            poster.post_many([ev])
        txt = ev.get("text")
        if isinstance(txt, str) and txt.strip():
            print("ASR:", txt.strip())

    client = AliASRClient(api_key=api_key, time_base=time.time, on_sentence=on_sentence)
    print("Starting streaming ASR to DashScope…")
    client.start()

    try:
        record_wav(wav_path, args.duration, samplerate=client.sample_rate, channels=1, device=args.device)
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as exc:
        print("Recording failed:", exc)
        return 1
    finally:
        print("Stopping ASR client…")
        client.stop()
        print(f"ASR events saved to: {jsonl_path}")
        print(f"Raw audio saved to: {wav_path}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Record classroom audio and run DashScope ASR.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_file = sub.add_parser("file", help="Record WAV then transcribe the WAV by sending chunks to DashScope")
    p_file.add_argument("--out-prefix", required=True, help="Output prefix path (without extension)")
    p_file.add_argument("--duration", type=float, default=10.0)
    p_file.add_argument("--sample-rate", type=int, default=16000)
    p_file.add_argument("--chunk-ms", type=int, default=100)
    p_file.add_argument("--api-key", type=str, default=None)
    p_file.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    p_file.add_argument("--device", type=int, default=None, help="Audio input device index (see --list-devices)")
    p_file.set_defaults(func=_cmd_file)

    p_stream = sub.add_parser("stream", help="Stream mic to DashScope in real time (also records WAV)")
    p_stream.add_argument("--duration", type=float, default=60.0)
    p_stream.add_argument("--out-dir", type=str, default=str(PROJ_ROOT / "logs"))
    p_stream.add_argument("--api-key", type=str, default=None)
    p_stream.add_argument("--push-url", type=str, default=None, help="Optional: POST ASR events to web server `/push`")
    p_stream.add_argument("--device", type=int, default=None, help="Audio input device index for WAV recording")
    p_stream.set_defaults(func=_cmd_stream)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

