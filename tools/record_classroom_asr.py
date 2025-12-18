"""Record classroom audio to WAV and run DashScope ASR on the recorded file.

Produces:
  - {out_prefix}.wav
  - {out_prefix}.asr.jsonl

Notes:
  - Requires `sounddevice` (for recording) and `dashscope` for transcription.
  - If `dashscope` is not available, the script will still record WAV but will write a placeholder JSONL.
"""
import time
import wave
import json
from pathlib import Path
from typing import List, Dict, Any, Optional


def list_audio_devices():
    try:
        import sounddevice as sd
    except Exception as e:
        print("sounddevice not available:", e)
        return []
    devs = sd.query_devices()
    out = []
    for i, d in enumerate(devs):
        out.append({"index": i, "name": d.get('name'), "max_input_channels": d.get('max_input_channels'), "default_samplerate": d.get('default_samplerate')})
    return out


def record_wav(path: str, duration: float, samplerate: int = 16000, channels: int = 1, device: Optional[int] = None):
    try:
        import sounddevice as sd
        import numpy as np
    except Exception as e:
        raise RuntimeError("sounddevice and numpy are required to record audio: %s" % e)

    print(f"Recording {duration:.1f}s -> {path} @ {samplerate}Hz (device={device})")
    kwargs = {}
    if device is not None:
        kwargs['device'] = device
    frames = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels, dtype='int16', **kwargs)
    sd.wait()
    data = frames.astype('int16')
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(data.tobytes())


def transcribe_wav_with_dashscope(wav_path: str, out_jsonl: str, api_key: Optional[str], sample_rate: int = 16000, chunk_ms: int = 100):
    # Prefer to use asr_client.transcribe_file when available (no pyaudio required)
    try:
        from asr.asr_client import transcribe_file
    except Exception:
        transcribe_file = None

    if transcribe_file is not None:
        try:
            transcribe_file(api_key, wav_path, lambda: time.time(), lambda ev: append_jsonl(Path(out_jsonl), ev), sample_rate, chunk_ms)
            return
        except Exception as e:
            print('transcribe_file failed, falling back to direct dashscope:', e)

    try:
        import dashscope
        from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult
    except Exception:
        print("dashscope not available; writing placeholder JSONL")
        # write placeholder event
        with open(out_jsonl, 'w', encoding='utf-8') as f:
            f.write(json.dumps({"ts": time.time(), "text": "(no dashscope)"}, ensure_ascii=False) + "\n")
        return

    if api_key:
        dashscope.api_key = api_key

    events: List[Dict[str, Any]] = []

    class _CB(RecognitionCallback):
        def __init__(self, time_base_fn):
            super().__init__()
            self.time_base_fn = time_base_fn

        def on_open(self) -> None:
            pass

        def on_close(self) -> None:
            pass

        def on_event(self, result: RecognitionResult) -> None:
            try:
                txt = result.get_sentence()
            except Exception:
                txt = None
            if not txt:
                return
            ts = float(self.time_base_fn())
            events.append({"ts": ts, "text": txt, "raw": result.__dict__})

    recog = Recognition(model="fun-asr-realtime", format="pcm", sample_rate=sample_rate, callback=_CB(lambda: time.time()))
    recog.start()
    try:
        # read wav in chunks and send
        with wave.open(wav_path, 'rb') as wf:
            bytes_per_frame = wf.getsampwidth() * wf.getnchannels()
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

    # write events normalized relative to recording start (approx)
    with open(out_jsonl, 'w', encoding='utf-8') as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=float, default=10.0)
    parser.add_argument('--out-prefix', required=True)
    parser.add_argument('--sample-rate', type=int, default=16000)
    parser.add_argument('--api-key', type=str, default=None)
    parser.add_argument('--list-devices', action='store_true', help='List audio devices and exit')
    parser.add_argument('--device', type=int, default=None, help='Device index to use for recording (from --list-devices)')
    args = parser.parse_args()

    if args.list_devices:
        devs = list_audio_devices()
        if not devs:
            print('No audio devices found or sounddevice not installed')
            return
        print('Available audio devices:')
        for d in devs:
            print(f"  {d['index']}: {d['name']} (inputs={d['max_input_channels']}, sr={d['default_samplerate']})")
        return

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    wav_path = str(out_prefix) + '.wav'
    jsonl_path = str(out_prefix) + '.asr.jsonl'

    start = time.time()
    try:
        record_wav(wav_path, args.duration, samplerate=args.sample_rate, channels=1, device=args.device)
    except Exception as e:
        print('Recording failed:', e)
        return
    record_end = time.time()

    print('Recorded WAV ->', wav_path)
    print('Starting transcription (may take a few seconds)')
    try:
        transcribe_wav_with_dashscope(wav_path, jsonl_path, args.api_key, sample_rate=args.sample_rate)
    except Exception as e:
        print('Transcription failed:', e)
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps({"ts": record_end, "text": "(transcription failed)"}, ensure_ascii=False) + "\n")

    print('Wrote ASR JSONL ->', jsonl_path)


if __name__ == '__main__':
    main()
"""Record a classroom session: save ASR events (JSONL) and raw audio (WAV).

Usage:
  - Ensure `.env` contains DASHSCOPE_API_KEY (or pass key as first arg).
  - Run:
      .venv\\Scripts\\python.exe tools\\record_classroom_asr.py [api_key] [duration_seconds]

Notes:
  - This script starts `AliASRClient` to stream audio to DashScope and
    writes ASR sentence events to `logs/asr_events.jsonl`.
  - Simultaneously it records raw audio locally using `sounddevice` to
    `logs/classroom_audio.wav` for the same duration. This allows offline
    playback and alignment with ASR events.
  - There can be platform-dependent contention when two audio libraries
    access the same device. If you see errors, try closing other apps
    that use the microphone.
"""
import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional
import threading
import queue
import requests

import sounddevice as sd
import numpy as np
import wave
from dotenv import load_dotenv

# Ensure project root on sys.path
proj_root = Path(__file__).resolve().parents[1]
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))

from asr.asr_client import AliASRClient


def ensure_logs_dir():
    p = proj_root / 'logs'
    p.mkdir(exist_ok=True)
    return p


def append_jsonl(path: Path, obj: Dict[str, Any]):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')


class _Poster:
    """Background poster that batches JSON events and sends them to a HTTP endpoint."""
    def __init__(self, url: Optional[str], batch_interval: float = 0.1, max_batch: int = 200):
        self.url = url
        self._q = queue.Queue()
        self._thr = None
        self._stop = threading.Event()
        self.batch_interval = float(batch_interval)
        self.max_batch = int(max_batch)

    def start(self):
        if not self.url:
            return
        if self._thr and self._thr.is_alive():
            return
        self._thr = threading.Thread(target=self._run, daemon=True)
        self._thr.start()

    def send(self, obj: Dict[str, Any]):
        if not self.url:
            return
        try:
            self._q.put_nowait(obj)
        except Exception:
            pass

    def _run(self):
        while not self._stop.is_set():
            batch = []
            try:
                ev = self._q.get(timeout=self.batch_interval)
                batch.append(ev)
            except queue.Empty:
                # timeout, nothing to send
                continue

            # drain additional items up to max_batch without blocking
            while len(batch) < self.max_batch:
                try:
                    ev = self._q.get_nowait()
                    batch.append(ev)
                except queue.Empty:
                    break

            # send as a single batch payload
            payload = {"type": "batch", "events": batch}
            try:
                requests.post(self.url, json=payload, timeout=3)
            except Exception:
                # best-effort; ignore failures
                pass

    def stop(self):
        self._stop.set()
        if self._thr:
            self._thr.join(timeout=1.0)


def record_audio_to_wav(filename: Path, duration: float, samplerate: int = 16000, channels: int = 1):
    print(f'Recording raw audio to {filename} for {duration:.1f}s...')
    frames = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels, dtype='int16')
    sd.wait()
    # frames shape: (N, channels)
    frames = np.atleast_2d(frames)
    with wave.open(str(filename), 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # int16
        wf.setframerate(samplerate)
        wf.writeframes(frames.tobytes())
    print('Audio saved.')


def main():
    load_dotenv(proj_root / '.env')
    api_key = None
    dur = 60.0
    push_url = None
    if len(sys.argv) > 1 and sys.argv[1]:
        api_key = sys.argv[1]
    else:
        api_key = os.getenv('DASHSCOPE_API_KEY') or os.getenv('DASH_SCOPE_API_KEY')
    if len(sys.argv) > 2:
        try:
            dur = float(sys.argv[2])
        except Exception:
            pass
    # optional push url (e.g. http://localhost:8000/push)
    if len(sys.argv) > 3 and sys.argv[3]:
        push_url = sys.argv[3]

    logs = ensure_logs_dir()
    jsonl_path = logs / 'asr_events.jsonl'
    wav_path = logs / 'classroom_audio.wav'

    def on_sentence(ev: Dict[str, Any]):
        # Write event and print concise text
        append_jsonl(jsonl_path, ev)
        # optionally push to websocket server via HTTP endpoint
        try:
            if poster is not None:
                poster.send(ev)
        except Exception:
            pass
        # extract text similar to test script
        txt = ''
        t = ev.get('text') if isinstance(ev, dict) else None
        if isinstance(t, str):
            txt = t
        elif isinstance(t, dict):
            txt = t.get('text') or ''.join([w.get('text', '') for w in (t.get('words') or [])])
        if txt and txt.strip():
            print('ASR:', txt)

    client = AliASRClient(api_key=api_key, time_base=time.time, on_sentence=on_sentence)
    print('Starting ASR client (streaming to DashScope)...')
    client.start()

    poster = None
    if push_url:
        poster = _Poster(push_url)
        poster.start()

    try:
        # Record raw audio in foreground (blocking) while ASR runs in background
        record_audio_to_wav(wav_path, duration=dur, samplerate=client.sample_rate, channels=1)
    except KeyboardInterrupt:
        print('Interrupted by user')
    finally:
        print('Stopping ASR client...')
        client.stop()
        if poster is not None:
            poster.stop()
        print(f'ASR events saved to: {jsonl_path}')
        print(f'Raw audio saved to: {wav_path}')


if __name__ == '__main__':
    main()
