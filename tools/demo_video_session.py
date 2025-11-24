"""End-to-end demo script that runs CV + ASR + labeling on a prerecorded video.

Example usage:
  python tools/demo_video_session.py --video samples/students.mp4 --out-prefix out/demo_session

Outputs:
  - {out-prefix}.faces.jsonl         : per-frame face track snapshots
  - {out-prefix}.cv_events.jsonl      : face analyzer event stream (drowsy/blink/down)
  - {out-prefix}.asr.jsonl            : ASR sentence segments
  - {out-prefix}.labeled.jsonl        : teacher/student labeled segments
  - {out-prefix}.timeline.png         : visualization of labeled segments (unless --no-timeline)
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import requests

# Ensure project root on sys.path so direct execution works even via python path/to/script
PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from analysis.teacher_labeler import label_asr_segments, load_face_tracks_jsonl
from asr.asr_client import transcribe_file
from cv.face_analyzer import FaceAnalyzer, FaceAnalyzerConfig
from viz.visualizer import plot_timeline, save_labeled_jsonl


def append_jsonl(handle, obj: Dict) -> None:
    handle.write(json.dumps(obj, ensure_ascii=False) + "\n")


def extract_audio(input_video: Path, wav_path: Path, sample_rate: int, ffmpeg_bin: str) -> None:
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(input_video),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(wav_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        raise RuntimeError("ffmpeg executable not found; install ffmpeg or pass --ffmpeg")
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg failed: {exc.stderr.decode('utf-8', errors='ignore')[:200]}") from exc


def _parse_text_field(txt_obj) -> str:
    if isinstance(txt_obj, str):
        return txt_obj.strip()
    if isinstance(txt_obj, dict):
        if txt_obj.get("text"):
            return str(txt_obj.get("text")).strip()
        words = txt_obj.get("words")
        if isinstance(words, list):
            return "".join(str(w.get("text", "")) for w in words).strip()
    if isinstance(txt_obj, list):
        return "".join(_parse_text_field(t) for t in txt_obj).strip()
    return ""


def _derive_segment_times(event: Dict, next_event: Optional[Dict]) -> (float, float):
    start = float(event.get("ts", 0.0))
    raw = event.get("raw") or {}
    # Try to pull precise times from raw payloads (DashScope often returns ms-based timestamps)
    candidates = []
    if isinstance(raw, dict):
        sentence = raw.get("sentence") or raw.get("result") or {}
        if isinstance(sentence, dict):
            for key in ("begin_time", "start_time", "begin_seconds", "start_timestamp"):
                val = sentence.get(key)
                if val is not None:
                    candidates.append(float(val) / (1000.0 if abs(float(val)) > 10 else 1.0))
            for key in ("end_time", "end_seconds", "end_timestamp"):
                val = sentence.get(key)
                if val is not None:
                    end_val = float(val) / (1000.0 if abs(float(val)) > 10 else 1.0)
                    return candidates[0] if candidates else start, max(end_val, start + 0.5)
    if candidates:
        start = candidates[0]
    if next_event and next_event.get("ts") is not None:
        end = float(next_event["ts"])
        if end <= start:
            end = start + 1.5
        return start, end
    return start, start + 2.0


def build_asr_segments(events: List[Dict]) -> List[Dict]:
    if not events:
        return []
    events = sorted(events, key=lambda e: e.get("ts", 0.0))
    segments: List[Dict] = []
    for idx, ev in enumerate(events):
        next_ev = events[idx + 1] if idx + 1 < len(events) else None
        start, end = _derive_segment_times(ev, next_ev)
        text = _parse_text_field(ev.get("text"))
        segments.append({
            "start": float(start),
            "end": float(end),
            "text": text,
            "raw": ev.get("raw"),
        })
    return segments


def post_segments(push_url: str, segments: List[Dict]) -> None:
    payload = []
    for idx, seg in enumerate(segments):
        payload.append({
            "type": "asr_segment",
            "id": f"seg{idx}",
            "start": seg.get("start"),
            "end": seg.get("end"),
            "label": seg.get("label", "unknown"),
            "text": seg.get("text", ""),
        })
    if not payload:
        return
    try:
        requests.post(push_url, json={"type": "batch", "events": payload}, timeout=4)
    except Exception as exc:
        print(f"Warning: failed to push segments to {push_url}: {exc}")


def main():
    ap = argparse.ArgumentParser(description="Run full pipeline on a prerecorded video")
    ap.add_argument("--video", required=True, help="Input video file")
    ap.add_argument("--out-prefix", required=True, help="Output prefix (without extension)")
    ap.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg executable path")
    ap.add_argument("--sample-rate", type=int, default=16000, help="Audio sample rate for ASR")
    ap.add_argument("--chunk-ms", type=int, default=100, help="Chunk duration sent to ASR")
    ap.add_argument("--asr-stability-secs", type=float, default=0.8, help="Seconds of unchanged hypothesis to consider stable")
    ap.add_argument("--asr-bucket-width", type=float, default=0.5, help="Bucket width (s) to group nearby ASR events for stability")
    ap.add_argument("--api-key", type=str, default=None, help="DashScope API key (overrides env)")
    ap.add_argument("--no-timeline", action="store_true", help="Skip timeline PNG output")
    ap.add_argument("--show", action="store_true", help="Show OpenCV debug window while processing")
    # Tunable thresholds for quick experiments
    ap.add_argument("--ear-ratio", type=float, default=None, help="EAR threshold ratio (ear_ratio * open_baseline)")
    ap.add_argument("--ear-min", type=float, default=None, help="Minimum EAR threshold")
    ap.add_argument("--ear-ema-alpha", type=float, default=None, help="EAR EMA alpha (responsiveness)")
    ap.add_argument("--drowsy-secs", type=float, default=None, help="Seconds eyes must remain closed to mark drowsy")
    ap.add_argument("--pitch-down-deg", type=float, default=None, help="Pitch threshold (deg) for looking-down detection")
    ap.add_argument("--pitch-ema-alpha", type=float, default=None, help="Pitch EMA alpha (responsiveness)")
    ap.add_argument("--down-secs", type=float, default=None, help="Seconds head-down must persist to mark looking-down")
    ap.add_argument("--font-path", type=str, default=None, help="Optional font path for timeline rendering")
    ap.add_argument("--push-url", type=str, default=None, help="Optional HTTP endpoint to push labeled segments")
    ap.add_argument("--max-frames", type=int, default=None, help="Optional limit when debugging")
    ap.add_argument("--skip-asr", action="store_true", help="Skip ASR (keeps previous asr.jsonl if present)")
    args = ap.parse_args()

    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    out_prefix = Path(args.out_prefix).expanduser().resolve()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    faces_path = out_prefix.with_suffix(".faces.jsonl")
    cv_events_path = out_prefix.with_suffix(".cv_events.jsonl")
    asr_path = out_prefix.with_suffix(".asr.jsonl")
    labeled_path = out_prefix.with_suffix(".labeled.jsonl")
    timeline_path = out_prefix.with_suffix(".timeline.png")

    # 1. Extract audio to WAV for ASR
    wav_path = out_prefix.with_suffix(".wav")
    # Only extract audio when ASR is requested. If skipping ASR and a WAV
    # already exists we won't touch it; this avoids requiring ffmpeg when
    # the user only wants CV processing.
    if not args.skip_asr and not wav_path.exists():
        print(f"[demo] Extracting audio -> {wav_path}")
        extract_audio(video_path, wav_path, args.sample_rate, args.ffmpeg)

    # 2. Run face analyzer over video frames
    print("[demo] Running face analyzer")
    cfg = FaceAnalyzerConfig(debug_draw=args.show)
    # apply runtime overrides if provided
    if args.ear_ratio is not None:
        cfg.ear_ratio = float(args.ear_ratio)
    if args.ear_min is not None:
        cfg.ear_min = float(args.ear_min)
    if args.ear_ema_alpha is not None:
        cfg.ear_ema_alpha = float(args.ear_ema_alpha)
    if args.drowsy_secs is not None:
        cfg.drowsy_secs = float(args.drowsy_secs)
    if args.pitch_down_deg is not None:
        cfg.pitch_down_deg = float(args.pitch_down_deg)
    if args.pitch_ema_alpha is not None:
        cfg.pitch_ema_alpha = float(args.pitch_ema_alpha)
    if args.down_secs is not None:
        cfg.down_secs = float(args.down_secs)
    analyzer = FaceAnalyzer(cfg)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3:
        fps = 25.0
    frame_time = 1.0 / max(fps, 1e-3)
    frame_idx = 0
    t0 = time.time()
    with open(faces_path, "w", encoding="utf-8") as faces_f, open(cv_events_path, "w", encoding="utf-8") as cv_f:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ts = frame_idx * frame_time
            results, events = analyzer.analyze_frame(frame, ts)
            for r in results:
                snap = {
                    "ts": float(ts),
                    "track_id": str(r.get("track_id", r.get("student_id"))),
                    "bbox": r.get("bbox"),
                    "center_x": r.get("center_x"),
                    "center_y": r.get("center_y"),
                    "area": r.get("area"),
                    "state": r.get("state"),
                    "ear": r.get("ear"),
                    "pitch": r.get("pitch"),
                    "frame": frame_idx,
                }
                append_jsonl(faces_f, snap)
            for ev in events:
                ev_out = dict(ev)
                ev_out.setdefault("ts", float(ts))
                append_jsonl(cv_f, ev_out)
            frame_idx += 1
            if args.max_frames and frame_idx >= args.max_frames:
                break
            if args.show:
                # display debug overlay if debug_draw enabled
                cv2.imshow('Demo', frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    print('ESC pressed, exiting early from video')
                    break
    cap.release()
    if args.show:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
    print(f"[demo] Face analyzer processed {frame_idx} frames in {time.time() - t0:.2f}s (fps source={fps:.2f})")

    # 3. Transcribe audio (if requested)
    asr_events: List[Dict] = []
    if not args.skip_asr:
        if args.api_key:
            os.environ["DASHSCOPE_API_KEY"] = args.api_key
        playback_clock = {"t": 0.0}

        def time_base() -> float:
            return playback_clock["t"]

        def progress_hook(seconds: float) -> None:
            playback_clock["t"] = seconds

        # Implement final-first + stability-fallback ASR buffering.
        # We group incoming sentence events into small time buckets (configurable)
        # and accept a sentence immediately if it's marked final by the ASR raw
        # payload. Otherwise we wait until the hypothesis for that bucket
        # remains unchanged for `asr_stability_secs` and then accept it.
        asr_buffers: Dict[int, Dict] = {}
        asr_stability_secs = float(args.asr_stability_secs)
        asr_bucket_width = float(args.asr_bucket_width)

        def _is_final_raw(raw: Dict) -> bool:
            # Heuristic checks for final markers in the raw result dict
            if not isinstance(raw, dict):
                return False
            for k in ("is_final", "final", "is_final_result", "final_result", "is_last"):
                v = raw.get(k)
                if v:
                    return True
            if raw.get("status") in ("final", "completed"):
                return True
            return False

        def _bucket_for_ts(ts: float) -> int:
            return int(ts // asr_bucket_width)

        def _accept_buffer(bucket: int):
            buf = asr_buffers.get(bucket)
            if not buf:
                return
            text = buf.get("final_text") or buf.get("last_text") or ""
            ev = {"ts": float(buf.get("first_ts", 0.0)), "type": "ASR_SENTENCE", "text": text, "raw": buf.get("raw_last")}
            asr_events.append(ev)
            # once accepted, remove buffer
            del asr_buffers[bucket]

        def on_sentence(ev: Dict) -> None:
            # Called by transcribe_file for each recognition event
            ts = float(ev.get("ts", 0.0))
            text = _parse_text_field(ev.get("text") or ev.get("text", ""))
            raw = ev.get("raw") or {}
            now = time_base()
            bucket = _bucket_for_ts(ts)
            buf = asr_buffers.get(bucket)
            if buf is None:
                buf = {
                    "first_ts": ts,
                    "last_text": text,
                    "raw_last": raw,
                    "last_change_time": now,
                    "last_seen_time": now,
                    "final_text": None,
                }
                asr_buffers[bucket] = buf
            else:
                # update last seen
                if text != buf.get("last_text"):
                    buf["last_text"] = text
                    buf["raw_last"] = raw
                    buf["last_change_time"] = now
                buf["last_seen_time"] = now

            # If raw payload indicates this is a final result, accept immediately
            if _is_final_raw(raw):
                buf["final_text"] = text
                _accept_buffer(bucket)
                return

            # Otherwise, if hypothesis hasn't changed for stability window, accept
            if now - buf.get("last_change_time", now) >= asr_stability_secs:
                _accept_buffer(bucket)
                return

            # Periodically, flush any older buffers that haven't been seen for a while
            # to avoid leaking memory in case of strange ASR event timing.
            stale_threshold = max(5.0, asr_stability_secs * 4)
            stale = [b for b, v in asr_buffers.items() if (now - v.get("last_seen_time", now)) > stale_threshold]
            for b in stale:
                _accept_buffer(b)

        print("[demo] Running ASR transcription (DashScope)")
        transcribe_file(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            wav_path=str(wav_path),
            time_base=time_base,
            on_sentence=on_sentence,
            sample_rate=args.sample_rate,
            chunk_ms=args.chunk_ms,
            progress_hook=progress_hook,
        )
        # After transcription finishes, flush any remaining ASR buffers
        # so hypotheses that never received a final flag but were stable
        # near the end of the file are still accepted.
        for b in list(asr_buffers.keys()):
            try:
                _accept_buffer(b)
            except Exception:
                pass
        print(f"[demo] ASR produced {len(asr_events)} sentence events")
        segments = build_asr_segments(asr_events)
        with open(asr_path, "w", encoding="utf-8") as asr_f:
            for seg in segments:
                append_jsonl(asr_f, seg)
    else:
        print("[demo] Skipping ASR step")

    # 4. Label teacher/student segments
    print("[demo] Loading ASR + face tracks for labeling")
    asr_segments = []
    if asr_path.exists():
        with open(asr_path, "r", encoding="utf-8") as f:
            asr_segments = [json.loads(line) for line in f if line.strip()]
    face_tracks = load_face_tracks_jsonl(str(faces_path)) if faces_path.exists() else {}
    labeled = label_asr_segments(asr_segments, face_tracks)

    save_labeled_jsonl(labeled, str(labeled_path))
    if not args.no_timeline:
        plot_timeline(labeled, str(timeline_path), title="ASR Teacher/Student Timeline", face_tracks=face_tracks, show_text=True, font_path=args.font_path)
        print(f"[demo] Timeline written to {timeline_path}")
    else:
        print("[demo] Timeline output disabled via --no-timeline")

    if args.push_url:
        print(f"[demo] Pushing {len(labeled)} segments to {args.push_url}")
        post_segments(args.push_url, labeled)

    print("[demo] Outputs:")
    for path in [faces_path, cv_events_path, asr_path, labeled_path] + ([] if args.no_timeline else [timeline_path]):
        if Path(path).exists():
            print(f"  - {path}")


if __name__ == "__main__":
    main()
