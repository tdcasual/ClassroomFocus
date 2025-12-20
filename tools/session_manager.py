import base64
import cv2
import time
import threading
import json
import os
import sounddevice as sd
import soundfile as sf
import numpy as np
from pathlib import Path
from queue import Queue, Empty
import subprocess
import logging
from contextlib import nullcontext
from typing import Dict

# Ensure project root is in path
import sys
PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from config import load_session_config
from cv.face_analyzer import FaceAnalyzer, FaceAnalyzerConfig, AdaptiveScheduler
from tools.thumb_utils import effective_refresh_sec

logger = logging.getLogger("SessionManager")
logging.basicConfig(level=logging.INFO)

class SessionManager:
    def __init__(self, ffmpeg_path="ffmpeg", camera_index=0, mic_device=None):
        self.ffmpeg_path = ffmpeg_path
        self.camera_index = camera_index
        self.mic_device = mic_device
        env_cfg = load_session_config()
        self.preview_enabled = bool(env_cfg.preview_enabled)
        self.preview_interval_sec = float(env_cfg.preview_interval_sec)
        self.preview_max_width = int(env_cfg.preview_max_width)
        self.preview_max_height = int(env_cfg.preview_max_height)
        self.video_write_enabled = bool(env_cfg.video_write_enabled)
        self.audio_enabled = bool(env_cfg.audio_enabled)
        self.faces_write_enabled = bool(env_cfg.faces_write_enabled)
        self.faces_sample_sec = float(env_cfg.faces_sample_sec)
        self.stop_soft_timeout_sec = float(env_cfg.stop_soft_timeout_sec)
        self.stop_hard_timeout_sec = float(env_cfg.stop_hard_timeout_sec)
        self.preview_thumbs_enabled = bool(env_cfg.preview_thumbs_enabled)
        self.preview_thumb_size = int(env_cfg.preview_thumb_size)
        self.preview_thumb_quality = int(env_cfg.preview_thumb_quality)
        self.preview_thumb_pad = float(env_cfg.preview_thumb_pad)
        self.preview_thumb_refresh_sec = float(env_cfg.preview_thumb_refresh_sec)
        self.preview_thumb_refresh_frames = int(env_cfg.preview_thumb_refresh_frames)
        self.face_cfg = None
        self.device_profile = None
        self.profile_overrides = {}
        self.adaptive_scheduler = None
        
        self.is_recording = False
        self.stop_event = threading.Event()
        self.output_dir = None
        self.session_id = None
        
        self.video_thread = None
        self.audio_thread = None
        self.session_start_wall = None
        self.video_start_wall = None
        self.audio_start_wall = None
        
        self.face_analyzer = None
        self.on_data_callback = None # Function to call with frame data for web viz
        self.audio_error = None
        self.video_error = None
        self._video_cap = None
        self._audio_stream = None
        self._preview_thumb_ts = {}
        self._preview_thumb_cache = {}
        self.last_stop_duration_sec = None
        self.last_stop_hard_used = False
        
        # Audio buffer
        self.audio_queue = Queue()
        self.sample_rate = 16000
        self.channels = 1

    def set_preview(self, enabled=None, interval_sec=None):
        if enabled is not None:
            self.preview_enabled = bool(enabled)
        if interval_sec is not None:
            try:
                self.preview_interval_sec = float(interval_sec)
            except Exception:
                pass

    def set_preview_thumbs(self, enabled=None, size=None, quality=None, pad=None, refresh_sec=None, refresh_frames=None):
        if enabled is not None:
            self.preview_thumbs_enabled = bool(enabled)
        if size is not None:
            try:
                self.preview_thumb_size = int(size)
            except Exception:
                pass
        if quality is not None:
            try:
                self.preview_thumb_quality = int(quality)
            except Exception:
                pass
        if pad is not None:
            try:
                self.preview_thumb_pad = float(pad)
            except Exception:
                pass
        if refresh_sec is not None:
            try:
                self.preview_thumb_refresh_sec = float(refresh_sec)
            except Exception:
                pass
        if refresh_frames is not None:
            try:
                self.preview_thumb_refresh_frames = int(refresh_frames)
            except Exception:
                pass

    def set_video_write(self, enabled=None):
        if enabled is not None:
            self.video_write_enabled = bool(enabled)

    def set_audio_enabled(self, enabled=None):
        if enabled is not None:
            self.audio_enabled = bool(enabled)

    def set_faces_write(self, enabled=None, sample_sec=None):
        if enabled is not None:
            self.faces_write_enabled = bool(enabled)
        if sample_sec is not None:
            try:
                self.faces_sample_sec = float(sample_sec)
            except Exception:
                pass

    def set_face_config(self, cfg):
        self.face_cfg = cfg

    def set_device_profile(self, profile, overrides=None):
        self.device_profile = profile
        self.profile_overrides = dict(overrides or {})

    def set_adaptive_scheduler(self, enabled=None, **kwargs):
        if enabled is False:
            self.adaptive_scheduler = None
            return
        if enabled is True:
            self.adaptive_scheduler = AdaptiveScheduler(**kwargs)

    def set_callback(self, callback):
        self.on_data_callback = callback

    def _reset_preview_thumbs(self) -> None:
        self._preview_thumb_ts = {}
        self._preview_thumb_cache = {}

    def _preview_thumb_refresh_window(self) -> float:
        return effective_refresh_sec(
            self.preview_thumb_refresh_sec,
            self.preview_interval_sec,
            self.preview_thumb_refresh_frames,
        )

    def _resize_preview_frame(self, frame: np.ndarray) -> np.ndarray:
        try:
            h, w = frame.shape[:2]
        except Exception:
            return frame
        max_w = int(self.preview_max_width)
        max_h = int(self.preview_max_height)
        if max_w <= 0 or max_h <= 0:
            return frame
        if w <= max_w and h <= max_h:
            return frame
        scale = min(float(max_w) / max(1.0, w), float(max_h) / max(1.0, h))
        if scale >= 1.0:
            return frame
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        try:
            return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        except Exception:
            return frame

    def _build_preview_thumbs(self, frame: np.ndarray, results: list, ts: float) -> Dict[str, str]:
        if not self.preview_thumbs_enabled or frame is None or not results:
            return {}
        refresh_sec = self._preview_thumb_refresh_window()
        now_ts = float(ts)
        h, w = frame.shape[:2]
        if h <= 0 or w <= 0:
            return {}

        updates: Dict[str, str] = {}
        pad = float(self.preview_thumb_pad)
        size = int(self.preview_thumb_size)
        quality = int(self.preview_thumb_quality)

        for r in results:
            if not isinstance(r, dict):
                continue
            sid = r.get("track_id")
            if sid is None:
                sid = r.get("student_id")
            if sid is None:
                continue
            sid = str(sid)
            last_ts = self._preview_thumb_ts.get(sid)
            if sid in self._preview_thumb_cache and refresh_sec > 0 and last_ts is not None:
                if (now_ts - float(last_ts)) < refresh_sec:
                    continue
            bbox = r.get("bbox")
            if not (isinstance(bbox, list) and len(bbox) >= 4):
                continue
            try:
                nx, ny, nw, nh = [float(b) for b in bbox[:4]]
            except Exception:
                continue
            if nw <= 0 or nh <= 0:
                continue
            cx = nx + nw / 2.0
            cy = ny + nh / 2.0
            box = max(nw, nh) * (1.0 + pad * 2.0)
            sx0 = max(0.0, min(1.0, cx - box / 2.0))
            sy0 = max(0.0, min(1.0, cy - box / 2.0))
            sx1 = max(0.0, min(1.0, cx + box / 2.0))
            sy1 = max(0.0, min(1.0, cy + box / 2.0))
            x0 = int(max(0, min(w - 1, round(sx0 * w))))
            y0 = int(max(0, min(h - 1, round(sy0 * h))))
            x1 = int(max(1, min(w, round(sx1 * w))))
            y1 = int(max(1, min(h, round(sy1 * h))))
            if x1 <= x0 or y1 <= y0:
                continue
            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            try:
                thumb = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode(".jpg", thumb, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            except Exception:
                continue
            if not ok:
                continue
            b64 = base64.b64encode(buf).decode("utf-8")
            self._preview_thumb_cache[sid] = b64
            self._preview_thumb_ts[sid] = now_ts
            updates[sid] = b64
        return updates

    def _force_stop_resources(self) -> None:
        try:
            if self._video_cap is not None:
                self._video_cap.release()
        except Exception:
            pass
        try:
            if self._audio_stream is not None:
                try:
                    self._audio_stream.stop()
                except Exception:
                    pass
                try:
                    self._audio_stream.close()
                except Exception:
                    pass
        except Exception:
            pass

    def start(self, output_dir_base="out"):
        if self.is_recording:
            logger.warning("Session already running")
            return self.session_id

        self.session_id = f"session_{int(time.time())}"
        self.output_dir = Path(output_dir_base) / self.session_id
        self.video_thread = None
        self.audio_thread = None
        self._reset_preview_thumbs()
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            # Reset state so /api/session/status won't report a phantom session.
            self.session_id = None
            self.output_dir = None
            raise
        
        logger.info(f"Starting session: {self.session_id} in {self.output_dir}")

        self.stop_event.clear()
        self.audio_error = None
        self.video_error = None
        self.session_start_wall = time.time()
        self.video_start_wall = None
        self.audio_start_wall = None

        try:
            # Preflight camera so we fail fast and don't leave the audio thread running.
            cap = cv2.VideoCapture(self.camera_index)
            try:
                if not cap.isOpened():
                    raise RuntimeError(f"Could not open camera (index={self.camera_index})")
            finally:
                try:
                    cap.release()
                except Exception:
                    pass

            # Initialize FaceAnalyzer
            if self.face_cfg is not None:
                cfg = self.face_cfg
            elif self.device_profile is not None:
                cfg = FaceAnalyzerConfig.from_device_profile(self.device_profile)
                for key, value in (self.profile_overrides or {}).items():
                    if hasattr(cfg, key):
                        setattr(cfg, key, value)
            else:
                cfg = FaceAnalyzerConfig() # Use defaults or allow tuning
            self.face_analyzer = FaceAnalyzer(cfg)
            if self.adaptive_scheduler:
                self.adaptive_scheduler.reset()

            self.is_recording = True

            # Start threads
            self.video_thread = threading.Thread(target=self._video_loop, daemon=True)
            self.audio_thread = threading.Thread(target=self._audio_loop, daemon=True) if self.audio_enabled else None
            
            self.video_thread.start()
            if self.audio_thread:
                self.audio_thread.start()
            
            return self.session_id
        except Exception:
            # Roll back state; keep output dir for debugging but avoid dangling "recording" state.
            self.is_recording = False
            self.stop_event.set()
            try:
                if self.video_thread:
                    self.video_thread.join(timeout=1.0)
            except Exception:
                pass
            try:
                if self.audio_thread:
                    self.audio_thread.join(timeout=1.0)
            except Exception:
                pass
            self.video_thread = None
            self.audio_thread = None
            # Expose no active session on failures.
            self.session_id = None
            self.output_dir = None
            raise

    def stop(self):
        # Allow stop() even if a worker thread already flipped is_recording due to an error.
        if not self.session_id or not self.output_dir:
            return None
        t0 = time.perf_counter()
        logger.info("Stopping session...")
        self.stop_event.set()

        # Flip state after signaling stop (threads may check is_recording).
        self.is_recording = False

        soft_timeout = max(0.1, float(self.stop_soft_timeout_sec))
        hard_timeout = max(0.1, float(self.stop_hard_timeout_sec))
        self.last_stop_hard_used = False

        if self.video_thread:
            self.video_thread.join(timeout=soft_timeout)
        if self.audio_thread:
            self.audio_thread.join(timeout=soft_timeout)

        hard_needed = False
        if self.video_thread and self.video_thread.is_alive():
            hard_needed = True
        if self.audio_thread and self.audio_thread.is_alive():
            hard_needed = True

        if hard_needed:
            self.last_stop_hard_used = True
            self._force_stop_resources()
            if self.video_thread:
                self.video_thread.join(timeout=hard_timeout)
            if self.audio_thread:
                self.audio_thread.join(timeout=hard_timeout)

        if self.video_thread and self.video_thread.is_alive():
            logger.warning("Video thread did not stop within %.2fs", soft_timeout + hard_timeout)
        if self.audio_thread and self.audio_thread.is_alive():
            logger.warning("Audio thread did not stop within %.2fs", soft_timeout + hard_timeout)

        self.video_thread = None
        self.audio_thread = None
        self._reset_preview_thumbs()
        self.last_stop_duration_sec = time.perf_counter() - t0
        logger.info("Stop complete in %.2fs (hard=%s)", self.last_stop_duration_sec, self.last_stop_hard_used)

        # Persist sync metadata to align CV timestamps with audio/ASR timestamps.
        try:
            if self.output_dir:
                sync = {
                    "session_start_wall": float(self.session_start_wall) if self.session_start_wall else None,
                    "video_start_wall": float(self.video_start_wall) if self.video_start_wall else None,
                    "audio_start_wall": float(self.audio_start_wall) if self.audio_start_wall else None,
                    "audio_offset_sec": (float(self.audio_start_wall) - float(self.session_start_wall)) if (self.audio_start_wall and self.session_start_wall) else 0.0,
                    "video_offset_sec": (float(self.video_start_wall) - float(self.session_start_wall)) if (self.video_start_wall and self.session_start_wall) else 0.0,
                    "sample_rate": int(self.sample_rate),
                    "channels": int(self.channels),
                }
                with open(self.output_dir / "sync.json", "w", encoding="utf-8") as fh:
                    json.dump(sync, fh, ensure_ascii=False, indent=2)
        except Exception:
            pass

        # Ensure we actually captured audio for the whole session.
        audio_path = (self.output_dir / "temp_audio.wav") if (self.output_dir and self.audio_enabled) else None
            
        # Post-processing: Mux video and audio
        self._mux_files()

        # Validate audio/video after mux so artifacts are still produced for debugging.
        if self.audio_enabled and self.audio_error:
            raise RuntimeError(f"Audio recording failed: {self.audio_error}")
        if self.audio_enabled and audio_path and (not audio_path.exists() or audio_path.stat().st_size <= 0):
            raise RuntimeError("Audio recording output is missing/empty (temp_audio.wav).")
        if self.video_error:
            # Video problems are reported as warnings so we still keep the audio
            # and any partial artifacts for debugging/reporting.
            logger.warning(f"Video recording warning: {self.video_error}")
        
        logger.info("Session stopped and saved.")
        return str(self.output_dir)

    def _video_loop(self):
        if self.video_start_wall is None:
            self.video_start_wall = time.time()
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            msg = f"Could not open camera (index={self.camera_index})"
            self.video_error = msg
            logger.error(msg)
            self.is_recording = False
            self.stop_event.set()
            try:
                if self.on_data_callback:
                    self.on_data_callback({"type": "error", "ts": 0.0, "error": msg}, None)
            except Exception:
                pass
            try:
                cap.release()
            except Exception:
                pass
            return
        self._video_cap = cap

        # Video Writer setup
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        out = None
        if self.video_write_enabled:
            temp_video_path = self.output_dir / "temp_video.avi"
            # MJPG is usually safe and widely supported for AVI
            fourcc = cv2.VideoWriter_fourcc(*'MJPG') 
            out = cv2.VideoWriter(str(temp_video_path), fourcc, fps, (width, height))

        faces_path = self.output_dir / "faces.jsonl"
        events_path = self.output_dir / "cv_events.jsonl"
        
        frame_idx = 0
        start_time = self.session_start_wall or time.time()
        last_preview_ts = -1e9
        preview_interval_sec = self.preview_interval_sec
        preview_enabled = bool(self.preview_enabled) and preview_interval_sec > 0.0
        last_face_write_ts = -1e9
        faces_write_enabled = bool(self.faces_write_enabled)
        faces_sample_sec = float(self.faces_sample_sec or 0.0)

        try:
            faces_ctx = open(faces_path, "w", encoding="utf-8") if faces_write_enabled else nullcontext(None)
            with faces_ctx as f_faces, \
                 open(events_path, "w", encoding="utf-8") as f_events:
                
                while not self.stop_event.is_set():
                    ret, frame = cap.read()
                    if not ret:
                        # Camera disconnected / stream ended.
                        self.video_error = "camera read failed"
                        self.stop_event.set()
                        break
                    
                    current_time = time.time()
                    ts = current_time - start_time
                    
                    # Analyze
                    infer_t0 = time.perf_counter()
                    results, events = self.face_analyzer.analyze_frame(frame, ts)
                    infer_ms = (time.perf_counter() - infer_t0) * 1000.0
                    if self.adaptive_scheduler and self.face_analyzer and self.face_analyzer._mesh is not None:
                        processed = True
                        if self.face_analyzer.cfg.process_every_n > 1:
                            processed = (self.face_analyzer._frame_count % self.face_analyzer.cfg.process_every_n) == 0
                        if processed:
                            self.adaptive_scheduler.update(infer_ms=infer_ms, cpu_percent=None)
                            self.adaptive_scheduler.apply_to_config(self.face_analyzer.cfg)
                    
                    # Write to files
                    write_faces = False
                    if f_faces is not None:
                        write_faces = faces_sample_sec <= 0 or (ts - last_face_write_ts) >= faces_sample_sec
                    if write_faces:
                        last_face_write_ts = ts
                        for r in results:
                            # Add frame index
                            r['frame'] = frame_idx
                            f_faces.write(json.dumps(r, ensure_ascii=False) + "\n")
                    
                    for e in events:
                        if 'ts' not in e or e['ts'] is None:
                            e['ts'] = ts
                        f_events.write(json.dumps(e, ensure_ascii=False) + "\n")
    
                    # Write video
                    if out:
                        out.write(frame)
                    
                    # Callback for Web Viz
                    if self.on_data_callback:
                        img_str = None
                        thumb_updates = None
                        # Throttle preview frames to reduce CPU/network usage.
                        if preview_enabled and (ts - last_preview_ts) >= preview_interval_sec:
                            last_preview_ts = ts
                            try:
                                preview_frame = self._resize_preview_frame(frame)
                                _, buffer = cv2.imencode('.jpg', preview_frame)
                                img_str = buffer.tobytes()
                                if self.preview_thumbs_enabled:
                                    thumb_updates = self._build_preview_thumbs(preview_frame, results, ts)
                            except Exception:
                                img_str = None
                        
                        data = {
                            "type": "frame_data",
                            "ts": ts,
                            "faces": results,
                            "events": events,
                        }
                        if thumb_updates:
                            data["thumbs"] = thumb_updates
                        self.on_data_callback(data, img_str)
    
                    frame_idx += 1
        except Exception as exc:
            self.video_error = str(exc)
            self.stop_event.set()
        finally:
            try:
                cap.release()
            except Exception:
                pass
            self._video_cap = None
            if out:
                try:
                    out.release()
                except Exception:
                    pass

    def _audio_loop(self):
        temp_audio_path = self.output_dir / "temp_audio.wav"
        
        try:
            with sf.SoundFile(str(temp_audio_path), mode='w', samplerate=self.sample_rate, channels=self.channels, subtype="PCM_16") as file:
                with sd.InputStream(
                    samplerate=self.sample_rate,
                    device=self.mic_device,
                    channels=self.channels,
                    dtype="int16",
                    callback=self._audio_callback,
                ) as stream:
                    self._audio_stream = stream
                    if self.audio_start_wall is None:
                        self.audio_start_wall = time.time()
                    while not self.stop_event.is_set():
                        try:
                            block = self.audio_queue.get(timeout=0.2)
                        except Empty:
                            continue
                        file.write(block)
                    # Drain any remaining buffered audio frames.
                    while True:
                        try:
                            block = self.audio_queue.get_nowait()
                        except Empty:
                            break
                        file.write(block)
        except Exception as e:
            self.audio_error = str(e)
            logger.error(f"Audio recording failed: {e}")
        finally:
            self._audio_stream = None

    def _audio_callback(self, indata, frames, time, status):
        if status:
            logger.warning(status)
        self.audio_queue.put(indata.copy())

    def _mux_files(self):
        # Combine temp_video.avi and temp_audio.wav into session.mp4
        video_in = self.output_dir / "temp_video.avi"
        audio_in = self.output_dir / "temp_audio.wav"
        output_mp4 = self.output_dir / "session.mp4"
        
        if not video_in.exists():
            return

        cmd = [
            self.ffmpeg_path, "-y",
            "-i", str(video_in),
        ]
        
        if audio_in.exists():
            cmd.extend(["-i", str(audio_in)])
            # Map streams: video from 0, audio from 1
            # Use aac for audio, copy or h264 for video
            cmd.extend(["-c:v", "copy", "-c:a", "aac"])
        else:
            cmd.extend(["-c:v", "copy"])
            
        cmd.append(str(output_mp4))
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # Cleanup temps
            # video_in.unlink()
            # audio_in.unlink()
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg muxing failed: {e.stderr.decode()}")
        except FileNotFoundError:
            logger.error("FFmpeg executable not found.")

if __name__ == "__main__":
    # Simple test
    mgr = SessionManager(ffmpeg_path=r"C:\Users\HP\Downloads\ffmpeg\bin\ffmpeg.exe")
    mgr.start()
    time.sleep(10)
    mgr.stop()
