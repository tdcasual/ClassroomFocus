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

# Ensure project root is in path
import sys
PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from cv.face_analyzer import FaceAnalyzer, FaceAnalyzerConfig

logger = logging.getLogger("SessionManager")
logging.basicConfig(level=logging.INFO)

class SessionManager:
    def __init__(self, ffmpeg_path="ffmpeg", camera_index=0, mic_device=None):
        self.ffmpeg_path = ffmpeg_path
        self.camera_index = camera_index
        self.mic_device = mic_device
        
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
        
        # Audio buffer
        self.audio_queue = Queue()
        self.sample_rate = 16000
        self.channels = 1

    def set_callback(self, callback):
        self.on_data_callback = callback

    def start(self, output_dir_base="out"):
        if self.is_recording:
            logger.warning("Session already running")
            return self.session_id

        self.session_id = f"session_{int(time.time())}"
        self.output_dir = Path(output_dir_base) / self.session_id
        self.video_thread = None
        self.audio_thread = None
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
            cfg = FaceAnalyzerConfig() # Use defaults or allow tuning
            self.face_analyzer = FaceAnalyzer(cfg)

            self.is_recording = True

            # Start threads
            self.video_thread = threading.Thread(target=self._video_loop, daemon=True)
            self.audio_thread = threading.Thread(target=self._audio_loop, daemon=True)
            
            self.video_thread.start()
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
            
        logger.info("Stopping session...")
        self.stop_event.set()

        # Flip state after signaling stop (threads may check is_recording).
        self.is_recording = False
        
        if self.video_thread:
            self.video_thread.join()
        if self.audio_thread:
            self.audio_thread.join()
        self.video_thread = None
        self.audio_thread = None

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
        audio_path = (self.output_dir / "temp_audio.wav") if self.output_dir else None
            
        # Post-processing: Mux video and audio
        self._mux_files()

        # Validate audio/video after mux so artifacts are still produced for debugging.
        if self.audio_error:
            raise RuntimeError(f"Audio recording failed: {self.audio_error}")
        if audio_path and (not audio_path.exists() or audio_path.stat().st_size <= 0):
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

        # Video Writer setup
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        temp_video_path = self.output_dir / "temp_video.avi"
        # MJPG is usually safe and widely supported for AVI
        fourcc = cv2.VideoWriter_fourcc(*'MJPG') 
        out = cv2.VideoWriter(str(temp_video_path), fourcc, fps, (width, height))

        faces_path = self.output_dir / "faces.jsonl"
        events_path = self.output_dir / "cv_events.jsonl"
        
        frame_idx = 0
        start_time = self.session_start_wall or time.time()
        last_preview_ts = -1e9
        preview_interval_sec = float(os.getenv("WEB_PREVIEW_INTERVAL_SEC", "0.15"))  # ~6-7 fps by default

        try:
            with open(faces_path, "w", encoding="utf-8") as f_faces, \
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
                    results, events = self.face_analyzer.analyze_frame(frame, ts)
                    
                    # Write to files
                    for r in results:
                        # Add frame index
                        r['frame'] = frame_idx
                        f_faces.write(json.dumps(r, ensure_ascii=False) + "\n")
                    
                    for e in events:
                        if 'ts' not in e or e['ts'] is None:
                            e['ts'] = ts
                        f_events.write(json.dumps(e, ensure_ascii=False) + "\n")
    
                    # Write video
                    out.write(frame)
                    
                    # Callback for Web Viz
                    if self.on_data_callback:
                        img_str = None
                        # Throttle preview frames to reduce CPU/network usage.
                        if (ts - last_preview_ts) >= preview_interval_sec:
                            last_preview_ts = ts
                            try:
                                _, buffer = cv2.imencode('.jpg', cv2.resize(frame, (320, 240)))
                                img_str = buffer.tobytes()
                            except Exception:
                                img_str = None
                        
                        data = {
                            "type": "frame_data",
                            "ts": ts,
                            "faces": results,
                            "events": events,
                        }
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
                ):
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
