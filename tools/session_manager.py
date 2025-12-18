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
        
        self.face_analyzer = None
        self.on_data_callback = None # Function to call with frame data for web viz
        
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
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting session: {self.session_id} in {self.output_dir}")

        self.stop_event.clear()
        self.is_recording = True
        
        # Initialize FaceAnalyzer
        cfg = FaceAnalyzerConfig() # Use defaults or allow tuning
        self.face_analyzer = FaceAnalyzer(cfg)

        # Start threads
        self.video_thread = threading.Thread(target=self._video_loop)
        self.audio_thread = threading.Thread(target=self._audio_loop)
        
        self.video_thread.start()
        self.audio_thread.start()
        
        return self.session_id

    def stop(self):
        if not self.is_recording:
            return None
            
        logger.info("Stopping session...")
        self.is_recording = False
        self.stop_event.set()
        
        if self.video_thread:
            self.video_thread.join()
        if self.audio_thread:
            self.audio_thread.join()
            
        # Post-processing: Mux video and audio
        self._mux_files()
        
        logger.info("Session stopped and saved.")
        return str(self.output_dir)

    def _video_loop(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            logger.error("Could not open camera")
            self.is_recording = False
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
        start_time = time.time()
        last_preview_ts = -1e9
        preview_interval_sec = float(os.getenv("WEB_PREVIEW_INTERVAL_SEC", "0.15"))  # ~6-7 fps by default

        with open(faces_path, "w", encoding="utf-8") as f_faces, \
             open(events_path, "w", encoding="utf-8") as f_events:
            
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
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
                    
                    # Save thumbnail if needed (e.g. every 5 seconds or on event)
                    # For now, let's just pass the data to callback
                
                for e in events:
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
                        # "image": img_str # Handle binary separately or base64 encode if needed
                    }
                    self.on_data_callback(data, img_str)

                frame_idx += 1
                # Control FPS if needed, but camera read usually blocks to FPS
        
        cap.release()
        out.release()

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
                    while not self.stop_event.is_set():
                        try:
                            block = self.audio_queue.get(timeout=0.2)
                        except Empty:
                            continue
                        file.write(block)
        except Exception as e:
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
