# cv/face_analyzer.py
# -*- coding: utf-8 -*-
"""Face analysis: EAR-based drowsiness, pitch-based looking-down detection, simple multi-face tracking."""
from __future__ import annotations

import time
import threading
import queue
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None
try:
    import mediapipe as mp  # type: ignore
except Exception:  # pragma: no cover
    mp = None
import numpy as np
try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None


def refresh_results_timestamps(results: List[Dict[str, Any]], timestamp: float) -> None:
    """Update cached result timestamps for skipped frames."""
    if not isinstance(results, list):
        return
    try:
        ts = float(timestamp)
    except Exception:
        return
    for item in results:
        if isinstance(item, dict):
            item["ts"] = ts


# -----------------------------------------------------------------------------
# CaptureQueue: Frame dropping queue for constrained devices
# -----------------------------------------------------------------------------

class CaptureQueue:
    """
    A frame capture queue that drops old frames when the processing thread
    can't keep up. Designed for constrained devices like Raspberry Pi.
    
    Usage:
        cap = cv2.VideoCapture(0)
        queue = CaptureQueue(maxsize=1, drop_old=True)
        queue.start_capture(cap)
        
        while running:
            frame, ts = queue.get_frame(timeout=0.1)
            if frame is not None:
                results, events = analyzer.analyze_frame(frame, ts)
        
        queue.stop()
    """
    
    def __init__(self, maxsize: int = 1, drop_old: bool = True):
        """
        Args:
            maxsize: Maximum frames to buffer (1 recommended for real-time)
            drop_old: If True, drop oldest frame when full; if False, drop new frame
        """
        self._maxsize = max(1, int(maxsize))
        self._drop_old = drop_old
        self._queue: queue.Queue = queue.Queue(maxsize=self._maxsize + 1)  # +1 for drop logic
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cap = None
        self._frames_captured = 0
        self._frames_dropped = 0
        self._lock = threading.Lock()
    
    def start_capture(self, cap) -> None:
        """Start the capture thread with the given VideoCapture object."""
        self._cap = cap
        self._stop_event.clear()
        self._frames_captured = 0
        self._frames_dropped = 0
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
    
    def _capture_loop(self) -> None:
        """Internal capture loop running in a separate thread."""
        while not self._stop_event.is_set() and self._cap is not None:
            try:
                ok, frame = self._cap.read()
                if not ok or frame is None:
                    time.sleep(0.01)
                    continue
                
                ts = time.time()
                self._frames_captured += 1
                
                with self._lock:
                    if self._queue.full():
                        if self._drop_old:
                            # Drop oldest frame
                            try:
                                self._queue.get_nowait()
                                self._frames_dropped += 1
                            except queue.Empty:
                                pass
                        else:
                            # Drop new frame
                            self._frames_dropped += 1
                            continue
                    
                    try:
                        self._queue.put_nowait((frame, ts))
                    except queue.Full:
                        self._frames_dropped += 1
                        
            except Exception:
                time.sleep(0.01)
    
    def get_frame(self, timeout: float = 0.1) -> Tuple[Optional[np.ndarray], float]:
        """
        Get the next frame from the queue.
        
        Returns:
            (frame, timestamp) or (None, 0.0) if no frame available
        """
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None, 0.0
    
    def get_frame_nowait(self) -> Tuple[Optional[np.ndarray], float]:
        """Get frame without waiting, returns (None, 0.0) if empty."""
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None, 0.0
    
    def stop(self) -> None:
        """Stop the capture thread."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
    
    def stats(self) -> Dict[str, int]:
        """Return capture statistics."""
        return {
            "frames_captured": self._frames_captured,
            "frames_dropped": self._frames_dropped,
            "drop_rate": self._frames_dropped / max(1, self._frames_captured),
            "queue_size": self._queue.qsize(),
        }
    
    def is_running(self) -> bool:
        """Check if capture thread is running."""
        return self._thread is not None and self._thread.is_alive()


# -----------------------------------------------------------------------------
# AdaptiveScheduler: Dynamic performance adjustment based on CPU/latency
# -----------------------------------------------------------------------------

class AdaptiveScheduler:
    """
    Soft real-time adaptive scheduler that dynamically adjusts processing parameters
    based on CPU usage and inference latency. Uses a sliding window and PID-style control.
    
    When CPU > threshold or latency > target, increases process_every_n or decreases input_scale.
    When system is underutilized, gradually restores quality.
    
    Usage:
        scheduler = AdaptiveScheduler(target_cpu=75, target_latency_ms=100)
        
        while running:
            results, events = analyzer.analyze_frame(frame, ts)
            scheduler.update(infer_ms=infer_time, cpu_percent=cpu)
            
            # Apply scheduler recommendations
            if scheduler.should_skip_frame():
                continue
            
            # Get current recommended scale
            scale = scheduler.recommended_scale
    """
    
    def __init__(
        self,
        target_cpu: float = 75.0,
        target_latency_ms: float = 100.0,
        window_size: int = 30,
        min_scale: float = 0.25,
        max_scale: float = 1.0,
        min_skip: int = 1,
        max_skip: int = 5,
        # PID-style gains
        kp: float = 0.1,  # Proportional gain
        ki: float = 0.02,  # Integral gain
        kd: float = 0.05,  # Derivative gain
    ):
        self.target_cpu = target_cpu
        self.target_latency_ms = target_latency_ms
        self.window_size = window_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_skip = min_skip
        self.max_skip = max_skip
        
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        # State
        self._cpu_history: List[float] = []
        self._latency_history: List[float] = []
        self._frame_count = 0
        self._recommended_scale = 1.0
        self._recommended_skip = 1
        self._integral_error = 0.0
        self._prev_error = 0.0
        self._last_update = time.time()
        
        # Try to import psutil for CPU monitoring
        self._psutil = None
        try:
            import psutil
            self._psutil = psutil
        except ImportError:
            pass
    
    def update(self, infer_ms: float, cpu_percent: Optional[float] = None) -> None:
        """
        Update scheduler with latest metrics.
        
        Args:
            infer_ms: Latest inference time in milliseconds
            cpu_percent: Current CPU usage (if None, will try to sample)
        """
        # Sample CPU if not provided
        if cpu_percent is None and self._psutil:
            cpu_percent = self._psutil.cpu_percent()
        
        # Update histories
        self._latency_history.append(infer_ms)
        if cpu_percent is not None:
            self._cpu_history.append(cpu_percent)
        
        # Trim histories
        if len(self._latency_history) > self.window_size:
            self._latency_history = self._latency_history[-self.window_size:]
        if len(self._cpu_history) > self.window_size:
            self._cpu_history = self._cpu_history[-self.window_size:]
        
        self._frame_count += 1
        
        # Only update recommendations every few frames
        if self._frame_count % 10 != 0:
            return
        
        self._update_recommendations()
    
    def _update_recommendations(self) -> None:
        """Update scale/skip recommendations using PID-style control."""
        if len(self._latency_history) < 5:
            return
        
        # Calculate current averages
        avg_latency = sum(self._latency_history) / len(self._latency_history)
        avg_cpu = sum(self._cpu_history) / len(self._cpu_history) if self._cpu_history else 50.0
        
        # Combined error: weighted sum of latency and CPU errors
        # Positive error means we need to reduce load
        latency_error = (avg_latency - self.target_latency_ms) / self.target_latency_ms
        cpu_error = (avg_cpu - self.target_cpu) / self.target_cpu
        error = max(latency_error, cpu_error)  # Use worst case
        
        # PID control
        dt = max(0.001, time.time() - self._last_update)
        self._last_update = time.time()
        
        p_term = self.kp * error
        self._integral_error = max(-2.0, min(2.0, self._integral_error + error * dt))
        i_term = self.ki * self._integral_error
        d_term = self.kd * (error - self._prev_error) / dt if dt > 0 else 0.0
        self._prev_error = error
        
        adjustment = p_term + i_term + d_term
        
        # Apply adjustment to scale (negative adjustment = increase quality)
        new_scale = self._recommended_scale - adjustment * 0.1
        self._recommended_scale = max(self.min_scale, min(self.max_scale, new_scale))
        
        # Compute skip rate based on load
        if error > 0.3:  # Significantly overloaded
            self._recommended_skip = min(self.max_skip, self._recommended_skip + 1)
        elif error < -0.2 and self._recommended_skip > self.min_skip:  # Underloaded
            self._recommended_skip = max(self.min_skip, self._recommended_skip - 1)
    
    @property
    def recommended_scale(self) -> float:
        """Get current recommended input scale (0.25 to 1.0)."""
        return self._recommended_scale
    
    @property
    def recommended_skip(self) -> int:
        """Get current recommended process_every_n value."""
        return self._recommended_skip
    
    def should_skip_frame(self) -> bool:
        """Check if current frame should be skipped based on recommended_skip."""
        return (self._frame_count % self._recommended_skip) != 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current scheduler statistics."""
        return {
            "frame_count": self._frame_count,
            "recommended_scale": round(self._recommended_scale, 3),
            "recommended_skip": self._recommended_skip,
            "avg_latency_ms": round(sum(self._latency_history) / max(1, len(self._latency_history)), 1),
            "avg_cpu": round(sum(self._cpu_history) / max(1, len(self._cpu_history)), 1) if self._cpu_history else None,
            "integral_error": round(self._integral_error, 3),
        }
    
    def apply_to_config(self, cfg: "FaceAnalyzerConfig") -> None:
        """Apply current recommendations to a FaceAnalyzerConfig."""
        cfg.input_scale = self._recommended_scale
        cfg.process_every_n = self._recommended_skip
    
    def reset(self) -> None:
        """Reset scheduler to initial state."""
        self._cpu_history = []
        self._latency_history = []
        self._frame_count = 0
        self._recommended_scale = 1.0
        self._recommended_skip = 1
        self._integral_error = 0.0
        self._prev_error = 0.0


# -----------------------------------------------------------------------------
# FaceMeshWorker: Multiprocessing-based inference to avoid GIL
# -----------------------------------------------------------------------------

def _worker_process(input_queue, output_queue, cfg_dict: Dict, stop_event):
    """
    Worker process that runs FaceMesh inference.
    Avoids GIL interference by running in a separate process.
    """
    # Import heavy dependencies inside worker process
    try:
        import cv2 as _cv2
        import numpy as _np
        import mediapipe as _mp
    except ImportError as e:
        output_queue.put({"error": f"Import error: {e}"})
        return
    
    # Create FaceAnalyzer with config
    from dataclasses import fields
    cfg = FaceAnalyzerConfig()
    for f in fields(cfg):
        if f.name in cfg_dict:
            setattr(cfg, f.name, cfg_dict[f.name])
    
    analyzer = FaceAnalyzer(cfg)
    
    while not stop_event.is_set():
        try:
            # Get frame from input queue
            item = input_queue.get(timeout=0.1)
            if item is None:  # Shutdown signal
                break
            
            frame_bytes, ts, shape, dtype = item
            frame = _np.frombuffer(frame_bytes, dtype=dtype).reshape(shape)
            
            # Run inference
            t0 = time.perf_counter()
            results, events = analyzer.analyze_frame(frame, ts)
            infer_ms = (time.perf_counter() - t0) * 1000
            
            # Send results back
            output_queue.put({
                "results": results,
                "events": events,
                "ts": ts,
                "infer_ms": infer_ms,
            })
        except queue.Empty:
            continue
        except Exception as e:
            output_queue.put({"error": str(e), "ts": 0})


class FaceMeshWorker:
    """
    Multiprocessing-based FaceMesh worker that runs inference in a separate
    process to avoid GIL interference from the main Python thread.
    
    Uses shared memory / multiprocessing queues for frame passing.
    
    Usage:
        worker = FaceMeshWorker(cfg)
        worker.start()
        
        while running:
            frame, ts = capture_frame()
            worker.submit_frame(frame, ts)
            
            # Get results (non-blocking)
            result = worker.get_result(timeout=0.01)
            if result:
                faces = result["results"]
        
        worker.stop()
    """
    
    def __init__(self, cfg: FaceAnalyzerConfig):
        import multiprocessing as mp
        
        self.cfg = cfg
        self._ctx = mp.get_context("spawn")  # Use spawn for cross-platform compatibility
        self._input_queue = self._ctx.Queue(maxsize=1)
        self._output_queue = self._ctx.Queue(maxsize=10)
        self._stop_event = self._ctx.Event()
        self._process: Optional[mp.Process] = None
        self._frames_submitted = 0
        self._frames_dropped = 0
    
    def start(self) -> None:
        """Start the worker process."""
        if self._process and self._process.is_alive():
            return
        
        # Convert config to dict for passing to worker
        from dataclasses import fields, asdict
        cfg_dict = {}
        for f in fields(self.cfg):
            cfg_dict[f.name] = getattr(self.cfg, f.name)
        
        self._stop_event.clear()
        self._process = self._ctx.Process(
            target=_worker_process,
            args=(self._input_queue, self._output_queue, cfg_dict, self._stop_event),
            daemon=True,
        )
        self._process.start()
    
    def submit_frame(self, frame: np.ndarray, ts: float) -> bool:
        """
        Submit a frame for processing.
        
        Returns True if submitted, False if queue is full (frame dropped).
        """
        self._frames_submitted += 1
        
        # Convert frame to bytes for queue transfer
        frame_bytes = frame.tobytes()
        shape = frame.shape
        dtype = str(frame.dtype)
        
        try:
            # Non-blocking put with immediate drop if full
            self._input_queue.put_nowait((frame_bytes, ts, shape, dtype))
            return True
        except queue.Full:
            self._frames_dropped += 1
            return False
    
    def get_result(self, timeout: float = 0.01) -> Optional[Dict]:
        """
        Get inference result from worker.
        
        Returns dict with keys: results, events, ts, infer_ms
        Or None if no result available.
        """
        try:
            return self._output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_all_results(self) -> List[Dict]:
        """Get all available results without blocking."""
        results = []
        while True:
            try:
                r = self._output_queue.get_nowait()
                results.append(r)
            except queue.Empty:
                break
        return results
    
    def stop(self) -> None:
        """Stop the worker process."""
        self._stop_event.set()
        
        # Send shutdown signal
        try:
            self._input_queue.put_nowait(None)
        except queue.Full:
            pass
        
        if self._process and self._process.is_alive():
            self._process.join(timeout=2.0)
            if self._process.is_alive():
                self._process.terminate()
        self._process = None
    
    def is_running(self) -> bool:
        """Check if worker process is running."""
        return self._process is not None and self._process.is_alive()
    
    def stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        return {
            "is_running": self.is_running(),
            "frames_submitted": self._frames_submitted,
            "frames_dropped": self._frames_dropped,
            "drop_rate": self._frames_dropped / max(1, self._frames_submitted),
            "input_queue_size": self._input_queue.qsize() if self._input_queue else 0,
            "output_queue_size": self._output_queue.qsize() if self._output_queue else 0,
        }


# -----------------------------------------------------------------------------
# AsyncVideoWriter: Thread-based async video writer to avoid blocking main loop
# -----------------------------------------------------------------------------

class AsyncVideoWriter:
    """
    Async video writer that runs encoding in a separate thread.
    Avoids blocking the main processing loop during frame writes.
    
    Usage:
        writer = AsyncVideoWriter("output.avi", fps=30, size=(640, 480))
        writer.start()
        
        while recording:
            frame = capture_frame()
            writer.write(frame)  # Non-blocking
        
        writer.stop()  # Waits for queue to drain
    """
    
    def __init__(
        self,
        output_path: str,
        fps: float = 30.0,
        size: Tuple[int, int] = (640, 480),
        fourcc: str = "XVID",
        max_queue_size: int = 60,  # ~2 seconds at 30fps
    ):
        self.output_path = output_path
        self.fps = fps
        self.size = size
        self.fourcc = fourcc
        self.max_queue_size = max_queue_size
        
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._writer = None
        self._frames_written = 0
        self._frames_dropped = 0
        self._error: Optional[str] = None
    
    def start(self) -> None:
        """Start the writer thread."""
        if self._thread and self._thread.is_alive():
            return
        
        self._stop_event.clear()
        self._frames_written = 0
        self._frames_dropped = 0
        self._error = None
        
        self._thread = threading.Thread(target=self._write_loop, daemon=True)
        self._thread.start()
    
    def _write_loop(self) -> None:
        """Internal write loop running in separate thread."""
        try:
            if cv2 is None:
                self._error = "OpenCV not available"
                return
            
            fourcc = cv2.VideoWriter_fourcc(*self.fourcc)
            self._writer = cv2.VideoWriter(
                self.output_path, fourcc, self.fps, self.size
            )
            
            if not self._writer.isOpened():
                self._error = f"Failed to open video writer: {self.output_path}"
                return
            
            while not self._stop_event.is_set() or not self._queue.empty():
                try:
                    frame = self._queue.get(timeout=0.1)
                    if frame is None:  # Shutdown signal
                        break
                    
                    # Resize if needed
                    h, w = frame.shape[:2]
                    if (w, h) != self.size:
                        frame = cv2.resize(frame, self.size)
                    
                    self._writer.write(frame)
                    self._frames_written += 1
                except queue.Empty:
                    continue
                except Exception as e:
                    self._error = str(e)
                    
        finally:
            if self._writer:
                self._writer.release()
                self._writer = None
    
    def write(self, frame: np.ndarray) -> bool:
        """
        Write a frame (non-blocking).
        
        Returns True if queued, False if dropped (queue full).
        """
        try:
            self._queue.put_nowait(frame.copy())  # Copy to avoid reference issues
            return True
        except queue.Full:
            self._frames_dropped += 1
            return False
    
    def stop(self, timeout: float = 5.0) -> None:
        """Stop the writer thread, waiting for queue to drain."""
        self._stop_event.set()
        
        # Send shutdown signal
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._thread = None
    
    def is_running(self) -> bool:
        """Check if writer thread is running."""
        return self._thread is not None and self._thread.is_alive()
    
    def stats(self) -> Dict[str, Any]:
        """Get writer statistics."""
        return {
            "is_running": self.is_running(),
            "frames_written": self._frames_written,
            "frames_dropped": self._frames_dropped,
            "queue_size": self._queue.qsize(),
            "error": self._error,
        }
    
    @property
    def error(self) -> Optional[str]:
        """Get last error message, if any."""
        return self._error

# FaceMesh landmark indices for eye geometry
LEFT_EYE_H = (33, 133)
LEFT_EYE_V1 = (159, 145)
LEFT_EYE_V2 = (160, 144)
RIGHT_EYE_H = (263, 362)
RIGHT_EYE_V1 = (386, 374)
RIGHT_EYE_V2 = (385, 380)

# 6-point PnP landmarks: nose, chin, eye corners, mouth corners
PNP_IDXS = [1, 152, 33, 263, 61, 291]
MODEL_POINTS = np.array([
    [0.0,    0.0,    0.0],
    [0.0,  -330.0, -65.0],
    [-225., 170.,  -135.],
    [225.,  170.,  -135.],
    [-150., -150., -125.],
    [150.,  -150., -125.],
], dtype=np.float32)

# Lightweight ReID uses a small set of stable landmarks (translation/scale/rotation normalized).
REID_IDXS = sorted(set([
    1, 152, 33, 133, 263, 362, 61, 291, 4, 94, 324, 199
]))


def _compute_reid_descriptor(pts2d: np.ndarray) -> Optional[np.ndarray]:
    try:
        max_idx = max(REID_IDXS + [LEFT_EYE_H[0], RIGHT_EYE_H[0]])
    except Exception:
        return None
    if pts2d is None or pts2d.shape[0] <= max_idx:
        return None
    left = pts2d[LEFT_EYE_H[0]]
    right = pts2d[RIGHT_EYE_H[0]]
    eye_vec = right - left
    scale = float(np.hypot(eye_vec[0], eye_vec[1]))
    if not np.isfinite(scale) or scale <= 1e-6:
        return None
    center = (left + right) / 2.0
    angle = float(np.arctan2(eye_vec[1], eye_vec[0]))
    cos_a = float(np.cos(-angle))
    sin_a = float(np.sin(-angle))
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    pts = pts2d[REID_IDXS].astype(np.float32)
    pts = (pts - center) / scale
    pts = pts @ R.T
    return pts.reshape(-1)


def _dist(a, b) -> float:
    """Fast Euclidean distance between two 2D points."""
    dx = float(a[0] - b[0])
    dy = float(a[1] - b[1])
    return (dx * dx + dy * dy) ** 0.5


def _ear_from_pts(pts2d: np.ndarray,
                  H: Tuple[int, int],
                  V1: Tuple[int, int],
                  V2: Tuple[int, int]) -> float:
    """Compute single-eye EAR from 6 landmark points.
    
    Uses inline distance calculation to avoid function call overhead.
    """
    # Horizontal distance (eye width)
    h0, h1 = pts2d[H[0]], pts2d[H[1]]
    dx = h0[0] - h1[0]
    dy = h0[1] - h1[1]
    ph = (dx * dx + dy * dy) ** 0.5 + 1e-6
    
    # Vertical distances (eye height at two points)
    v10, v11 = pts2d[V1[0]], pts2d[V1[1]]
    dx1 = v10[0] - v11[0]
    dy1 = v10[1] - v11[1]
    d1 = (dx1 * dx1 + dy1 * dy1) ** 0.5
    
    v20, v21 = pts2d[V2[0]], pts2d[V2[1]]
    dx2 = v20[0] - v21[0]
    dy2 = v20[1] - v21[1]
    d2 = (dx2 * dx2 + dy2 * dy2) ** 0.5
    
    return float((d1 + d2) / (2.0 * ph))


@dataclass
class TrackState:
    id: int
    last_ts: Optional[float] = None
    prev_ts: Optional[float] = None
    center: Tuple[float, float] = (0.0, 0.0)
    prev_center: Tuple[float, float] = (0.0, 0.0)
    miss_count: int = 0
    reid_vec: Optional[np.ndarray] = None
    reid_ts: Optional[float] = None

    ear_ema: Optional[float] = None
    pitch_ema: Optional[float] = None

    seen_secs: float = 0.0
    ear_open_baseline: Optional[float] = None

    eye_closed: bool = False
    closed_timer: float = 0.0
    drowsy_active: bool = False
    drowsy_start_ts: Optional[float] = None

    down: bool = False
    down_timer: float = 0.0
    down_active: bool = False
    down_start_ts: Optional[float] = None

    blink_count: int = 0
    state: str = "awake"  # awake / drowsy / down / drowsy+down


@dataclass
class FaceAnalyzerConfig:
    max_faces: int = 5
    refine_landmarks: bool = False
    min_det_conf: float = 0.5
    min_trk_conf: float = 0.5
    # Model complexity: 0=lite (faster), 1=full (more accurate)
    model_complexity: int = 1
    # Tuning defaults (made more sensitive to reduce missed drowsy detection)
    ear_min: float = 0.18
    ear_ratio: float = 0.85
    calibrate_secs: float = 3.0
    ear_ema_alpha: float = 0.6

    drowsy_secs: float = 1.6
    blink_max_secs: float = 0.25
    recover_secs: float = 0.8

    pitch_down_deg: float = -12.0
    pitch_ema_alpha: float = 0.5
    down_secs: float = 1.0

    match_max_px: float = 80.0
    # allowed extra pixels per second of absence when matching (time-aware slack)
    match_speed_px_per_sec: float = 200.0
    # Cap how much time contributes to the matching slack (prevents huge gates
    # after long occlusions; prefer creating a new track over wrong re-attach).
    match_slack_max_secs: float = 0.8
    # Allow temporary occlusions (e.g. head down / eyes not visible) without
    # killing a track too aggressively; keeps IDs stable within a class.
    max_miss_count: int = 200
    # Hard cap for how long we keep a track without seeing it (seconds).
    # Useful when FPS varies (miss_count is frame-rate dependent).
    max_miss_secs: float = 12.0
    # Optional stabilization against camera motion (pan/tilt).
    # When enabled, we estimate a global translation between the previous tracks
    # and current detections and compensate before matching.
    compensate_camera_shift: bool = True
    camera_shift_max_px: float = 600.0
    # Also compensate rotation/zoom (similarity transform) when enough faces are
    # present. This is a cheap alternative to running a face-recognition model
    # when you only need stable IDs within one session.
    compensate_camera_similarity: bool = True
    camera_rot_max_deg: float = 18.0
    camera_scale_max: float = 1.25
    camera_icp_iters: int = 2
    camera_icp_inlier_px: float = 120.0

    # Lightweight ReID (landmark-based, only when ambiguous)
    reid_enabled: bool = True
    reid_ambiguity_px: float = 30.0
    reid_ambiguity_ratio: float = 1.15
    reid_accept_dist: float = 0.35
    reid_reject_dist: float = 0.55
    reid_bias_px: float = 24.0
    reid_max_age_sec: float = 4.0

    debug_draw: bool = False
    
    # Frame processing (from device profile)
    target_fps: float = 30.0
    process_every_n: int = 1
    input_scale: float = 1.0
    input_max_width: int = 1920
    input_max_height: int = 1080
    
    # Capture queue settings
    use_capture_queue: bool = False
    capture_queue_size: int = 2
    drop_old_frames: bool = True
    async_video_write: bool = False

    @classmethod
    def from_device_profile(cls, profile: "DeviceProfile") -> "FaceAnalyzerConfig":
        """Create FaceAnalyzerConfig from a DeviceProfile."""
        return cls(
            max_faces=profile.max_faces,
            refine_landmarks=profile.refine_landmarks,
            model_complexity=profile.model_complexity,
            min_det_conf=profile.min_det_conf,
            min_trk_conf=profile.min_trk_conf,
            debug_draw=profile.debug_draw,
            target_fps=profile.target_fps,
            process_every_n=profile.process_every_n,
            input_scale=profile.input_scale,
            input_max_width=profile.input_max_width,
            input_max_height=profile.input_max_height,
            use_capture_queue=profile.use_capture_queue,
            capture_queue_size=profile.capture_queue_size,
            drop_old_frames=profile.drop_old_frames,
            async_video_write=profile.async_video_write,
        )

    def apply_device_profile(self, profile: "DeviceProfile") -> None:
        """Apply device profile settings to this config (in-place update)."""
        self.max_faces = profile.max_faces
        self.refine_landmarks = profile.refine_landmarks
        self.model_complexity = profile.model_complexity
        self.min_det_conf = profile.min_det_conf
        self.min_trk_conf = profile.min_trk_conf
        self.target_fps = profile.target_fps
        self.process_every_n = profile.process_every_n
        self.input_scale = profile.input_scale
        self.input_max_width = profile.input_max_width
        self.input_max_height = profile.input_max_height
        self.use_capture_queue = profile.use_capture_queue
        self.capture_queue_size = profile.capture_queue_size
        self.drop_old_frames = profile.drop_old_frames
        self.async_video_write = profile.async_video_write


class FaceAnalyzer:
    """Analyze frames for EAR/pitch and emit state-change events."""

    def __init__(self, cfg: FaceAnalyzerConfig = FaceAnalyzerConfig()):
        self.cfg = cfg
        self._mesh = None
        self._frame_count = 0  # For process_every_n
        self._last_results = []  # Cache last results for skipped frames
        self._last_events = []
        
        if mp is not None:
            try:
                self._mp = mp.solutions.face_mesh
                # Use model_complexity for performance tuning (0=lite, 1=full)
                self._mesh = self._mp.FaceMesh(
                    max_num_faces=cfg.max_faces,
                    refine_landmarks=cfg.refine_landmarks,
                    min_detection_confidence=cfg.min_det_conf,
                    min_tracking_confidence=cfg.min_trk_conf,
                )
            except Exception:
                self._mesh = None
        self.tracks: Dict[int, TrackState] = {}
        self._next_id: int = 0
        
        # Pre-allocated buffers for constrained devices
        self._scaled_frame = None
        self._pnp_image_points = np.zeros((6, 2), dtype=np.float32)  # For _compute_pitch
        self._camera_matrix = None  # Cached camera matrix
        self._dist_coeffs = np.zeros(5, dtype=np.float32)  # Cached distortion coeffs

    def analyze_frame(self, frame_bgr: np.ndarray, timestamp: float):
        if cv2 is None or self._mesh is None:
            return [], []
        
        # Frame skipping for constrained devices
        self._frame_count += 1
        if self.cfg.process_every_n > 1 and (self._frame_count % self.cfg.process_every_n) != 0:
            # Return cached results for skipped frames
            refresh_results_timestamps(self._last_results, timestamp)
            return self._last_results, []
        
        H_orig, W_orig = frame_bgr.shape[:2]
        
        # Input scaling for constrained devices
        scale = self.cfg.input_scale
        max_w, max_h = self.cfg.input_max_width, self.cfg.input_max_height
        
        # Apply max resolution cap
        if W_orig > max_w or H_orig > max_h:
            cap_scale = min(max_w / W_orig, max_h / H_orig)
            scale = min(scale, cap_scale)
        
        if scale < 1.0:
            new_w = int(W_orig * scale)
            new_h = int(H_orig * scale)
            # Use pre-allocated buffer if possible
            if self._scaled_frame is None or self._scaled_frame.shape[:2] != (new_h, new_w):
                self._scaled_frame = np.empty((new_h, new_w, 3), dtype=np.uint8)
            cv2.resize(frame_bgr, (new_w, new_h), dst=self._scaled_frame, interpolation=cv2.INTER_LINEAR)
            frame_proc = self._scaled_frame
            H, W = new_h, new_w
            coord_scale = 1.0 / scale  # Scale coordinates back to original
        else:
            frame_proc = frame_bgr
            H, W = H_orig, W_orig
            coord_scale = 1.0
        
        rgb = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2RGB)
        res = self._mesh.process(rgb)

        faces_pts: List[np.ndarray] = []
        centers: List[Tuple[float, float]] = []
        if res.multi_face_landmarks:
            for fl in res.multi_face_landmarks[: self.cfg.max_faces]:
                n = len(fl.landmark)
                pts2d = np.zeros((n, 2), dtype=np.float32)
                # Vectorized landmark extraction - avoid Python list append
                for i, lm in enumerate(fl.landmark):
                    pts2d[i, 0] = lm.x * W
                    pts2d[i, 1] = lm.y * H
                faces_pts.append(pts2d)
                # Compute center directly from numpy array (vectorized mean)
                centers.append((float(pts2d[:, 0].mean()), float(pts2d[:, 1].mean())))

        assign, reid_desc = self._associate(centers, timestamp, faces_pts)

        results: List[Dict] = []
        events: List[Dict] = []
        used_ids = set()

        for idx, pts2d in enumerate(faces_pts):
            center = centers[idx]
            tid = assign.get(idx)
            if tid is None:
                tid = self._next_id
                self._next_id += 1
                self.tracks[tid] = TrackState(id=tid, center=center, prev_center=center, last_ts=timestamp)
            st = self.tracks[tid]
            used_ids.add(tid)
            dt = 0.0 if st.last_ts is None else max(0.0, timestamp - st.last_ts)
            # remember previous center before updating (useful for prediction/heuristics)
            st.prev_center = st.center
            st.prev_ts = st.last_ts
            st.last_ts = timestamp
            st.center = center
            st.miss_count = 0
            if idx in reid_desc:
                st.reid_vec = reid_desc[idx]
                st.reid_ts = timestamp

            # EAR
            left_ear = _ear_from_pts(pts2d, LEFT_EYE_H, LEFT_EYE_V1, LEFT_EYE_V2)
            right_ear = _ear_from_pts(pts2d, RIGHT_EYE_H, RIGHT_EYE_V1, RIGHT_EYE_V2)
            ear = (left_ear + right_ear) / 2.0
            st.ear_ema = ear if st.ear_ema is None else (
                self.cfg.ear_ema_alpha * ear + (1 - self.cfg.ear_ema_alpha) * st.ear_ema
            )

            # Calibration of open-eye baseline
            if st.seen_secs < self.cfg.calibrate_secs:
                st.seen_secs += dt
                if st.ear_open_baseline is None:
                    st.ear_open_baseline = st.ear_ema
                else:
                    st.ear_open_baseline = 0.9 * st.ear_open_baseline + 0.1 * st.ear_ema

            ear_base = st.ear_open_baseline if st.ear_open_baseline else 0.28
            ear_thresh = max(self.cfg.ear_min, self.cfg.ear_ratio * ear_base)

            # Pitch
            pitch = self._compute_pitch(pts2d, W, H)
            if pitch is not None:
                alpha = getattr(self.cfg, 'pitch_ema_alpha', 0.3)
                st.pitch_ema = pitch if st.pitch_ema is None else (alpha * pitch + (1 - alpha) * st.pitch_ema)

            # Eye/drowsy FSM
            was_closed = st.eye_closed
            st.eye_closed = (st.ear_ema is not None) and (st.ear_ema < ear_thresh)

            if st.eye_closed:
                st.closed_timer += dt
                if (not st.drowsy_active) and (st.closed_timer >= self.cfg.drowsy_secs):
                    st.drowsy_active = True
                    st.drowsy_start_ts = timestamp - st.closed_timer
                    events.append({
                        "ts": st.drowsy_start_ts,
                        "student_id": tid,
                        "type": "DROWSY_START",
                        "dur": None,
                        "ear": float(st.ear_ema),
                        "pitch": float(st.pitch_ema) if st.pitch_ema is not None else None,
                    })
            else:
                if was_closed:
                    if 0.0 < st.closed_timer < self.cfg.blink_max_secs:
                        st.blink_count += 1
                        events.append({
                            "ts": timestamp,
                            "student_id": tid,
                            "type": "BLINK",
                            "dur": float(st.closed_timer),
                            "ear": float(st.ear_ema),
                            "pitch": float(st.pitch_ema) if st.pitch_ema is not None else None,
                        })
                    if st.drowsy_active:
                        events.append({
                            "ts": timestamp,
                            "student_id": tid,
                            "type": "DROWSY_END",
                            "dur": float(timestamp - (st.drowsy_start_ts or timestamp)),
                            "ear": float(st.ear_ema),
                            "pitch": float(st.pitch_ema) if st.pitch_ema is not None else None,
                        })
                    st.drowsy_active = False
                    st.drowsy_start_ts = None
                st.closed_timer = 0.0

            # Looking down FSM (pitch)
            was_down = st.down
            is_down_now = (st.pitch_ema is not None) and (st.pitch_ema <= self.cfg.pitch_down_deg)
            st.down = is_down_now
            if st.down:
                st.down_timer += dt
                if (not st.down_active) and (st.down_timer >= self.cfg.down_secs):
                    st.down_active = True
                    st.down_start_ts = timestamp - st.down_timer
                    events.append({
                        "ts": st.down_start_ts,
                        "student_id": tid,
                        "type": "LOOKING_DOWN_START",
                        "dur": None,
                        "ear": float(st.ear_ema),
                        "pitch": float(st.pitch_ema) if st.pitch_ema is not None else None,
                    })
            else:
                if was_down and st.down_active:
                    events.append({
                        "ts": timestamp,
                        "student_id": tid,
                        "type": "LOOKING_DOWN_END",
                        "dur": float(timestamp - (st.down_start_ts or timestamp)),
                        "ear": float(st.ear_ema),
                        "pitch": float(st.pitch_ema) if st.pitch_ema is not None else None,
                    })
                st.down_active = False
                st.down_start_ts = None
                st.down_timer = 0.0

            state = "awake"
            if st.drowsy_active and st.down_active:
                state = "drowsy+down"
            elif st.drowsy_active:
                state = "drowsy"
            elif st.down_active:
                state = "down"
            st.state = state
            # Use vectorized min/max for bounding box (avoids multiple calls)
            if pts2d.size:
                x_coords = pts2d[:, 0]
                y_coords = pts2d[:, 1]
                x1 = float(x_coords.min())
                y1 = float(y_coords.min())
                x2 = float(x_coords.max())
                y2 = float(y_coords.max())
            else:
                x1 = y1 = x2 = y2 = 0.0
            w_px = max(1.0, x2 - x1)
            h_px = max(1.0, y2 - y1)
            bbox_norm = [
                float(max(0.0, min(1.0, x1 / max(1.0, W)))),
                float(max(0.0, min(1.0, y1 / max(1.0, H)))),
                float(max(0.0, min(1.0, w_px / max(1.0, W)))),
                float(max(0.0, min(1.0, h_px / max(1.0, H)))),
            ]
            cx_norm = float(center[0] / max(1.0, W))
            cy_norm = float(center[1] / max(1.0, H))
            area = float(bbox_norm[2] * bbox_norm[3])
            results.append({
                "student_id": tid,
                "track_id": tid,
                "ear": float(st.ear_ema) if st.ear_ema is not None else None,
                "pitch": float(st.pitch_ema) if st.pitch_ema is not None else None,
                "state": st.state,
                "ear_thresh": float(ear_thresh),
                "blink_count": st.blink_count,
                "bbox": bbox_norm,
                "center": [cx_norm, cy_norm],
                "center_x": cx_norm,
                "center_y": cy_norm,
                "area": area,
                "ts": float(timestamp),
            })

        # Cleanup missing tracks
        for tid, st in list(self.tracks.items()):
            if tid not in used_ids:
                st.miss_count += 1
                too_old = False
                if st.last_ts is not None:
                    try:
                        too_old = (timestamp - float(st.last_ts)) > float(getattr(self.cfg, "max_miss_secs", 3.0))
                    except Exception:
                        too_old = False
                if st.miss_count > self.cfg.max_miss_count or too_old:
                    if st.drowsy_active:
                        events.append({
                            "ts": st.last_ts if st.last_ts else 0.0,
                            "student_id": tid,
                            "type": "DROWSY_END",
                            "dur": None,
                        })
                    if st.down_active:
                        events.append({
                            "ts": st.last_ts if st.last_ts else 0.0,
                            "student_id": tid,
                            "type": "LOOKING_DOWN_END",
                            "dur": None,
                        })
                    del self.tracks[tid]

        if self.cfg.debug_draw:
            self._draw_debug(frame_bgr, results, faces_pts)

        # Cache results for frame skipping
        self._last_results = results
        self._last_events = events
        
        return results, events

    def _compute_pitch(self, pts2d: np.ndarray, W: int, H: int) -> Optional[float]:
        if cv2 is None:
            return None
        try:
            # Use pre-allocated buffer for image points
            for i, idx in enumerate(PNP_IDXS):
                self._pnp_image_points[i] = pts2d[idx]
            
            # Cache/update camera matrix only when resolution changes
            f = 1.2 * W
            if self._camera_matrix is None or self._camera_matrix[0, 0] != f:
                self._camera_matrix = np.array(
                    [[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1]], dtype=np.float32
                )
            
            ok, rvec, tvec = cv2.solvePnP(
                MODEL_POINTS, self._pnp_image_points, self._camera_matrix,
                self._dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not ok:
                return None
            R, _ = cv2.Rodrigues(rvec)
            sy = (R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0]) ** 0.5
            pitch = np.degrees(np.arctan2(-R[2, 0], sy))  # negative is looking down
            return float(pitch)
        except Exception:
            return None

    def _associate(
        self,
        centers: List[Tuple[float, float]],
        timestamp: float,
        faces_pts: Optional[List[np.ndarray]] = None,
    ) -> Tuple[Dict[int, int], Dict[int, np.ndarray]]:
        """Assign detections to existing tracks.

        This matcher is designed to be robust to:
        - Normal subject motion (constant-velocity prediction).
        - Small/medium camera pan/tilt (optional global translation compensation).
        - Brief occlusions (time-aware slack in gating threshold).

        NOTE: We intentionally keep this dependency-free (no SciPy/Hungarian),
        because `max_faces` is small and this runs on-device.
        """
        assign: Dict[int, int] = {}
        if not centers:
            return assign, {}
        track_ids = list(self.tracks.keys())
        if not track_ids:
            for i in range(len(centers)):
                assign[i] = None
            return assign, {}

        def _predict_center(st: TrackState) -> Tuple[float, float]:
            """Constant-velocity prediction from (prev_center, center) with timestamps."""
            if st.last_ts is None:
                return st.center
            if st.prev_ts is None or st.prev_ts >= st.last_ts:
                return st.center
            dt_hist = float(st.last_ts - st.prev_ts)
            if dt_hist <= 1e-6:
                return st.center
            vx = (st.center[0] - st.prev_center[0]) / dt_hist
            vy = (st.center[1] - st.prev_center[1]) / dt_hist
            dt_fwd = max(0.0, float(timestamp - st.last_ts))
            dt_fwd = min(dt_fwd, float(getattr(self.cfg, "match_slack_max_secs", 0.8)))
            return (st.center[0] + vx * dt_fwd, st.center[1] + vy * dt_fwd)

        pred = {tid: _predict_center(self.tracks[tid]) for tid in track_ids}

        def _estimate_similarity_umeyama(src: np.ndarray, dst: np.ndarray):
            """Estimate 2D similarity transform dst ~= s * R * src + t.

            Returns (s, R(2x2), t(2,)) or None.
            """
            if src.shape[0] < 2 or dst.shape[0] < 2:
                return None
            src = src.astype(np.float64)
            dst = dst.astype(np.float64)
            mu_src = np.mean(src, axis=0)
            mu_dst = np.mean(dst, axis=0)
            src_c = src - mu_src
            dst_c = dst - mu_dst
            var_src = float(np.mean(np.sum(src_c ** 2, axis=1)))
            if var_src <= 1e-9:
                return None
            cov = (dst_c.T @ src_c) / float(src.shape[0])
            try:
                U, S, Vt = np.linalg.svd(cov)
            except Exception:
                return None
            R = U @ Vt
            if np.linalg.det(R) < 0:
                U[:, -1] *= -1
                R = U @ Vt
            scale = float(np.sum(S) / var_src)
            if not np.isfinite(scale) or scale <= 0:
                return None
            t = mu_dst - scale * (R @ mu_src)
            return scale, R, t

        def _apply_similarity_to_point(p: Tuple[float, float], scale: float, R: np.ndarray, t: np.ndarray) -> Tuple[float, float]:
            x = np.array([float(p[0]), float(p[1])], dtype=np.float64)
            y = scale * (R @ x) + t
            return float(y[0]), float(y[1])

        def _try_camera_similarity_compensation() -> None:
            """Mutate `pred` in-place by applying a global similarity transform.

            Uses an ICP-like 1-2 iteration closest matching between detections and
            predicted tracks, then estimates a similarity transform (rotation+scale+translation).
            """
            if not getattr(self.cfg, "compensate_camera_similarity", True):
                return
            if len(track_ids) < 2 or len(centers) < 2:
                return

            # Start from current predicted positions
            pred_pts = np.array([pred[tid] for tid in track_ids], dtype=np.float64)  # (T,2)
            det_pts = np.array(centers, dtype=np.float64)  # (D,2)

            # Iteratively re-match and re-estimate transform.
            scale_acc = 1.0
            R_acc = np.eye(2, dtype=np.float64)
            t_acc = np.zeros(2, dtype=np.float64)

            iters = int(getattr(self.cfg, "camera_icp_iters", 2))
            inlier_px = float(getattr(self.cfg, "camera_icp_inlier_px", 120.0))
            max_scale = float(getattr(self.cfg, "camera_scale_max", 1.25))
            max_rot = float(getattr(self.cfg, "camera_rot_max_deg", 18.0))

            for _ in range(max(0, iters)):
                # transformed predicted points
                pred_t = (scale_acc * (pred_pts @ R_acc.T)) + t_acc  # (T,2)

                # Greedy unique matching (closest pairs) for correspondence proposal.
                candidates = []
                for di in range(det_pts.shape[0]):
                    for ti in range(pred_t.shape[0]):
                        d = float(np.hypot(det_pts[di, 0] - pred_t[ti, 0], det_pts[di, 1] - pred_t[ti, 1]))
                        candidates.append((d, di, ti))
                candidates.sort(key=lambda x: x[0])
                used_d = set()
                used_t = set()
                pairs = []
                for d, di, ti in candidates:
                    if di in used_d or ti in used_t:
                        continue
                    used_d.add(di)
                    used_t.add(ti)
                    pairs.append((di, ti, d))
                    if len(used_d) >= min(det_pts.shape[0], pred_t.shape[0]):
                        break

                if len(pairs) < 2:
                    return

                src = np.array([pred_t[ti] for (_, ti, _) in pairs], dtype=np.float64)
                dst = np.array([det_pts[di] for (di, _, _) in pairs], dtype=np.float64)

                est = _estimate_similarity_umeyama(src, dst)
                if not est:
                    return
                s, R, t = est

                # Quick sanity checks (per-step bounds).
                angle = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
                if abs(angle) > max_rot:
                    return
                if s < (1.0 / max_scale) or s > max_scale:
                    return

                # Inlier refinement
                pred_m = (s * (src @ R.T)) + t
                residuals = np.hypot(pred_m[:, 0] - dst[:, 0], pred_m[:, 1] - dst[:, 1])
                inliers = residuals <= inlier_px
                if np.sum(inliers) >= 2 and np.sum(inliers) < len(pairs):
                    est2 = _estimate_similarity_umeyama(src[inliers], dst[inliers])
                    if est2:
                        s, R, t = est2
                        angle = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
                        if abs(angle) > max_rot or s < (1.0 / max_scale) or s > max_scale:
                            return
                        pred_m = (s * (src[inliers] @ R.T)) + t
                        src_use = src[inliers]
                    else:
                        src_use = src
                else:
                    src_use = src

                # Reject wildly large motion (typically caused by wrong correspondences).
                max_shift = float(getattr(self.cfg, "camera_shift_max_px", 600.0))
                disp = np.hypot(pred_m[:, 0] - src_use[:, 0], pred_m[:, 1] - src_use[:, 1])
                if disp.size:
                    med_disp = float(np.median(disp))
                    if med_disp > max_shift:
                        return

                # Compose delta into accumulator: new = (s*R)*old + t
                scale_acc = float(s) * scale_acc
                R_acc = R @ R_acc
                t_acc = (float(s) * (R @ t_acc)) + t
                # Overall per-frame bounds
                if scale_acc < (1.0 / max_scale) or scale_acc > max_scale:
                    return
                ang_acc = float(np.degrees(np.arctan2(R_acc[1, 0], R_acc[0, 0])))
                if abs(ang_acc) > max_rot:
                    return

            # Apply accumulated transform to all predicted track positions.
            for tid in track_ids:
                pred[tid] = _apply_similarity_to_point(pred[tid], scale_acc, R_acc, t_acc)

        # Optional: compensate a global camera translation or similarity motion.
        #
        # - Multi-face case (>=2): estimate shift by robust median of
        #   nearest-neighbor deltas.
        # - Single-face case (1 track, 1 detection): allow shift only when the
        #   track was continuously present (miss_count==0), otherwise it's too
        #   ambiguous and may cause wrong re-attachment.
        if getattr(self.cfg, "compensate_camera_shift", True) and len(track_ids) == 1 and len(centers) == 1:
            tid = track_ids[0]
            st = self.tracks[tid]
            if getattr(st, "miss_count", 0) == 0:
                c = centers[0]
                pc = pred[tid]
                dx = float(c[0] - pc[0])
                dy = float(c[1] - pc[1])
                shift = float(np.hypot(dx, dy))
                max_shift = float(getattr(self.cfg, "camera_shift_max_px", 600.0))
                if shift > max_shift and shift > 1e-6:
                    scale = max_shift / shift
                    dx *= scale
                    dy *= scale
                pred[tid] = (pc[0] + dx, pc[1] + dy)
        elif getattr(self.cfg, "compensate_camera_shift", True) and len(track_ids) >= 2 and len(centers) >= 2:
            # First do a cheap translation estimate (helps ICP initialize well).
            deltas = []
            for c in centers:
                best_tid, best_d = None, 1e18
                for tid in track_ids:
                    pc = pred[tid]
                    d = float(np.hypot(c[0] - pc[0], c[1] - pc[1]))
                    if d < best_d:
                        best_d, best_tid = d, tid
                if best_tid is not None:
                    pc = pred[best_tid]
                    deltas.append((c[0] - pc[0], c[1] - pc[1]))
            if deltas:
                dx = float(np.median([d[0] for d in deltas]))
                dy = float(np.median([d[1] for d in deltas]))
                shift = float(np.hypot(dx, dy))
                max_shift = float(getattr(self.cfg, "camera_shift_max_px", 600.0))
                if shift > max_shift and shift > 1e-6:
                    scl = max_shift / shift
                    dx *= scl
                    dy *= scl
                for tid in track_ids:
                    pc = pred[tid]
                    pred[tid] = (pc[0] + dx, pc[1] + dy)

            # Then attempt similarity compensation (rotation + zoom + translation).
            _try_camera_similarity_compensation()

        # Build candidate pairs within geometric threshold.
        track_thresh: Dict[int, float] = {}
        for tid in track_ids:
            st = self.tracks[tid]
            dt = max(0.0, float(timestamp - (st.last_ts or timestamp)))
            dt_cap = float(getattr(self.cfg, "match_slack_max_secs", 0.8))
            slack = min(dt, dt_cap) * float(getattr(self.cfg, "match_speed_px_per_sec", 0.0))
            track_thresh[tid] = float(self.cfg.match_max_px) + slack

        det_cands: Dict[int, List[Tuple[float, int]]] = {i: [] for i in range(len(centers))}
        track_cands: Dict[int, List[Tuple[float, int]]] = {tid: [] for tid in track_ids}
        for i, c in enumerate(centers):
            for tid in track_ids:
                pc = pred[tid]
                d = float(np.hypot(c[0] - pc[0], c[1] - pc[1]))
                if d <= track_thresh.get(tid, float(self.cfg.match_max_px)):
                    det_cands[i].append((d, tid))
                    track_cands[tid].append((d, i))

        def _is_ambiguous(cands: List[Tuple[float, int]]) -> bool:
            if len(cands) < 2:
                return False
            cands = sorted(cands, key=lambda x: x[0])
            d1 = float(cands[0][0])
            d2 = float(cands[1][0])
            if (d2 - d1) <= float(getattr(self.cfg, "reid_ambiguity_px", 30.0)):
                return True
            if d1 > 1e-6 and (d2 / d1) <= float(getattr(self.cfg, "reid_ambiguity_ratio", 1.15)):
                return True
            return False

        ambiguous_dets = set()
        for i, cands in det_cands.items():
            if _is_ambiguous(cands):
                ambiguous_dets.add(i)
        for tid, cands in track_cands.items():
            if _is_ambiguous(cands):
                for _, i in cands:
                    ambiguous_dets.add(i)

        reid_desc: Dict[int, np.ndarray] = {}
        reid_dists: Dict[Tuple[int, int], float] = {}
        if (
            bool(getattr(self.cfg, "reid_enabled", True))
            and ambiguous_dets
            and faces_pts is not None
        ):
            for i in ambiguous_dets:
                if i >= len(faces_pts):
                    continue
                desc = _compute_reid_descriptor(faces_pts[i])
                if desc is not None:
                    reid_desc[i] = desc
            if reid_desc:
                max_age = float(getattr(self.cfg, "reid_max_age_sec", 4.0))
                track_desc: Dict[int, np.ndarray] = {}
                for tid in track_ids:
                    st = self.tracks[tid]
                    if st.reid_vec is None or st.reid_ts is None:
                        continue
                    age = float(timestamp - st.reid_ts)
                    if max_age > 0 and age > max_age:
                        continue
                    track_desc[tid] = st.reid_vec
                for i, cands in det_cands.items():
                    if i not in ambiguous_dets:
                        continue
                    desc = reid_desc.get(i)
                    if desc is None:
                        continue
                    for d, tid in cands:
                        tdesc = track_desc.get(tid)
                        if tdesc is None:
                            continue
                        try:
                            dist = float(np.linalg.norm(desc - tdesc))
                        except Exception:
                            continue
                        reid_dists[(i, tid)] = dist

        strong_reid: Dict[int, bool] = {}
        if reid_dists:
            accept_dist = float(getattr(self.cfg, "reid_accept_dist", 0.35))
            for i in ambiguous_dets:
                vals = [v for (di, _), v in reid_dists.items() if di == i]
                if vals and min(vals) <= accept_dist:
                    strong_reid[i] = True

        # Build all candidate pairs and do a global greedy assignment by distance.
        candidates: List[Tuple[float, float, int, int]] = []  # (eff_d, raw_d, det_i, tid)
        bias_px = float(getattr(self.cfg, "reid_bias_px", 24.0))
        reject_dist = float(getattr(self.cfg, "reid_reject_dist", 0.55))
        for i, cands in det_cands.items():
            for d, tid in cands:
                if i in strong_reid:
                    dist = reid_dists.get((i, tid))
                    if dist is None or dist > reject_dist:
                        continue
                eff_d = d
                dist = reid_dists.get((i, tid))
                if dist is not None:
                    eff_d = d + (bias_px * dist)
                candidates.append((eff_d, d, i, tid))
        candidates.sort(key=lambda x: x[0])

        assigned_dets = set()
        assigned_tracks = set()
        for _, _, i, tid in candidates:
            if i in assigned_dets or tid in assigned_tracks:
                continue
            assign[i] = tid
            assigned_dets.add(i)
            assigned_tracks.add(tid)

        for i in range(len(centers)):
            if i not in assign:
                assign[i] = None

        return assign, reid_desc

    def _draw_debug(self, frame_bgr: np.ndarray, results: List[Dict], faces_pts: List[np.ndarray]):
        """Draw landmarks and state text on the frame when debug_draw=True."""
        for i, r in enumerate(results):
            pts2d = faces_pts[i] if i < len(faces_pts) else None
            if pts2d is not None and pts2d.size != 0:
                for (x, y) in pts2d.astype(int):
                    cv2.circle(frame_bgr, (int(x), int(y)), 1, (0, 255, 255), -1)

                eye_idxs = [
                    LEFT_EYE_H[0], LEFT_EYE_H[1],
                    LEFT_EYE_V1[0], LEFT_EYE_V1[1],
                    LEFT_EYE_V2[0], LEFT_EYE_V2[1],
                    RIGHT_EYE_H[0], RIGHT_EYE_H[1],
                    RIGHT_EYE_V1[0], RIGHT_EYE_V1[1],
                    RIGHT_EYE_V2[0], RIGHT_EYE_V2[1],
                ]
                for idx in set(eye_idxs):
                    if idx < pts2d.shape[0]:
                        x, y = pts2d[idx].astype(int)
                        cv2.circle(frame_bgr, (int(x), int(y)), 3, (0, 0, 255), -1)

                for idx in PNP_IDXS:
                    if idx < pts2d.shape[0]:
                        x, y = pts2d[idx].astype(int)
                        cv2.circle(frame_bgr, (int(x), int(y)), 4, (255, 0, 0), 2)

                cx, cy = int(np.mean(pts2d[:, 0])), int(np.mean(pts2d[:, 1]))
                txt = f"ID{r['student_id']} {r['state']} EAR={r['ear']:.2f}"
                color = (0, 0, 255) if ("drowsy" in r['state'] or "down" in r['state']) else (0, 255, 0)
                cv2.putText(frame_bgr, txt, (cx - 80, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.circle(frame_bgr, (cx, cy), 12, color, 2)
            else:
                y = 30 + i * 24
                txt = f"ID{r['student_id']} {r['state']} EAR={r['ear']:.2f}"
                color = (0, 0, 255) if ("drowsy" in r['state'] or "down" in r['state']) else (0, 255, 0)
                cv2.putText(frame_bgr, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def list_cameras(max_index: int = 8, backend: Optional[int] = None, timeout: float = 1.0) -> List[Dict]:
    """Probe camera indices and return those that produce frames."""
    found: List[Dict] = []
    for idx in range(max_index):
        try:
            cap = cv2.VideoCapture(idx) if backend is None else cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                cap.release()
                continue

            t0 = time.time()
            ok = False
            while time.time() - t0 < timeout:
                ret, frame = cap.read()
                if ret and frame is not None:
                    h, w = frame.shape[:2]
                    found.append({"index": idx, "width": int(w), "height": int(h)})
                    ok = True
                    break
                time.sleep(0.05)

            cap.release()
            if not ok:
                continue
        except Exception:
            try:
                cap.release()
            except Exception:
                pass
            continue
    return found


class _Poster:
    """Background poster that batches JSON events and sends them to an HTTP endpoint."""
    def __init__(self, url: Optional[str], batch_interval: float = 0.1, max_batch: int = 200):
        self.url = url
        self._q = queue.Queue()
        self._thr: Optional[threading.Thread] = None
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
                continue

            while len(batch) < self.max_batch:
                try:
                    ev = self._q.get_nowait()
                    batch.append(ev)
                except queue.Empty:
                    break

            payload = {"type": "batch", "events": batch}
            try:
                if requests is not None:
                    requests.post(self.url, json=payload, timeout=3)
            except Exception:
                pass

    def stop(self):
        self._stop.set()
        if self._thr:
            self._thr.join(timeout=1.0)


def open_camera(index: int = 0, width: int = 640, height: int = 480, backend: Optional[int] = None) -> cv2.VideoCapture:
    """Open and configure a camera, returning the cv2.VideoCapture instance."""
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is not available.")
    cap = cv2.VideoCapture(index) if backend is None else cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        return cap
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


if __name__ == "__main__":
    import argparse
    import platform

    ap = argparse.ArgumentParser()
    ap.add_argument("--webcam", type=int, default=0, help="Webcam index (default 0)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--show", action="store_true", help="Show debug window")
    ap.add_argument("--list-cams", action="store_true", help="List available cameras and exit")
    ap.add_argument("--push-url", type=str, default=None, help="Optional HTTP push URL for events")
    args = ap.parse_args()

    cfg = FaceAnalyzerConfig(debug_draw=args.show)
    analyzer = FaceAnalyzer(cfg)

    current_os = platform.system()
    if current_os == "Windows":
        backend = cv2.CAP_DSHOW
    elif current_os == "Linux":
        backend = cv2.CAP_V4L2
    else:
        backend = cv2.CAP_ANY

    if args.list_cams:
        cams = list_cameras(max_index=12, backend=backend, timeout=0.8)
        if not cams:
            print("No cameras found (tried indices 0-11)")
        else:
            print("Found cameras:")
            for c in cams:
                print(f"  index={c['index']}, resolution={c['width']}x{c['height']}")
        raise SystemExit(0)

    print(f"Starting camera on {current_os} (backend={backend}) ...")
    cap = open_camera(index=args.webcam, width=args.width, height=args.height, backend=backend)

    if not cap.isOpened():
        print(f"Could not open camera index={args.webcam}. Check connections or try --webcam 1")
        raise SystemExit(1)

    poster = _Poster(args.push_url) if args.push_url else None
    if poster:
        poster.start()

    t0 = time.time()
    frames = 0
    start_wall = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read video frame, exiting...")
                break

            ts = time.time() - start_wall
            results, events = analyzer.analyze_frame(frame, ts)
            frames += 1

            for e in events:
                print(e)
                if poster:
                    poster.send(e)

            if args.show:
                cv2.imshow("FaceAnalyzer", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if poster:
            poster.stop()
        dur = max(1e-6, time.time() - t0)
        print(f"Frames={frames}, FPS={frames/dur:.2f}")
