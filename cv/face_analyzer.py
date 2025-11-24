# cv/face_analyzer.py
# -*- coding: utf-8 -*-
"""Face analysis: EAR-based drowsiness, pitch-based looking-down detection, simple multi-face tracking."""
import time
import threading
import queue
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import requests

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


def _dist(a, b) -> float:
    return float(np.linalg.norm(a - b))


def _ear_from_pts(pts2d: np.ndarray,
                  H: Tuple[int, int],
                  V1: Tuple[int, int],
                  V2: Tuple[int, int]) -> float:
    """Compute single-eye EAR from 6 landmark points."""
    ph = _dist(pts2d[H[0]], pts2d[H[1]]) + 1e-6
    pv = _dist(pts2d[V1[0]], pts2d[V1[1]]) + _dist(pts2d[V2[0]], pts2d[V2[1]])
    return float(pv / (2.0 * ph))


@dataclass
class TrackState:
    id: int
    last_ts: Optional[float] = None
    center: Tuple[float, float] = (0.0, 0.0)
    prev_center: Tuple[float, float] = (0.0, 0.0)
    miss_count: int = 0

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
    max_miss_count: int = 10

    debug_draw: bool = False


class FaceAnalyzer:
    """Analyze frames for EAR/pitch and emit state-change events."""

    def __init__(self, cfg: FaceAnalyzerConfig = FaceAnalyzerConfig()):
        self.cfg = cfg
        self._mp = mp.solutions.face_mesh
        self._mesh = self._mp.FaceMesh(
            max_num_faces=cfg.max_faces,
            refine_landmarks=cfg.refine_landmarks,
            min_detection_confidence=cfg.min_det_conf,
            min_tracking_confidence=cfg.min_trk_conf,
        )
        self.tracks: Dict[int, TrackState] = {}
        self._next_id: int = 0

    def analyze_frame(self, frame_bgr: np.ndarray, timestamp: float):
        H, W = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self._mesh.process(rgb)

        faces_pts: List[np.ndarray] = []
        centers: List[Tuple[float, float]] = []
        if res.multi_face_landmarks:
            for fl in res.multi_face_landmarks[: self.cfg.max_faces]:
                n = len(fl.landmark)
                pts2d = np.zeros((n, 2), dtype=np.float32)
                xs, ys = [], []
                for i, lm in enumerate(fl.landmark):
                    x, y = lm.x * W, lm.y * H
                    pts2d[i] = (x, y)
                    xs.append(x)
                    ys.append(y)
                faces_pts.append(pts2d)
                centers.append((float(np.mean(xs)), float(np.mean(ys))))

        assign = self._associate(centers, timestamp)

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
            st.last_ts = timestamp
            st.center = center
            st.miss_count = 0

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
            x1 = float(np.min(pts2d[:, 0])) if pts2d.size else 0.0
            y1 = float(np.min(pts2d[:, 1])) if pts2d.size else 0.0
            x2 = float(np.max(pts2d[:, 0])) if pts2d.size else 0.0
            y2 = float(np.max(pts2d[:, 1])) if pts2d.size else 0.0
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
                if st.miss_count > self.cfg.max_miss_count:
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

        return results, events

    def _compute_pitch(self, pts2d: np.ndarray, W: int, H: int) -> Optional[float]:
        try:
            image_points = np.array([pts2d[i] for i in PNP_IDXS], dtype=np.float32)
            f = 1.2 * W
            K = np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1]], dtype=np.float32)
            dist = np.zeros(5, dtype=np.float32)
            ok, rvec, tvec = cv2.solvePnP(MODEL_POINTS, image_points, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                return None
            R, _ = cv2.Rodrigues(rvec)
            sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
            pitch = np.degrees(np.arctan2(-R[2, 0], sy))  # negative is looking down
            return float(pitch)
        except Exception:
            return None

    def _associate(self, centers: List[Tuple[float, float]], timestamp: float) -> Dict[int, int]:
        """Greedy nearest-neighbor face-to-track assignment with time-aware tolerance.

        If a track was last seen some time ago, allow a larger matching radius
        proportional to the elapsed time (pixels/sec configured by
        `match_speed_px_per_sec`). This reduces ID churn when subjects move
        quickly or are briefly occluded.
        """
        assign: Dict[int, int] = {}
        if not centers:
            return assign
        unused_tracks = set(self.tracks.keys())
        for i, c in enumerate(centers):
            best_tid, best_d = None, 1e9
            for tid in list(unused_tracks):
                st = self.tracks[tid]
                d = np.hypot(c[0] - st.center[0], c[1] - st.center[1])
                if d < best_d:
                    best_d, best_tid = d, tid

            if best_tid is not None:
                st = self.tracks[best_tid]
                # time since last seen; if None, treat as just-seen (dt=0)
                dt = max(0.0, timestamp - (st.last_ts or timestamp))
                # allow extra slack proportional to time since last seen
                slack = dt * float(getattr(self.cfg, 'match_speed_px_per_sec', 0.0))
                threshold = float(self.cfg.match_max_px) + slack
                if best_d <= threshold:
                    assign[i] = best_tid
                    unused_tracks.remove(best_tid)
                else:
                    assign[i] = None
            else:
                assign[i] = None

        return assign

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
                requests.post(self.url, json=payload, timeout=3)
            except Exception:
                pass

    def stop(self):
        self._stop.set()
        if self._thr:
            self._thr.join(timeout=1.0)


def open_camera(index: int = 0, width: int = 640, height: int = 480, backend: Optional[int] = None) -> cv2.VideoCapture:
    """Open and configure a camera, returning the cv2.VideoCapture instance."""
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
