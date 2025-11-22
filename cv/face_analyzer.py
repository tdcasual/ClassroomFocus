# cv/face_analyzer.py
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

# ---------- 眼部关键点（MediaPipe FaceMesh 索引） ----------
# 左眼：水平(33,133)；垂直(159,145) 与 (160,144)
LEFT_EYE_H = (33, 133)
LEFT_EYE_V1 = (159, 145)
LEFT_EYE_V2 = (160, 144)
# 右眼：水平(263,362)；垂直(386,374) 与 (385,380)
RIGHT_EYE_H = (263, 362)
RIGHT_EYE_V1 = (386, 374)
RIGHT_EYE_V2 = (385, 380)

# 头姿 PnP 6 点（鼻尖、下巴、左眼外角、右眼外角、左嘴角、右嘴角）
PNP_IDXS = [1, 152, 33, 263, 61, 291]
MODEL_POINTS = np.array([
    [0.0,    0.0,    0.0],     # nose tip
    [0.0,  -330.0, -65.0],     # chin
    [-225., 170.,  -135.],     # left eye outer
    [225.,  170.,  -135.],     # right eye outer
    [-150., -150., -125.],     # left mouth corner
    [150.,  -150., -125.],     # right mouth corner
], dtype=np.float32)

def _dist(a, b) -> float:
    return float(np.linalg.norm(a - b))

def _ear_from_pts(pts2d: np.ndarray,
                  H: Tuple[int, int],
                  V1: Tuple[int, int],
                  V2: Tuple[int, int]) -> float:
    """根据眼部6个点计算单眼 EAR"""
    ph = _dist(pts2d[H[0]], pts2d[H[1]]) + 1e-6
    pv = _dist(pts2d[V1[0]], pts2d[V1[1]]) + _dist(pts2d[V2[0]], pts2d[V2[1]])
    return float(pv / (2.0 * ph))

def _avg_two(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None and b is None: return None
    if a is None: return b
    if b is None: return a
    return 0.5 * (a + b)

@dataclass
class TrackState:
    id: int
    last_ts: Optional[float] = None
    center: Tuple[float, float] = (0.0, 0.0)
    miss_count: int = 0

    # 平滑值
    ear_ema: Optional[float] = None
    pitch_ema: Optional[float] = None

    # 标定
    seen_secs: float = 0.0
    ear_open_baseline: Optional[float] = None

    # 眼睛与瞌睡
    eye_closed: bool = False
    closed_timer: float = 0.0
    drowsy_active: bool = False
    drowsy_start_ts: Optional[float] = None

    # 低头
    down: bool = False
    down_timer: float = 0.0
    down_active: bool = False
    down_start_ts: Optional[float] = None

    # 统计
    blink_count: int = 0
    state: str = "awake"  # 可选：awake / drowsy / down / both

@dataclass
class FaceAnalyzerConfig:
    max_faces: int = 5
    refine_landmarks: bool = False
    min_det_conf: float = 0.5
    min_trk_conf: float = 0.5

    # EAR 阈值与标定
    ear_min: float = 0.15
    ear_ratio: float = 0.70
    calibrate_secs: float = 3.0
    ear_ema_alpha: float = 0.3

    # 瞌睡/眨眼阈值
    drowsy_secs: float = 2.8
    blink_max_secs: float = 0.25
    recover_secs: float = 0.8

    # 低头（Pitch）阈值（绕 x 轴，向下为负多数情况）
    pitch_down_deg: float = -20.0
    down_secs: float = 1.6

    # 多脸匹配
    match_max_px: float = 80.0
    max_miss_count: int = 10

    # 调试显示
    debug_draw: bool = False

class FaceAnalyzer:
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

    # ---------- 核心：分析一帧 ----------
    def analyze_frame(self, frame_bgr: np.ndarray, timestamp: float):
        H, W = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self._mesh.process(rgb)

        # 收集本帧所有人脸的 2D 坐标与中心
        faces_pts: List[np.ndarray] = []
        centers: List[Tuple[float, float]] = []
        if res.multi_face_landmarks:
            for fl in res.multi_face_landmarks[: self.cfg.max_faces]:
                n = len(fl.landmark)  # 468 或 478
                pts2d = np.zeros((n, 2), dtype=np.float32)
                xs, ys = [], []
                for i, lm in enumerate(fl.landmark):
                    x, y = lm.x * W, lm.y * H
                    pts2d[i] = (x, y)
                    xs.append(x); ys.append(y)
                faces_pts.append(pts2d)
                centers.append((float(np.mean(xs)), float(np.mean(ys))))

        # 数据关联（最近邻）
        assign = self._associate(centers)

        results = []   # 每人脸的原始数值
        events = []    # 状态切换事件
        used_ids = set()

        for idx, pts2d in enumerate(faces_pts):
            center = centers[idx]
            tid = assign.get(idx)
            if tid is None:
                # 新建 track
                tid = self._next_id
                self._next_id += 1
                self.tracks[tid] = TrackState(id=tid, center=center, last_ts=timestamp)
            st = self.tracks[tid]
            used_ids.add(tid)

            # 计算 dt
            dt = 0.0 if st.last_ts is None else max(0.0, timestamp - st.last_ts)
            st.last_ts = timestamp
            st.center = center
            st.miss_count = 0  # 本帧命中

            # -------- EAR ----------
            left_ear = _ear_from_pts(pts2d, LEFT_EYE_H, LEFT_EYE_V1, LEFT_EYE_V2)
            right_ear = _ear_from_pts(pts2d, RIGHT_EYE_H, RIGHT_EYE_V1, RIGHT_EYE_V2)
            ear = (left_ear + right_ear) / 2.0

            # EMA 平滑
            st.ear_ema = ear if st.ear_ema is None else \
                (self.cfg.ear_ema_alpha * ear + (1 - self.cfg.ear_ema_alpha) * st.ear_ema)

            # 标定
            if st.seen_secs < self.cfg.calibrate_secs:
                st.seen_secs += dt
                if st.ear_open_baseline is None:
                    st.ear_open_baseline = st.ear_ema
                else:
                    # 慢速逼近
                    st.ear_open_baseline = 0.9 * st.ear_open_baseline + 0.1 * st.ear_ema

            ear_base = st.ear_open_baseline if st.ear_open_baseline else 0.28
            ear_thresh = max(self.cfg.ear_min, self.cfg.ear_ratio * ear_base)

            # -------- Pitch ----------
            pitch = self._compute_pitch(pts2d, W, H)
            if pitch is not None:
                st.pitch_ema = pitch if st.pitch_ema is None else 0.7 * st.pitch_ema + 0.3 * pitch

            # -------- 状态机：眼睛/瞌睡 ----------
            was_closed = st.eye_closed
            st.eye_closed = (st.ear_ema is not None) and (st.ear_ema < ear_thresh)

            if st.eye_closed:
                st.closed_timer += dt
                # 刚跨过 drowsy 阈值 → 触发 START
                if (not st.drowsy_active) and (st.closed_timer >= self.cfg.drowsy_secs):
                    st.drowsy_active = True
                    st.drowsy_start_ts = timestamp - st.closed_timer
                    events.append({
                        "ts": st.drowsy_start_ts,
                        "student_id": tid,
                        "type": "DROWSY_START",
                        "dur": None,
                        "ear": float(st.ear_ema),
                        "pitch": float(st.pitch_ema) if st.pitch_ema is not None else None
                    })
            else:
                # 刚从闭眼恢复
                if was_closed:
                    # 短暂闭眼：BLINK
                    if 0.0 < st.closed_timer < self.cfg.blink_max_secs:
                        st.blink_count += 1
                        events.append({
                            "ts": timestamp,
                            "student_id": tid,
                            "type": "BLINK",
                            "dur": float(st.closed_timer),
                            "ear": float(st.ear_ema),
                            "pitch": float(st.pitch_ema) if st.pitch_ema is not None else None
                        })
                    # 长闭眼结束：DROWSY_END（若已进入 drowsy）
                    if st.drowsy_active:
                        events.append({
                            "ts": timestamp,
                            "student_id": tid,
                            "type": "DROWSY_END",
                            "dur": float(timestamp - (st.drowsy_start_ts or timestamp)),
                            "ear": float(st.ear_ema),
                            "pitch": float(st.pitch_ema) if st.pitch_ema is not None else None
                        })
                    st.drowsy_active = False
                    st.drowsy_start_ts = None
                st.closed_timer = 0.0

            # -------- 状态机：低头 ----------
            was_down = st.down
            # 注意 Pitch 正负号可能因坐标系不同略有差异，这里按“向下为负”处理
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
                        "pitch": float(st.pitch_ema) if st.pitch_ema is not None else None
                    })
            else:
                if was_down and st.down_active:
                    events.append({
                        "ts": timestamp,
                        "student_id": tid,
                        "type": "LOOKING_DOWN_END",
                        "dur": float(timestamp - (st.down_start_ts or timestamp)),
                        "ear": float(st.ear_ema),
                        "pitch": float(st.pitch_ema) if st.pitch_ema is not None else None
                    })
                st.down_active = False
                st.down_start_ts = None
                st.down_timer = 0.0

            # -------- 输出每人脸的数值 --------
            # 归一化状态（便于上层显示）
            state = "awake"
            if st.drowsy_active and st.down_active:
                state = "drowsy+down"
            elif st.drowsy_active:
                state = "drowsy"
            elif st.down_active:
                state = "down"
            st.state = state

            results.append({
                "student_id": tid,
                "ear": float(st.ear_ema) if st.ear_ema is not None else None,
                "pitch": float(st.pitch_ema) if st.pitch_ema is not None else None,
                "state": st.state,
                "ear_thresh": float(ear_thresh),
                "blink_count": st.blink_count
            })

        # 未匹配到的人脸 track 递增 miss_count，超限回收
        for tid, st in list(self.tracks.items()):
            if tid not in used_ids:
                st.miss_count += 1
                if st.miss_count > self.cfg.max_miss_count:
                    # 若在消失时仍处于 active 状态，补一条 END 事件
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

        # 可选：在帧上叠加调试文本
        if self.cfg.debug_draw:
            self._draw_debug(frame_bgr, results)

        return results, events

    # ---------- 头姿求解 ----------
    def _compute_pitch(self, pts2d: np.ndarray, W: int, H: int) -> Optional[float]:
        try:
            image_points = np.array([pts2d[i] for i in PNP_IDXS], dtype=np.float32)
            f = 1.2 * W
            K = np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1]], dtype=np.float32)
            dist = np.zeros(5, dtype=np.float32)
            ok, rvec, tvec = cv2.solvePnP(MODEL_POINTS, image_points, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok: return None
            R, _ = cv2.Rodrigues(rvec)
            sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
            pitch = np.degrees(np.arctan2(-R[2, 0], sy))  # 向下为负；如方向相反，可取负号
            return float(pitch)
        except Exception:
            return None

    # ---------- 关联 ----------
    def _associate(self, centers: List[Tuple[float, float]]) -> Dict[int, int]:
        """把本帧的人脸 idx 关联到已有 track_id（最近邻+距离阈值）"""
        assign: Dict[int, int] = {}
        if not centers:
            return assign
        # 简单贪心：对于每个新中心，找最近的 track
        # （人脸数目不大，够用；需要更强时可用匈牙利算法）
        unused_tracks = set(self.tracks.keys())
        for i, c in enumerate(centers):
            best_tid, best_d = None, 1e9
            for tid in unused_tracks:
                st = self.tracks[tid]
                d = np.hypot(c[0] - st.center[0], c[1] - st.center[1])
                if d < best_d:
                    best_d, best_tid = d, tid
            if best_tid is not None and best_d <= self.cfg.match_max_px:
                assign[i] = best_tid
                unused_tracks.remove(best_tid)
            else:
                assign[i] = None
        return assign

    # ---------- 调试绘制 ----------
    def _draw_debug(self, frame_bgr: np.ndarray, results: List[Dict]):
        y = 30
        for r in results:
            txt = f"ID{r['student_id']} EAR={r['ear']:.2f} (thr {r['ear_thresh']:.2f}) " \
                  f"Pitch={r['pitch'] if r['pitch'] is not None else 'NA'}  {r['state']}"
            cv2.putText(frame_bgr, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255) if "drowsy" in r['state'] else (0, 255, 0), 2)
            y += 24


# -------- 便捷 CLI：实时摄像头演示 --------
if __name__ == "__main__":
    import argparse, time as _time
    import platform # 引入这个库来判断系统

    ap = argparse.ArgumentParser()
    ap.add_argument("--webcam", type=int, default=0, help="摄像头索引（默认0）")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--show", action="store_true", help="显示调试窗口")
    args = ap.parse_args()

    cfg = FaceAnalyzerConfig(debug_draw=args.show)
    analyzer = FaceAnalyzer(cfg)

    # --- 针对不同系统的兼容性修改 ---
    current_os = platform.system()
    if current_os == "Windows":
        # Windows 下推荐使用 DirectShow
        backend = cv2.CAP_DSHOW
    elif current_os == "Linux":
        # Linux 下使用 V4L2 (例如树莓派)
        backend = cv2.CAP_V4L2
    else:
        # macOS 或其他系统自动选择
        backend = cv2.CAP_ANY
    
    print(f"正在当前系统 ({current_os}) 上启动摄像头...")
    cap = cv2.VideoCapture(args.webcam, backend)
    # ----------------------------

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print(f"无法打开摄像头 (index={args.webcam})。请检查连接或尝试 --webcam 1")
        exit(0)

    t0 = _time.time()
    frames = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("无法读取视频帧，退出...")
                break
            
            ts = _time.time()
            results, events = analyzer.analyze_frame(frame, ts)
            frames += 1
            
            # 打印事件
            for e in events:
                print(e)
            
            if args.show:
                cv2.imshow("FaceAnalyzer", frame)
                # Windows 下 ESC 键退出
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        dur = max(1e-6, _time.time() - t0)
        print(f"Frames={frames}, FPS={frames/dur:.2f}")
