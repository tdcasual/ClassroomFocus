# tests/test_pitch_basic.py
import numpy as np
from cv.face_analyzer import FaceAnalyzer, FaceAnalyzerConfig, PNP_IDXS

def test_pitch_returns_number():
    fa = FaceAnalyzer(FaceAnalyzerConfig())
    # 构造近似正脸的 2D 点（仅做 smoke test）
    W, H = 640, 480
    pts2d = np.zeros((500,2), dtype=np.float32)
    # 粗略放置：鼻尖在中心，下巴在下方，左右眼外角在两侧，嘴角在中下
    mapping = {
        1: (W/2, H/2),       # nose
        152: (W/2, H/2+120), # chin
        33: (W/2-90, H/2-40),
        263:(W/2+90, H/2-40),
        61: (W/2-60, H/2+20),
        291:(W/2+60, H/2+20)
    }
    for i,xy in mapping.items():
        pts2d[i] = xy
    pitch = fa._compute_pitch(pts2d, W, H)
    assert pitch is None or isinstance(pitch, float)
