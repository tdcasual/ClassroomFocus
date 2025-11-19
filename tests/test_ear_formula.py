# tests/test_ear_formula.py
import numpy as np
from cv.face_analyzer import _ear_from_pts, LEFT_EYE_H, LEFT_EYE_V1, LEFT_EYE_V2

def test_ear_geometry():
    # 构造一个“眼睛”局部坐标系：水平100像素，竖直分别20/18像素
    pts2d = np.zeros((500,2), dtype=np.float32)
    # 左眼水平点
    pts2d[LEFT_EYE_H[0]] = (0, 0)
    pts2d[LEFT_EYE_H[1]] = (100, 0)
    # 两条竖线
    pts2d[LEFT_EYE_V1[0]] = (50, -10)
    pts2d[LEFT_EYE_V1[1]] = (50,  10)
    pts2d[LEFT_EYE_V2[0]] = (50,  -9)
    pts2d[LEFT_EYE_V2[1]] = (50,   9)
    ear = _ear_from_pts(pts2d, LEFT_EYE_H, LEFT_EYE_V1, LEFT_EYE_V2)
    # 期望 EAR = (20 + 18) / (2*100) = 0.19
    assert abs(ear - 0.19) < 1e-3


