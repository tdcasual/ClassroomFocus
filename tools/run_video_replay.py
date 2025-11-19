# tools/run_video_replay.py
import cv2, time
from cv.face_analyzer import FaceAnalyzer, FaceAnalyzerConfig

video = "samples/classroom_3p.mp4"  # 你的测试视频
cfg = FaceAnalyzerConfig(debug_draw=True)  # 开显示
fa = FaceAnalyzer(cfg)

cap = cv2.VideoCapture(video)
t0 = time.time(); frames = 0
while True:
    ok, frame = cap.read()
    if not ok: break
    ts = time.time()
    results, events = fa.analyze_frame(frame, ts)
    for e in events:
        print(e)
    frames += 1
    cv2.imshow("replay", frame)
    if cv2.waitKey(1) & 0xFF == 27: break
cap.release(); cv2.destroyAllWindows()
print("FPS:", frames / (time.time()-t0))
