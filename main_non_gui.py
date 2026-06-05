import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import time

MODEL_NAME = "yolov8s.pt"
VIDEO_SOURCE = 2

width = 1250
height = 700
margin = 25

MAX_CAPACITY = 80
CONFIDENCE_THRESHOLD = 0.45
USE_ROI = True
FRAME_SKIP = 2

cv2.setNumThreads(0)

print("YOLO 모델 로드 중...")
model = YOLO(MODEL_NAME)
model.to("cpu")

cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    raise RuntimeError("카메라 열기 실패")

tracker = sv.ByteTrack(
    track_activation_threshold=0.25,
    lost_track_buffer=30,
    minimum_matching_threshold=0.8,
    frame_rate=30,
)

ROI_POLYGON = np.array([
    [margin, margin],
    [width, margin],
    [width, height],
    [margin, height]
])

zone = sv.PolygonZone(
    polygon=ROI_POLYGON
)

prev_time = time.time()
frame_count = 0

print("실시간 모니터링 시작")
print("-" * 60)

try:
    while True:

        ret, frame = cap.read()

        if not ret:
            print("프레임 읽기 실패")
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (1280, 720))

        frame_count += 1

        if frame_count % FRAME_SKIP != 0:
            continue

        results = model.predict(
            source=frame,
            conf=CONFIDENCE_THRESHOLD,
            classes=[0],
            verbose=False,
            device="cpu"
        )[0]

        detections = sv.Detections.from_ultralytics(results)

        if USE_ROI:
            mask = zone.trigger(detections=detections)
            detections = detections[mask]

        detections = tracker.update_with_detections(detections)

        person_count = len(detections)

        density = min(
            (person_count / MAX_CAPACITY) * 100,
            100
        )

        if density < 30:
            status = "LOW"
        elif density < 70:
            status = "MEDIUM"
        else:
            status = "HIGH"

        current_time = time.time()
        fps = FRAME_SKIP / (current_time - prev_time)
        prev_time = current_time

        print(
            f"\rPeople: {person_count:3d} | "
            f"Density: {density:5.1f}% | "
            f"Status: {status:6s} | "
            f"FPS: {fps:5.1f}",
            end="",
            flush=True
        )

except KeyboardInterrupt:
    print("\n종료 요청됨")

finally:
    cap.release()
    print("카메라 종료")