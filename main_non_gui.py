import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import time
import csv
from datetime import datetime
import os

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


# ==========================================================
# [추가] CSV 파일 설정 및 초기화
# ==========================================================
csv_filename = "crowd_data.csv"
file_exists = os.path.isfile(csv_filename)

# 'a' (append) 모드로 오픈하여 기존 데이터 누적 저장
csv_file = open(csv_filename, mode='a', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file)

# 파일이 처음 생성되는 경우에만 헤더(컬럼명)를 작성
if not file_exists:
    csv_writer.writerow(["Timestamp", "People Count", "Density (%)", "Status"])
    csv_file.flush()
# ==========================================================


prev_time = time.time()
last_csv_save_time = time.time()  # [추가] 마지막 CSV 저장 시간을 기록할 타이머 변수
frame_count = 0

print("실시간 모니터링 시작")
print("-" * 60)

try:
    while True:

        ret, frame = cap.read()

        if not ret:
            print("\n프레임 읽기 실패")
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


        # ==========================================================
        # [추가] 초당 1회 CSV 데이터 저장 루틴
        # ==========================================================
        current_time = time.time()
        if current_time - last_csv_save_time >= 1.0:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            csv_writer.writerow([now, person_count, round(density, 1), status])
            csv_file.flush()  # 강제 종료 시 데이터가 유실되지 않도록 디스크에 즉시 기록
            last_csv_save_time = current_time
        # ==========================================================


        # 정확한 루프 주기를 반영하기 위해 current_time 재활용
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
    # ==========================================================
    # [추가/수정] 종료 시 CSV 파일 안전하게 닫고 자원 해제
    # ==========================================================
    csv_file.close()
    cap.release()
    print("카메라 종료 및 데이터 저장 완료")