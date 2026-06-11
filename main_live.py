import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import time
import csv
from datetime import datetime
import os

# 모델 설정
MODEL_NAME = "yolov8s.pt"

# 실시간 카메라( 인덱스 넘버 입력하기 )
VIDEO_SOURCE = 2

# ROI 범위
width = 1250
height = 700
margin = 25

# 최대 수용 인원
MAX_CAPACITY = 80

# 사람 검출 신뢰도
CONFIDENCE_THRESHOLD = 0.45

# ROI 사용 여부
USE_ROI = True

# 프레임 스킵
FRAME_SKIP = 2

# CPU 최적화
cv2.setNumThreads(0)


# YOLO 모델 로드
print("YOLO 모델 로드 중...")

model = YOLO(MODEL_NAME)

# CPU 사용
model.to("cpu")


# 카메라 열기
cap = cv2.VideoCapture(VIDEO_SOURCE)

# 해상도 설정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("카메라 열기 실패")
    exit()


# ByteTrack 추적기
tracker = sv.ByteTrack(
    track_activation_threshold=0.25,
    lost_track_buffer=30,
    minimum_matching_threshold=0.8,
    frame_rate=30,
)

# 박스 표시
box_annotator = sv.BoxAnnotator(thickness=2)

label_annotator = sv.LabelAnnotator(
    text_scale=0.6,
    text_thickness=1,
)


# ROI 설정
ROI_POLYGON = np.array([
    [margin, margin],
    [width, margin],
    [width, height],
    [margin, height]
])

zone = sv.PolygonZone(
    polygon=ROI_POLYGON
)

zone_annotator = sv.PolygonZoneAnnotator(
    zone=zone,
    color=sv.Color.GREEN,
    thickness=2,
)


# ==========================================================
# [추가] CSV 파일 설정
# ==========================================================
csv_filename = "crowd_data.csv"
file_exists = os.path.isfile(csv_filename)

# 파일이 없으면 헤더(컬럼명)를 쓰고, 있으면 기존 데이터 뒤에 이어 씁니다.
csv_file = open(csv_filename, mode='a', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file)

if not file_exists:
    csv_writer.writerow(["Timestamp", "People Count", "Density (%)", "Status"])
    csv_file.flush()
# ==========================================================


# FPS 및 CSV 저장 타이머 계산용 변수
prev_time = time.time()
last_csv_save_time = time.time()  # [추가] 마지막으로 CSV 저장한 시간을 기록


# 프레임 변수
frame_count = 0


# 메인 루프 (예외 처리를 위해 try 구문 추가)
try:
    while True:

        ret, frame = cap.read()

        if not ret:
            print("프레임 읽기 실패")
            break

        # 좌우 반전 (웹캠용)
        frame = cv2.flip(frame, 1)

        # 프레임 크기 조절
        frame = cv2.resize(frame, (1280, 720))

        frame_count += 1


        # 프레임 스킵
        if frame_count % FRAME_SKIP != 0:
            continue


        # YOLO 추론
        results = model.predict(
            source=frame,
            conf=CONFIDENCE_THRESHOLD,
            classes=[0],
            verbose=False,
            device="cpu"
        )[0]

        # supervision 형식 변환
        detections = sv.Detections.from_ultralytics(results)


        # ROI 필터링
        if USE_ROI:
            mask = zone.trigger(detections=detections)
            detections = detections[mask]


        # 추적
        detections = tracker.update_with_detections(detections)


        # 사람 수 계산
        person_count = len(detections)

        density = (person_count / MAX_CAPACITY) * 100
        density = min(density, 100)


        # 혼잡도 상태
        if density < 30:
            status = "LOW"
            color = (0, 255, 0)

        elif density < 70:
            status = "MEDIUM"
            color = (0, 255, 255)

        else:
            status = "HIGH"
            color = (0, 0, 255)


        # ==========================================================
        # [추가] 초당 1회 CSV 데이터 저장 루틴
        # ==========================================================
        current_time = time.time()
        if current_time - last_csv_save_time >= 1.0:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            csv_writer.writerow([now, person_count, round(density, 1), status])
            csv_file.flush()  # 버퍼의 데이터를 즉시 파일에 기록 (누락 방지)
            last_csv_save_time = current_time
        # ==========================================================


        # 라벨 생성
        labels = []

        for tracker_id in detections.tracker_id:
            labels.append(f"Person #{tracker_id}")


        # 화면 출력용 복사
        annotated_frame = frame.copy()

        # 박스 표시
        annotated_frame = box_annotator.annotate(
            scene=annotated_frame,
            detections=detections
        )

        # 라벨 표시
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels
        )

        # ROI 표시
        if USE_ROI:
            annotated_frame = zone_annotator.annotate(
                annotated_frame
            )


        # FPS 계산
        fps = 1 / (current_time - prev_time)
        prev_time = current_time


        # 정보 패널
        cv2.rectangle(
            annotated_frame,
            (20, 20),
            (430, 190),
            (30, 30, 30),
            -1
        )

        cv2.putText(
            annotated_frame,
            f"People Count : {person_count}",
            (40, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        cv2.putText(
            annotated_frame,
            f"Density : {density:.1f}%",
            (40, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

        cv2.putText(
            annotated_frame,
            f"Status : {status}",
            (40, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

        cv2.putText(
            annotated_frame,
            f"FPS : {fps:.1f}",
            (280, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


        # 화면 출력
        cv2.imshow(
            "Real-Time CCTV Crowd Monitoring",
            annotated_frame
        )

        # ESC 종료
        key = cv2.waitKey(1)

        if key == 27:
            break

finally:
    # ==========================================================
    # [추가/수정] 프로그램 종료 시 파일 및 카메라 안전하게 닫기
    # ==========================================================
    print("프로그램을 종료하고 데이터를 저장합니다...")
    csv_file.close()
    cap.release()
    cv2.destroyAllWindows()