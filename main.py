import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import time


# 설정
MODEL_NAME = "yolov8s.pt"
VIDEO_SOURCE = "walking.mp4"

# 가로 세로 범위
width = 1250
height = 700
margin = 25

# 최대 수용 가능 인원
MAX_CAPACITY = 80

# 사람 검출 신뢰도
CONFIDENCE_THRESHOLD = 0.45

# ROI 사용 여부
USE_ROI = True

# CPU 최적화
cv2.setNumThreads(0)

# 모델 로드
print("YOLO 모델 로드 중...")
model = YOLO(MODEL_NAME)

# CPU 강제 사용
model.to("cpu")

# 비디오 입력
cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    print("영상 열기 실패")
    exit()

# ByteTrack 추적기
tracker = sv.ByteTrack(
    track_activation_threshold=0.25,
    lost_track_buffer=30,
    minimum_matching_threshold=0.8,
    frame_rate=30,
)

# 박스 시각화
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

# FPS 계산
prev_time = time.time()

# 메인 루프
while True:
    ret, frame = cap.read()

    if not ret:
        break

    # 프레임 크기 조절
    frame = cv2.resize(frame, (1280, 720))

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

    # 포화도 계산
    density = (person_count / MAX_CAPACITY) * 100
    density = min(density, 100)

    # 혼잡도 단계
    if density < 30:
        status = "LOW"
        color = (0, 255, 0)

    elif density < 70:
        status = "MEDIUM"
        color = (0, 255, 255)

    else:
        status = "HIGH"
        color = (0, 0, 255)


    # 라벨 생성
    labels = []

    for tracker_id in detections.tracker_id:
        labels.append(f"Person #{tracker_id}")

    # 박스 표시
    annotated_frame = frame.copy()

    annotated_frame = box_annotator.annotate(
        scene=annotated_frame,
        detections=detections
    )

    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels
    )

    if USE_ROI:
        annotated_frame = zone_annotator.annotate(annotated_frame)

    # FPS 계산
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # 정보 표시
    cv2.rectangle(annotated_frame, (20, 20), (420, 190), (30, 30, 30), -1)

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

    # 출력
    cv2.imshow("CCTV Crowd Monitoring", annotated_frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

# 종료
cap.release()
cv2.destroyAllWindows()
