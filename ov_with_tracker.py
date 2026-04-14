import cv2
import time
from ultralytics import YOLO

from candy_tracker import CandyTracker


# ===== Camera =====
cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

if not cap.isOpened():
    raise RuntimeError("Не удалось открыть камеру")


# ===== Detector =====
model = YOLO("yolo26n_best_openvino_model")


# ===== Tracker =====
FRAME_WIDTH = 640
COUNT_LINE_X = int(FRAME_WIDTH * 0.60)  # вертикальная линия счета

tracker = CandyTracker(
    line_x=COUNT_LINE_X,
    max_distance=75,   # допустимый сдвиг центра между кадрами  default 50
    min_hits=10,        # минимум подтверждений до учета        default 3
    max_missed=15,      # сколько кадров держать потерянный трек  default 5
    direction="left_to_right",  # направление движения
)


# ===== Runtime =====
CONF_THRESHOLD = 0.45
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    results = model(frame, imgsz=320, verbose=False)  # 320
    result = results[0]
    annotated = frame.copy()

    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append((x1, y1, x2, y2))

    tracks = tracker.update(detections)

    # draw detections/tracks
    for track in tracks:
        x1, y1, x2, y2 = track.bbox
        cx, cy = track.center

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(
            annotated,
            f"ID {track.track_id}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    # count line
    cv2.line(annotated, (COUNT_LINE_X, 0), (COUNT_LINE_X, frame.shape[0]), (255, 255, 0), 2)

    frame_count += 1
    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0

    cv2.putText(
        annotated,
        f"FPS: {fps:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        annotated,
        f"COUNT: {tracker.total_count}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2,
    )

    cv2.imshow("OpenVINO + Tracker", annotated)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
