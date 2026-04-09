import cv2
import time
from ultralytics import YOLO

cap = cv2.VideoCapture("/dev/video8", cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

if not cap.isOpened():
    raise RuntimeError("Не удалось открыть камеру")

model = YOLO("yolo26n_openvino_model")

frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    results = model(frame, imgsz=320, verbose=False)
    annotated = results[0].plot()

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
        2
    )

    cv2.imshow("OpenVINO", annotated)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()