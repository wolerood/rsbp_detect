import cv2
import time
from ultralytics import YOLO
from picamera2 import Picamera2
from libcamera import Transform
import state  # общий модуль состояния


def detection_loop():
    # ===== настройки отображения =====
    video_play = True
    show_label = False
    show_conf = False

    # ===== инициализация камеры =====
    picam2 = Picamera2()
    picam2.configure(
        picam2.create_video_configuration(
            main={"size": (640, 480)},
            transform=Transform(hflip=False, vflip=True)
        )
    )
    picam2.start()
    time.sleep(2)  # задержка перед стартом

    # ===== загрузка модели =====
    model = YOLO("best.pt")

    # ===== счётчики FPS =====
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            frame = picam2.capture_array()
            if frame is None:
                continue

            # RGBA → RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

            # инференс
            results = model(frame, imgsz=320, verbose=False)

            # подсчёт FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0.0

            # количество объектов
            detect_num = len(results[0].boxes)

            # запись в общее состояние
            state.sweet_count = detect_num
            state.fps_value = fps

            # отладочное окно
            if video_play:
                annotated_frame = results[0].plot(
                    labels=show_label,
                    conf=show_conf
                )

                cv2.putText(
                    annotated_frame,
                    f"FPS: {fps:.2f}  sweetie: {detect_num}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

                cv2.imshow("Sweetie Detection", annotated_frame)

            print(f"FPS: {fps:.1f}, Обнаружено объектов: {detect_num}")

            # выход по ESC
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        # освобождение ресурсов
        cv2.destroyAllWindows()
        picam2.stop()
