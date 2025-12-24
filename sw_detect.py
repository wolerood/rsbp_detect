import cv2
import time
from ultralytics import YOLO
from picamera2 import Picamera2
from libcamera import Transform
import state #обмен данными между потоками


# Вывод видео
video_play = True
show_label = False  # показывать класс
show_conf  = False  # показыать confidence

# Start PiCamera stream
picam2 = Picamera2()
# specifying the resolution and performing a 180 degree rotation
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}, transform=Transform(hflip=False, vflip=True)))
picam2.start()
time.sleep(2)  # warm-up

# Load YOLOv8 nano model
model = YOLO("best.pt")

frame_count = 0
start_time = time.time()

while True:
    frame = picam2.capture_array()
    if frame is None:
        continue

    # Convert RGBA → RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

    results = model(frame, imgsz=320, verbose=False)

    # FPS count
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0

    # Detect object count
    detect_num = (len(results[0].boxes))

    if video_play:
        annotated_frame = results[0].plot(
            labels=show_label,   
            conf=show_conf,      # confidence
        )

        cv2.putText(annotated_frame, f"FPS: {fps:.2f} sweetie: {detect_num}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)
        
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Show annotated frame
        cv2.imshow("Sweetie Detection", rgb_frame)

    print(f'FPS: {"%.1f" % fps}, Обнаружено объектов: {detect_num}')
    
    sweet_count = detect_num

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cv2.destroyAllWindows()
