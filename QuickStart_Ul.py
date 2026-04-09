#Quick Start Guide: Raspberry Pi with Ultralytics YOLO26
#https://docs.ultralytics.com/guides/raspberry-pi/

import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
import time

# Initialize the Picamera2
picam2 = Picamera2()
#picam2.preview_configuration.main.size = (1280, 720)
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

frame_count = 0
start_time = time.time()

# Load the YOLO26 model
model = YOLO("yolo26n.pt")

while True:
    # Capture frame-by-frame
    frame = picam2.capture_array()

    #vert vlip
    frame = cv2.flip(frame, 0)

    # Run YOLO26 inference on the frame
    results = model(frame)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # FPS count
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0

    cv2.putText(annotated_frame, f"FPS: {fps:.2f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow("Camera", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# Release resources and close windows
cv2.destroyAllWindows()