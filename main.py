import threading
import sweetie_web
import sw_detect

threading.Thread(
    target=sweetie_web.run_flask,
    daemon=True
).start()

sw_detect.detection_loop()
