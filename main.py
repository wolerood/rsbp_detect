import threading
import sweetie_web
import sw_detect

'''
threading.Thread(
    target=sweetie_web.run_flask,
    daemon=True
).start()

sw_detect.detection_loop()
'''


# запуск детекции в фоне
threading.Thread(
    target=sw_detect.detection_loop,
    daemon=True
).start()

# Flask — ТОЛЬКО в главном потоке
sweetie_web.run_flask()
