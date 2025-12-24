import threading
import time
import state #обмен данными между потоками
from flask import Flask, render_template_string


# ===== Flask =====
app = Flask(__name__)

HTML_PAGE = """
<!doctype html>
<html>
<head>
    <meta charset="utf-8">
    <title>Sweetie Counter</title>
    <meta http-equiv="refresh" content="1">
    <style>
        body {
            font-family: Arial, sans-serif;
            text-align: center;
            margin-top: 100px;
        }
        .count {
            font-size: 72px;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <h1>Количество конфет</h1>
    <div class="count">{{ count }}</div>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_PAGE, count=state.sweet_count)

def run_flask():
    app.run(host="0.0.0.0", port=5000, debug=False)

# ===== Имитация  для тестов =====
def detection_loop_test():
    global sweet_count
    while True:
        # n
        sweet_count += 1
        time.sleep(1)

# ===== поток обнавлемый раз в секунду =====
def detection_loop():
    global sweet_count
    while True:
        time.sleep(1)

if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()

    #detection_loop_test()
    #detection_loop()
