from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO
import cv2
import sqlite3
from datetime import datetime

app = Flask(__name__)

model = YOLO('yolo12n.pt')
names = model.names

track_history = {}
in_count = 0
out_count = 0
line_x = 278

def init_db():
    conn = sqlite3.connect('people_counter.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS eventos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            direccion TEXT CHECK(direccion IN ('IN', 'OUT')) NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def registrar_evento(direccion):
    conn = sqlite3.connect('people_counter.db')
    c = conn.cursor()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    c.execute('INSERT INTO eventos (timestamp, direccion) VALUES (?, ?)', (timestamp, direccion))
    conn.commit()
    conn.close()

def obtener_estadisticas():
    conn = sqlite3.connect('people_counter.db')
    c = conn.cursor()
    hoy = datetime.now().strftime('%Y-%m-%d')
    c.execute("SELECT COUNT(*) FROM eventos WHERE direccion='IN' AND timestamp LIKE ?", (f'{hoy}%',))
    entradas = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM eventos WHERE direccion='OUT' AND timestamp LIKE ?", (f'{hoy}%',))
    salidas = c.fetchone()[0]
    conn.close()
    actuales = entradas - salidas
    return entradas, salidas, actuales

@app.route('/stats')
def stats():
    entradas, salidas, actuales = obtener_estadisticas()
    # Además enviamos el conteo actual para mostrarlo instantáneo (in_count y out_count)
    return jsonify({
        "entradas": entradas,
        "salidas": salidas,
        "actuales": actuales,
        "in_count": in_count,
        "out_count": out_count
    })

def gen_frames():
    global track_history, in_count, out_count

    cap = cv2.VideoCapture("rtsp://admin:admin@192.168.100.2:554/11")  # Reemplaza aquí

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.resize(frame, (1020, 600))

        results = model.track(frame, persist=True, classes=[0])

        if results[0].boxes.id is not None:
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            class_ids = results[0].boxes.cls.int().cpu().tolist()

            for track_id, box, class_id in zip(ids, boxes, class_ids):
                x1, y1, x2, y2 = box
                cx = int((x1 + x2) / 2)

                if track_id in track_history:
                    prev_cx, _ = track_history[track_id]

                    if prev_cx < line_x <= cx:
                        in_count += 1
                        registrar_evento('IN')

                    elif prev_cx > line_x >= cx:
                        out_count += 1
                        registrar_evento('OUT')

                track_history[track_id] = (cx, (y1 + y2) // 2)

                cv2.circle(frame, (cx, int((y1 + y2) / 2)), 4, (255, 0, 0), -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Línea de conteo
        cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (255, 255, 255), 2)

        # Texto en video
        cv2.putText(frame, f'IN: {in_count}', (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'OUT: {out_count}', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
