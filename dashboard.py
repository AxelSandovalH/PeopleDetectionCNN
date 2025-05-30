from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO
import cv2
import sqlite3
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

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

def obtener_datos_por_hora():
    conn = sqlite3.connect('people_counter.db')
    c = conn.cursor()
    hoy = datetime.now().strftime('%Y-%m-%d')
    
    # Obtener datos por hora del día actual
    c.execute("""
        SELECT 
            CAST(strftime('%H', timestamp) AS INTEGER) as hora,
            direccion,
            COUNT(*) as total
        FROM eventos 
        WHERE timestamp LIKE ? 
        GROUP BY hora, direccion
        ORDER BY hora
    """, (f'{hoy}%',))
    
    resultados = c.fetchall()
    conn.close()
    
    # Preparar datos para la gráfica (24 horas)
    datos_hora = []
    entradas_por_hora = defaultdict(int)
    salidas_por_hora = defaultdict(int)
    
    for hora, direccion, total in resultados:
        if direccion == 'IN':
            entradas_por_hora[hora] = total
        else:
            salidas_por_hora[hora] = total
    
    for h in range(24):
        datos_hora.append({
            'hora': f'{h:02d}:00',
            'entradas': entradas_por_hora[h],
            'salidas': salidas_por_hora[h],
            'flujo': entradas_por_hora[h] - salidas_por_hora[h]
        })
    
    return datos_hora

def obtener_datos_semanal():
    conn = sqlite3.connect('people_counter.db')
    c = conn.cursor()
    
    # Obtener datos de los últimos 7 días
    fecha_inicio = (datetime.now() - timedelta(days=6)).strftime('%Y-%m-%d')
    
    c.execute("""
        SELECT 
            DATE(timestamp) as fecha,
            direccion,
            COUNT(*) as total
        FROM eventos 
        WHERE DATE(timestamp) >= ?
        GROUP BY fecha, direccion
        ORDER BY fecha
    """, (fecha_inicio,))
    
    resultados = c.fetchall()
    conn.close()
    
    # Preparar datos para los últimos 7 días
    datos_semana = {}
    
    for i in range(7):
        fecha = (datetime.now() - timedelta(days=6-i)).strftime('%Y-%m-%d')
        dia_semana = (datetime.now() - timedelta(days=6-i)).strftime('%a')
        datos_semana[fecha] = {
            'fecha': fecha,
            'dia': dia_semana,
            'entradas': 0,
            'salidas': 0
        }
    
    for fecha, direccion, total in resultados:
        if fecha in datos_semana:
            if direccion == 'IN':
                datos_semana[fecha]['entradas'] = total
            else:
                datos_semana[fecha]['salidas'] = total
    
    return list(datos_semana.values())

@app.route('/api/status')
def stats():
    entradas, salidas, actuales = obtener_estadisticas()
    return jsonify({
        "entradas": entradas,
        "salidas": salidas,
        "actuales": actuales
    })

@app.route('/api/datos-hora')
def datos_por_hora():
    datos = obtener_datos_por_hora()
    return jsonify(datos)

@app.route('/api/datos-semana')
def datos_semanal():
    datos = obtener_datos_semanal()
    return jsonify(datos)

def gen_frames():
    global track_history, in_count, out_count

    try:
        cap = cv2.VideoCapture("rtsp://admin:admin@192.168.100.8:554/11")  # Reemplaza con tu fuente de video
        if not cap.isOpened():
            raise RuntimeError("No se pudo abrir la fuente de video")
            
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
            cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (0, 0, 255), 2)

            # Texto en video
            cv2.putText(frame, f'IN: {in_count}', (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'OUT: {out_count}', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    except Exception as e:
        print(f"Error en gen_frames: {e}")
        # Generar frame de error
        error_frame = np.zeros((300, 600, 3), dtype=np.uint8)
        cv2.putText(error_frame, f"Error: {str(e)}", (50, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        _, buffer = cv2.imencode('.jpg', error_frame)
        error_frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + error_frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)