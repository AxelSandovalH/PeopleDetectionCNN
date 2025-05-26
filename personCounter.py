import cv2
from ultralytics import YOLO
import cvzone
import sqlite3
from datetime import datetime

# Cargar modelo YOLOv8
model = YOLO('yolo12n.pt')
names = model.names

# Línea vertical para conteo (ajustar si cambias resolución)
line_x = 278

# Historial de tracking
track_history = {}

# Contadores
in_count = 0
out_count = 0

# ✅ RTSP de tu cámara IP
cap = cv2.VideoCapture("rtsp://admin:admin@192.168.100.2:554/11")

# Función para registrar eventos en la base de datos
def registrar_evento(direccion):
    conn = sqlite3.connect('people_counter.db')
    c = conn.cursor()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    c.execute('INSERT INTO eventos (timestamp, direccion) VALUES (?, ?)', (timestamp, direccion))
    conn.commit()
    conn.close()

# Para obtener coordenadas del mouse (opcional para ajustar línea)
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Mouse moved to: [{x}, {y}]")

# Mostrar coordenadas con mouse
cv2.namedWindow("RGB")
cv2.setMouseCallback("RGB", RGB)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo acceder al stream RTSP.")
        break

    frame_count += 1
    if frame_count % 2 != 0:
        continue  # Saltar cuadros para aligerar el proceso

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

                # Entrada
                if prev_cx < line_x <= cx:
                    in_count += 1
                    registrar_evento('IN')

                # Salida
                elif prev_cx > line_x >= cx:
                    out_count += 1
                    registrar_evento('OUT')

            cv2.circle(frame, (cx, int((y1 + y2) / 2)), 4, (255, 0, 0), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cvzone.putTextRect(frame, f'{track_id}', (x1, y1), 1, 1)
            track_history[track_id] = (cx, (y1 + y2) // 2)

    # Mostrar contadores
    cvzone.putTextRect(frame, f'IN: {in_count}', (40, 60), scale=2, thickness=2,
                       colorT=(255, 255, 255), colorR=(0, 128, 0))
    cvzone.putTextRect(frame, f'OUT: {out_count}', (40, 100), scale=2, thickness=2,
                       colorT=(255, 255, 255), colorR=(0, 0, 255))

    # Línea de conteo
    cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (255, 255, 255), 2)

    # Mostrar frame
    cv2.imshow("RGB", frame)

    # Salir con ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
