import cv2
import time
import json
from collections import defaultdict
from ultralytics import YOLO

# config
VIDEO_PATH        = r"C:\Users\cesar\Downloads\YOLO_video.mp4"
OUTPUT_PATH       = r"C:\Users\cesar\Downloads\YOLO_output.mp4"
CLASES_INTERES    = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
UMBRAL_QUIETO_PX  = 15
UMBRAL_QUIETO_SEG = 2.0
COOLDOWN_EVENTO   = 5.0
KAFKA_BROKER      = "localhost:9092"
KAFKA_TOPIC       = "semaforo-eventos"
SEGUNDOS_SALTAR   = 10
FRAME_SKIP        = 3

# colores BGR
COLOR_QUIETO   = (0, 255, 100)
COLOR_MOVIENDO = (180, 180, 180)
COLOR_HUD      = (255, 255, 0)

# kafka con fallback
try:
    from kafka import KafkaProducer
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )
    kafka_ok = True
except Exception:
    producer = None
    kafka_ok = False

# setup
model = YOLO("yolov8n.pt")
model.to("cpu")

# estado
historial_pos        = defaultdict(list)
primer_quieto        = {}
ids_avisados         = {}
ultimo_boxes         = {}
conteo_quietos       = 0
ultimo_evento_global = 0.0
frame_count          = 0


def centroide(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def esta_quieto(historial):
    if len(historial) < 5:
        return False
    recientes = historial[-10:]
    xs = [p[0] for p in recientes]
    ys = [p[1] for p in recientes]
    return max(max(xs) - min(xs), max(ys) - min(ys)) < UMBRAL_QUIETO_PX


def mandar_evento(objeto_id, clase, cx, cy, seg_quieto):
    global conteo_quietos
    conteo_quietos += 1
    payload = {
        "evento":     "objeto_quieto",
        "objeto_id":  int(objeto_id),
        "clase":      clase,
        "posicion":   {"x": cx, "y": cy},
        "seg_quieto": round(seg_quieto, 2),
        "timestamp":  round(time.time(), 3),
        "accion":     "cambiar_verde"
    }
    if kafka_ok:
        producer.send(KAFKA_TOPIC, value=payload)
        producer.flush()


def dibujar_objeto(frame, obj_id, clase, box, cx, cy, quieto, seg_quieto):
    color  = COLOR_QUIETO if quieto else COLOR_MOVIENDO
    grosor = 2 if quieto else 1
    x1, y1, x2, y2 = map(int, box)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, grosor)
    cv2.circle(frame, (cx, cy), 5, color, -1)

    label = f"ID:{obj_id} {clase}"
    if quieto:
        label += f"  DETENIDO {seg_quieto:.1f}s"

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)


# main
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"No se pudo abrir: {VIDEO_PATH}")

fps_video  = cap.get(cv2.CAP_PROP_FPS) or 30
frame_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
ms_por_frame = 1000 / fps_video

cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps_video * SEGUNDOS_SALTAR))

out = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps_video,
    (frame_w, frame_h)
)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Procesando {total_frames} frames -> {OUTPUT_PATH}")
print("Esto puede tardar unos minutos, espera...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    now = time.time()

    # solo inferencia cada FRAME_SKIP frames, pero dibuja cached en todos
    if frame_count % FRAME_SKIP == 0:
        resultados = model.track(
            frame,
            persist=True,
            classes=list(CLASES_INTERES.keys()),
            conf=0.35,
            verbose=False
        )

        ids_en_frame = set()

        if resultados[0].boxes.id is not None:
            boxes  = resultados[0].boxes.xyxy.cpu().numpy()
            ids    = resultados[0].boxes.id.cpu().numpy().astype(int)
            clases = resultados[0].boxes.cls.cpu().numpy().astype(int)

            for box, obj_id, clase_id in zip(boxes, ids, clases):
                if clase_id not in CLASES_INTERES:
                    continue

                clase = CLASES_INTERES[clase_id]
                cx, cy = centroide(box)
                ids_en_frame.add(obj_id)

                historial_pos[obj_id].append((cx, cy))
                if len(historial_pos[obj_id]) > 30:
                    historial_pos[obj_id].pop(0)

                quieto     = esta_quieto(historial_pos[obj_id])
                seg_quieto = 0.0

                if quieto:
                    if obj_id not in primer_quieto:
                        primer_quieto[obj_id] = now
                    seg_quieto = now - primer_quieto[obj_id]

                    ultimo_aviso  = ids_avisados.get(obj_id, 0)
                    tiempo_global = now - ultimo_evento_global

                    if (seg_quieto >= UMBRAL_QUIETO_SEG
                            and now - ultimo_aviso > COOLDOWN_EVENTO
                            and tiempo_global > COOLDOWN_EVENTO):
                        mandar_evento(obj_id, clase, cx, cy, seg_quieto)
                        ids_avisados[obj_id]  = now
                        ultimo_evento_global  = now
                else:
                    primer_quieto.pop(obj_id, None)

                ultimo_boxes[obj_id] = (box, clase, cx, cy, quieto, seg_quieto)

            for vid in set(historial_pos) - ids_en_frame:
                historial_pos.pop(vid, None)
                primer_quieto.pop(vid, None)
                ultimo_boxes.pop(vid, None)

    # dibuja los últimos boxes conocidos en cada frame
    for obj_id, (box, clase, cx, cy, quieto, seg_quieto) in ultimo_boxes.items():
        dibujar_objeto(frame, obj_id, clase, box, cx, cy, quieto, seg_quieto)

    cv2.putText(frame, f"Eventos: {conteo_quietos}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_HUD, 2)

    out.write(frame)

    if frame_count % 100 == 0:
        pct = int(100 * frame_count / max(total_frames, 1))
        print(f"  {pct}% ({frame_count}/{total_frames} frames)")

cap.release()
out.release()
if kafka_ok:
    producer.close()

print(f"\nListo. Video guardado en: {OUTPUT_PATH}")
print(f"Eventos enviados a Kafka: {conteo_quietos}")
