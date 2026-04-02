import cv2
import time
import math
import os

from modules import (
    load_face_model,
    open_camera,
    load_embeddings,
    recognize_face
)

from modules.logger import log_event
from modules.attendance import register_attendance


# =========================
# CONFIG
# =========================

CAMERA_INDEX = 0
FRAME_WIDTH = 480
FRAME_HEIGHT = 270

PROCESS_EVERY_N_FRAMES = 5
RECOGNITION_INTERVAL = 10

FACE_DISTANCE_THRESHOLD = 50
AREA_THRESHOLD = 5000

RECOGNITION_CONFIRM_FRAMES = 5
BUFFER_TIMEOUT = 2
STATUS_DURATION = 2
COOLDOWN_SECONDS = 10


# =========================
# GPU / CPU INFO
# =========================

USE_GPU = os.getenv("USE_GPU", "0") == "1"
print(f"Modo ejecucion: {'GPU' if USE_GPU else 'CPU'}")


# =========================
# ESTADO GLOBAL
# =========================

next_face_id = 0
tracked_faces = []
recognition_buffer = {}
status_display = {}
last_seen = {}

fps = 0.0
frame_counter = 0
start_time = time.time()
frame_count = 0

faces = []  # IMPORTANTE


# =========================
# TRACKING
# =========================

def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def get_area(bbox):
    x1, y1, x2, y2 = bbox
    return abs((x2 - x1) * (y2 - y1))


def is_same_face(bbox, tracked_faces):
    cx, cy = get_center(bbox)
    area = get_area(bbox)

    for face in tracked_faces:
        fx, fy = face["center"]
        f_area = face["area"]

        distance = math.hypot(cx - fx, cy - fy)
        area_diff = abs(area - f_area)

        if distance < FACE_DISTANCE_THRESHOLD and area_diff < AREA_THRESHOLD:
            return face

    return None


# =========================
# INICIALIZACION
# =========================

print("Inicializando modelo")
app = load_face_model()  

database = load_embeddings()
print(f"Embeddings cargados: {list(database.keys())}")

cap = open_camera(CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT)

print("Sistema iniciado, esperando detección facial")


# =========================
# LOOP
# =========================

while True:

    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    frame_count += 1
    frame_counter += 1

    small_frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    # DETECCION
    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
        faces = app.get(small_frame)

    new_tracked_faces = []

    for face in faces:

        bbox = tuple(map(int, face.bbox))
        center = get_center(bbox)
        area = get_area(bbox)

        existing = is_same_face(bbox, tracked_faces)

        if existing:
            face_id = existing["id"]
            embedding = existing["embedding"]
        else:
            face_id = next_face_id
            next_face_id += 1
            embedding = face.embedding

        # =========================
        # BUFFER INIT
        # =========================

        if face_id not in recognition_buffer:
            recognition_buffer[face_id] = {
                "votes": {},
                "frames": 0,
                "last_update": now,
                "name": None,
                "confirmed": False
            }

        buffer = recognition_buffer[face_id]

        # =========================
        # RECONOCIMIENTO CONTROLADO
        # =========================

        if frame_count % RECOGNITION_INTERVAL == 0:

            name, score = recognize_face(face.embedding, database)

            votes = buffer["votes"]
            votes[name] = votes.get(name, 0) + 1

            buffer["frames"] += 1
            buffer["last_update"] = now

            # CONFIRMAR IDENTIDAD
            if buffer["frames"] >= RECOGNITION_CONFIRM_FRAMES:

                best_name = max(votes, key=votes.get)

                buffer["name"] = best_name
                buffer["confirmed"] = True

                # COOLDOWN
                if best_name != "Desconocido":

                    if best_name not in last_seen or (now - last_seen[best_name]) > COOLDOWN_SECONDS:
                        register_attendance(best_name)
                        log_event(f"Asistencia registrada: {best_name}")
                        last_seen[best_name] = now
                        status_display[face_id] = ("Asistencia ", now)
                    else:
                        status_display[face_id] = ("Reconocido", now)
                else:
                    status_display[face_id] = ("Desconocido", now)

        # =========================
        # TRACK UPDATE
        # =========================

        new_tracked_faces.append({
            "id": face_id,
            "bbox": bbox,
            "center": center,
            "area": area,
            "embedding": embedding
        })

        # =========================
        # VISUAL
        # =========================

        display_name = buffer["name"] if buffer["confirmed"] else ""
        status_text = "Detectando."

        if face_id in status_display:
            status, timestamp = status_display[face_id]

            if status == "Asistencia " and now - timestamp > STATUS_DURATION:
                status_text = "Reconocido"
            else:
                status_text = status

        color = (0, 255, 0) if display_name else (0, 0, 255)

        x1, y1, x2, y2 = bbox

        cv2.rectangle(small_frame, (x1, y1), (x2, y2), color, 2)

        if display_name:
            cv2.putText(
                small_frame,
                display_name,
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        cv2.putText(
            small_frame,
            status_text,
            (x1, y2 + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    tracked_faces = new_tracked_faces

    # =========================
    # LIMPIEZA DE BUFFERS
    # =========================

    expired_ids = []

    for fid, buf in recognition_buffer.items():
        if now - buf["last_update"] > BUFFER_TIMEOUT:
            expired_ids.append(fid)

    for fid in expired_ids:
        del recognition_buffer[fid]

    # =========================
    # FPS
    # =========================

    elapsed = time.time() - start_time

    if elapsed >= 1.0:
        fps = frame_counter / elapsed
        frame_counter = 0
        start_time = time.time()

    cv2.putText(
        small_frame,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.imshow("Registro de Asitencia", small_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break


# =========================
# Limpieza
# =========================

cap.release()
cv2.destroyAllWindows()

print("Programa cerrado")