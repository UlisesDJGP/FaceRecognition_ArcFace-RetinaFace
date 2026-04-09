import cv2
import time
import math
import os
import subprocess

from modules import (
    load_face_model,
    open_camera,
    load_embeddings,
    recognize_face,
    detect_faces_4k_double_buffer
)

from modules.logger import log_event
from modules.attendance import register_attendance


# =========================
# CONFIG
# =========================

CAMERA_INDEX = "http://192.168.100.11:8080/video"
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
DETECTION_WIDTH = 640
DETECTION_HEIGHT = 360

PROCESS_EVERY_N_FRAMES = 1
RECOGNITION_INTERVAL = 3

FACE_DISTANCE_THRESHOLD = 50
AREA_THRESHOLD = 5000

RECOGNITION_CONFIRM_FRAMES = 5
BUFFER_TIMEOUT = 2
STATUS_DURATION = 2
COOLDOWN_SECONDS = 10

# =========================
# THERMAL CONTROL
# =========================
MAX_TEMP_THRESHOLD = 80
SAFE_TEMP_THRESHOLD = 70
NORMAL_PROCESS_N_FRAMES = 1
THROTTLE_PROCESS_N_FRAMES = 15
throttling_active = False

def get_gpu_temp():
    """Monitoreo térmico NVIDIA. nvidia-smi.exe está en PATH en Windows si los drivers están instalados."""
    try:
        res = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
            creationflags=subprocess.CREATE_NO_WINDOW  # Evita ventana CMD emergente en Windows
        )
        return int(res.decode("utf-8").strip())
    except Exception:
        return 0

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

# Inicializar el Motor C++ ONNX Bridge (si la DLL está compilada)
from modules.kernel_ffi import init_srf_engine, srf_lib
if srf_lib is not None:
    model_path = os.path.join(os.path.expanduser("~"), ".insightface", "models", "buffalo_s", "w600k_mbf.onnx")
    print("Inyectando Inteligencia de Reconocimiento a C++ (ONNX Runtime Edge)...")
    init_srf_engine(model_path)
else:
    print("[Info] Motor C++ no compilado. Usando InsightFace Python puro (funcional).")

database = load_embeddings()
print(f"Embeddings cargados: {list(database.keys())}")

cap = open_camera(CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT)

print("Sistema iniciado, esperando detección facial")


# =========================
# LOOP
# =========================

while True:

    ret, frame = cap.read()
    if not ret or frame is None:
        time.sleep(0.01)
        continue

    now = time.time()
    frame_count += 1
    frame_counter += 1

    # =========================
    # THERMAL THROTTLING CHECK
    # =========================
    if frame_count % 30 == 0:
        current_temp = get_gpu_temp()
        if USE_GPU and current_temp > 0:
            if current_temp >= MAX_TEMP_THRESHOLD and not throttling_active:
                throttling_active = True
                PROCESS_EVERY_N_FRAMES = THROTTLE_PROCESS_N_FRAMES
                print(f"[ALERTA TERMICA] GPU a {current_temp}°C. Iniciando Throttling (FPS reducidos).")
            elif current_temp <= SAFE_TEMP_THRESHOLD and throttling_active:
                throttling_active = False
                PROCESS_EVERY_N_FRAMES = NORMAL_PROCESS_N_FRAMES
                print(f"[TERMICAS ESTABLES] GPU a {current_temp}°C. Throttling desactivado.")

    small_frame = cv2.resize(frame, (DETECTION_WIDTH, DETECTION_HEIGHT))

    # DETECCION
    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
        faces = detect_faces_4k_double_buffer(app, frame, small_frame)

    new_tracked_faces = []
    available_trackers = list(tracked_faces)

    for face in faces:

        bbox = tuple(map(int, face.bbox))
        center = get_center(bbox)
        area = get_area(bbox)

        best_existing = None
        best_dist = float('inf')

        for t_face in available_trackers:
            cx, cy = center
            fx, fy = t_face["center"]
            distance = math.hypot(cx - fx, cy - fy)
            
            # Tolerancia amplia de ~350px para soportar saltos por bajos FPS tras doble detección 4K
            if distance < 400 and distance < best_dist:
                best_dist = distance
                best_existing = t_face

        if best_existing:
            face_id = best_existing["id"]
            embedding = face.embedding
            available_trackers.remove(best_existing)
        else:
            face_id = next_face_id
            next_face_id += 1
            embedding = face.embedding

        # =========================
        # BUFFER INIT
        # =========================

        if face_id not in recognition_buffer:
            recognition_buffer[face_id] = {
                "vote_history": [],
                "frames": 0,
                "last_update": now,
                "name": None,
                "confirmed": False
            }

        buffer = recognition_buffer[face_id]

        if frame_count % RECOGNITION_INTERVAL == 0:

            name, score = recognize_face(embedding, database)

            # Ventana deslizante para votos
            vote_history = buffer.setdefault("vote_history", [])
            vote_history.append(name)
            if len(vote_history) > 8:
                vote_history.pop(0)

            buffer["frames"] += 1
            buffer["last_update"] = now

            if buffer["frames"] >= RECOGNITION_CONFIRM_FRAMES:

                # Encontrar el elemento más común en la ventana reciente
                best_name = max(set(vote_history), key=vote_history.count)

                buffer["name"] = best_name
                buffer["confirmed"] = True

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

        # Mapeo Visual: Las cajas estan en UHD, pero la UI se pinta en small_frame
        ratio_x = frame.shape[1] / small_frame.shape[1]
        ratio_y = frame.shape[0] / small_frame.shape[0]
        
        sx1, sy1 = int(x1 / ratio_x), int(y1 / ratio_y)
        sx2, sy2 = int(x2 / ratio_x), int(y2 / ratio_y)

        cv2.rectangle(small_frame, (sx1, sy1), (sx2, sy2), color, 2)

        if display_name:
            cv2.putText(
                small_frame,
                display_name,
                (sx1, sy2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        cv2.putText(
            small_frame,
            status_text,
            (sx1, sy2 + 40),
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

    if throttling_active:
        cv2.putText(
            small_frame,
            "!! THROTTLING (TEMP ALTA) !!",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    cv2.imshow("Registro de Asistencia", small_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break


# =========================
# Limpieza
# =========================

cap.release()
cv2.destroyAllWindows()

print("Programa cerrado")