import cv2
import time
import os

from modules import load_face_model, open_camera, save_embedding
from modules.recognizer import detect_faces_4k_double_buffer

PERSON_NAME = input("Nombre de la persona: ")
SAMPLES = 20
MIN_DET_SCORE = 0.75
CAPTURE_DELAY = 0.5  # segundos entre capturas

CAMERA_URL = "http://192.168.100.11:8080/video"
app = load_face_model()

# Inicializar Motor C++ ONNX Bridge (si la DLL está compilada)
from modules.kernel_ffi import init_srf_engine, srf_lib
if srf_lib is not None:
    model_path = os.path.join(os.path.expanduser("~"), ".insightface", "models", "buffalo_s", "w600k_mbf.onnx")
    init_srf_engine(model_path)

cap = open_camera(camera_index=CAMERA_URL, width=1920, height=1080)

count = 0
last_capture_time = 0

print("Iniciando registro facial...")
print("Asegúrate de estar bien iluminado y mueve tu cabeza lentamente (lados, arriba y abajo)")

while count < SAMPLES:

    ret, frame = cap.read()
    if not ret:
        break

    # Creamos buffer reducido para la UI y la detección rápida (evita el lag masivo)
    small_frame = cv2.resize(frame, (640, 360))
    faces = detect_faces_4k_double_buffer(app, frame, small_frame)

    status_text = "Buscando rostro..."
    color = (0, 0, 255)

    if len(faces) == 1:

        det_score = faces[0].det_score

        if det_score >= MIN_DET_SCORE:

            current_time = time.time()

            if current_time - last_capture_time > CAPTURE_DELAY:

                embedding = faces[0].embedding
                if embedding is not None:
                    save_embedding(PERSON_NAME, embedding)

                    count += 1
                    last_capture_time = current_time

                    print(f"Sample {count}/{SAMPLES}")
                else:
                    print(f"Advertencia: Rostro detectado pero no se pudo generar el embedding.")

            status_text = f"Capturando... {count}/{SAMPLES}"
            color = (0, 255, 0)
            
            # Dibujar el recuadro visual de la detección mapeando la UI
            bbox = tuple(map(int, faces[0].bbox))
            rx, ry = frame.shape[1] / 640, frame.shape[0] / 360
            sx1, sy1 = int(bbox[0]/rx), int(bbox[1]/ry)
            sx2, sy2 = int(bbox[2]/rx), int(bbox[3]/ry)
            cv2.rectangle(small_frame, (sx1, sy1), (sx2, sy2), color, 2)

        else:
            status_text = f"Baja calidad ({det_score:.2f})"
            color = (0, 165, 255)  # naranja

    elif len(faces) > 1:
        status_text = "Multiples rostros"
        color = (0, 0, 255)

    else:
        status_text = "Sin rostro"
        color = (0, 0, 255)

    # Mostrar estado en pantalla UI
    cv2.putText(
        small_frame,
        status_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )

    cv2.imshow("Registro Facial", small_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

print(f"Registro completado para {PERSON_NAME}")