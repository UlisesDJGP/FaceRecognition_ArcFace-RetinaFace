import cv2
import time

from modules import load_face_model, open_camera, save_embedding

PERSON_NAME = input("Nombre de la persona: ")
SAMPLES = 20
MIN_DET_SCORE = 0.75
CAPTURE_DELAY = 0.5  # segundos entre capturas

app = load_face_model()
cap = open_camera()

count = 0
last_capture_time = 0

print("Iniciando registro facial...")
print("Asegúrate de estar bien iluminado y solo frente a la cámara")

while count < SAMPLES:

    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)

    status_text = "Buscando rostro..."
    color = (0, 0, 255)

    if len(faces) == 1:

        det_score = faces[0].det_score

        if det_score >= MIN_DET_SCORE:

            current_time = time.time()

            if current_time - last_capture_time > CAPTURE_DELAY:

                embedding = faces[0].embedding
                save_embedding(PERSON_NAME, embedding)

                count += 1
                last_capture_time = current_time

                print(f"Sample {count}/{SAMPLES}")

            status_text = f"Capturando... {count}/{SAMPLES}"
            color = (0, 255, 0)

        else:
            status_text = f"Baja calidad ({det_score:.2f})"
            color = (0, 165, 255)  # naranja

    elif len(faces) > 1:
        status_text = "Multiples rostros"
        color = (0, 0, 255)

    else:
        status_text = "Sin rostro"
        color = (0, 0, 255)

    # Mostrar estado en pantalla
    cv2.putText(
        frame,
        status_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )

    cv2.imshow("Registro Facial", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

print(f"Registro completado para {PERSON_NAME}")