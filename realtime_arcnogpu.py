import cv2
from insightface.app import FaceAnalysis

# CONFIGURACIÓN GENERAL PARA LA CAPTURA

CAMERA_INDEX = 0           # 0 = webcam por defecto
FRAME_WIDTH = 640          # Resolución baja para pc sin gpu
FRAME_HEIGHT = 360
PROCESS_EVERY_N_FRAMES = 3 # Procesa 1 de cada N frames


# INICIALIZAR MODELO (CPU)
app = FaceAnalysis(
    name="buffalo_s",                 # Modelo ligero
    providers=["CPUExecutionProvider"]
)

# ctx_id = -1 fuerza CPU
app.prepare(ctx_id=-1, det_size=(320, 320))

print(" ArcFace (CPU) cargado correctamente")


# INICIAR CÁMARA
cap = cv2.VideoCapture(CAMERA_INDEX)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap.isOpened():
    print(" No se pudo abrir la cámara")
    exit()

frame_count = 0

# LOOP PRINCIPAL


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Reducir resolución para el modelo
    small_frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    faces = []

    # Procesar solamente ciertos frames
    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
        faces = app.get(small_frame)

    # Dibujar resultados
    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)

        cv2.rectangle(
            small_frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            small_frame,
            f"score: {face.det_score:.2f}",
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

        # embedding

    cv2.imshow("ArcFace CPU - Tiempo Real", small_frame)

    # ESC para salir
    if cv2.waitKey(1) & 0xFF == 27:
        break


# LIMPIEZA


cap.release()
cv2.destroyAllWindows()
