import cv2
from insightface.app import FaceAnalysis

# Inicializar modelo
app = FaceAnalysis(
    name="buffalo_l",      # Modelo estándar, estable
    providers=["CPUExecutionProvider"]  # luego lo pasamos a GPU
)

app.prepare(ctx_id=0, det_size=(640, 640))

print("Modelo ArcFace cargado correctamente")


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)

    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Mostrar score
        cv2.putText(
            frame,
            f"{face.det_score:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
        print(face.embedding.shape)


    cv2.imshow("ArcFace - Deteccion en tiempo real", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
