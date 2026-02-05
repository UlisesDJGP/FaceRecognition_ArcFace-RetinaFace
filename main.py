import cv2
import time
from modules import (
    load_face_model,
    open_camera,
    load_embeddings,
    save_embedding,
    recognize_face
)

# ajustes de la cámara

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 360
PROCESS_EVERY_N_FRAMES = 5
FACE_DISTANCE_THRESHOLD = 50  # píxeles



# esta función verifica si una cara detectada ya está siendo rastreada
def is_same_face(bbox, tracked_faces, threshold=50):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    for face in tracked_faces:
        fx, fy = face["center"]
        distance = ((cx - fx) ** 2 + (cy - fy) ** 2) ** 0.5
        if distance < threshold:
            return face

    return None


#Inicializar el modelo de reconocimiento facial 
app = load_face_model()
print(" Modelo ArcFace (CPU) cargado correctamente")



#se cargan los embeddings guardados
database = load_embeddings()
print(f" Embeddings cargados: {list(database.keys())}")



# Inicializar la cámara
cap = open_camera(
    camera_index=CAMERA_INDEX,
    width=FRAME_WIDTH,
    height=FRAME_HEIGHT
)

# Variables para el procesamiento de frames y el rastreo de caras
frame_count = 0
faces = []
tracked_faces = []


#contador de fps
fps = 0.0
frame_counter = 0
start_time = time.time()



# Bucle principal de procesamiento de video
while True:
    ret, frame = cap.read()
    if not ret:
        print(" Error al leer frame")
        break

    frame_count += 1
    frame_counter += 1

    small_frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

# Procesar cada N frames para mejorar el rendimiento

    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
        faces = app.get(small_frame)

    new_tracked_faces = []

    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)
        bbox = (x1, y1, x2, y2)

        existing = is_same_face(bbox, tracked_faces, FACE_DISTANCE_THRESHOLD)

        if existing:
            name = existing["name"]
            score = existing["score"]
            embedding = existing["embedding"]
        else:
            embedding = face.embedding
            name, score = recognize_face(embedding, database)

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        new_tracked_faces.append({
            "bbox": bbox,
            "center": (cx, cy),
            "embedding": embedding,
            "name": name,
            "score": score
        })

        color = (0, 255, 0) if name != "Desconocido" else (0, 0, 255)

        cv2.rectangle(
            small_frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.putText(
            small_frame,
            f"{name} ({score:.2f})",
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    tracked_faces = new_tracked_faces


    # actualizar y calcular fps
    elapsed_time = time.time() - start_time
    if elapsed_time >= 1.0:
        fps = frame_counter / elapsed_time
        frame_counter = 0
        start_time = time.time()

   
    # mostrar fps en la pantalla
    cv2.putText(
        small_frame,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.imshow("ArcFace CPU - Tiempo Real", small_frame)

    
    # Manejo de teclas para guardar embeddings o salir
    key = cv2.waitKey(1) & 0xFF

    # Guardar embedding
    if key == ord("s") and tracked_faces:
        person_name = input(" Nombre de la persona: ")
        save_embedding(person_name, tracked_faces[0]["embedding"])
        database = load_embeddings()
        print(f" Embedding guardado para: {person_name}")

    # Salir
    if key == 27:
        break



# Liberar recursos
cap.release()
cv2.destroyAllWindows()
print(" Programa cerrado correctamente")
