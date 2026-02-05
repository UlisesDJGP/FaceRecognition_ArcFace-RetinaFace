import numpy as np
from insightface.app import FaceAnalysis


# Esta funcion carga el modelo de reconocimiento facial ArcFace

def load_face_model():
    app = FaceAnalysis(
        name="buffalo_s",
        providers=["CPUExecutionProvider"]
    )
    app.prepare(ctx_id=-1, det_size=(192, 192)) #resolucion de la camara
    return app


# Este funcion recibe un embedding y una base de datos de embeddings guardados
# Devuelve el nombre de la persona reconocida y el puntaje de similitud


def recognize_face(embedding, database, threshold=0.6):
    if not database:
        return "Desconocido", 0.0

    best_match = "Desconocido"
    best_score = -1

    for name, saved_embedding in database.items():
        score = np.dot(embedding, saved_embedding) / (
            np.linalg.norm(embedding) * np.linalg.norm(saved_embedding)
        )

        if score > best_score:
            best_score = score
            best_match = name

    if best_score < threshold:
        return "Desconocido", best_score

    return best_match, best_score
