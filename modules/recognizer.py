import numpy as np
from insightface.app import FaceAnalysis


# =========================
# CARGA DEL MODELO
# =========================

def load_face_model():
    app = FaceAnalysis(
        name="buffalo_s",
        providers=["CPUExecutionProvider"]
    )
    app.prepare(ctx_id=-1, det_size=(192, 192))
    return app


# =========================
# UTILIDADES
# =========================

def normalize(embedding):
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm


def cosine_similarity(e1, e2):
    return np.dot(e1, e2)  # ya están normalizados


# =========================
# RECONOCIMIENTO
# =========================

def recognize_face(embedding, database, threshold=0.75):

    if not database:
        return "Desconocido", 0.0

    # Normalizar embedding de entrada
    embedding = normalize(embedding)

    best_match = "Desconocido"
    best_score = -1.0

    for name, saved_embeddings in database.items():

        # asegurar lista
        if not isinstance(saved_embeddings, list):
            saved_embeddings = [saved_embeddings]

        for saved_embedding in saved_embeddings:

            saved_embedding = np.array(saved_embedding).flatten()
            saved_embedding = normalize(saved_embedding)

            score = cosine_similarity(embedding, saved_embedding)

            # DEBUG 
            #print(f"[DEBUG] Comparando con {name}: {score:.4f}")

            if score > best_score:
                best_score = score
                best_match = name

    # Decisión final
    if best_score < threshold:
        return "Desconocido", best_score

    return best_match, best_score