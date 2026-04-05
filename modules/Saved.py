import pickle
import os
import numpy as np

EMBEDDINGS_FILE = "embeddings.pkl"

MAX_EMBEDDINGS = 30
SIMILARITY_THRESHOLD = 0.97


def load_embeddings():
    if not os.path.exists(EMBEDDINGS_FILE):
        return {}

    with open(EMBEDDINGS_FILE, "rb") as f:
        return pickle.load(f)


def save_embeddings(database):
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(database, f)


# UTILIDADES


def normalize(embedding):
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm


def cosine_similarity(e1, e2):
    return np.dot(e1, e2)


def is_duplicate(new_embedding, existing_embeddings):
    for emb in existing_embeddings:
        if cosine_similarity(new_embedding, emb) > SIMILARITY_THRESHOLD:
            return True
    return False



# FUNCIÓN PRINCIPAL

def save_embedding(name, embedding):

    database = load_embeddings()

    # Normalizar embedding
    embedding = normalize(embedding)

    # Crear entrada si no existe
    if name not in database:
        database[name] = []

    # Evitar duplicados
    if is_duplicate(embedding, database[name]):
        print(f"[INFO] Embedding duplicado ignorado para {name}")
        return

    # Agregar embedding
    database[name].append(embedding)

    # Limitar cantidad
    if len(database[name]) > MAX_EMBEDDINGS:
        database[name].pop(0)  # elimina el más viejo

    # Guardar base
    save_embeddings(database)

    print(f"[OK] Embedding guardado para {name} ({len(database[name])}/{MAX_EMBEDDINGS})")S