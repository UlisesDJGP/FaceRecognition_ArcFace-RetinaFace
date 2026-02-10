import pickle
import os
import numpy as np

# Archivo donde se guardan los embeddings
EMBEDDINGS_FILE = "faces.pkl"

# Cargar todos los embeddings guardados desde el archivo
def load_embeddings():
    if not os.path.exists(EMBEDDINGS_FILE):
        return {}

    with open(EMBEDDINGS_FILE, "rb") as f:
        return pickle.load(f)

# Guardar un nuevo embedding en el archivo
def save_embedding(name, embedding):
    data = load_embeddings()

    if name not in data:
        data[name] = []

    data[name].append(embedding)
# Guardar de nuevo todos los embeddings en el archivo
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(data, f)
