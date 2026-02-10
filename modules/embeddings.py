import os
import numpy as np

# directorio donde se guardan los embeddings
EMBEDDINGS_DIR = "modules/Saved"

# Cargar todos los embeddings guardados desde el directorio
def load_embeddings():
    database = {}
    if not os.path.exists(EMBEDDINGS_DIR):
        return database
# recorrer todos los archivos .npy en el directorio
    for file in os.listdir(EMBEDDINGS_DIR):
        if file.endswith(".npy"):
            name = file.replace(".npy", "")
            database[name] = np.load(
                os.path.join(EMBEDDINGS_DIR, file)
            )
    return database

# Guardar un nuevo embedding en el directorio
def save_embedding(name, embedding):
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    path = os.path.join(EMBEDDINGS_DIR, f"{name}.npy")
    np.save(path, embedding)
