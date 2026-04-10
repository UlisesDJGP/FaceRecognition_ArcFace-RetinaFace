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
    if embedding is None:
        print("[Alerta] Intentando guardar un embedding vacío. Abortando persistencia.")
        return

    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    path = os.path.join(EMBEDDINGS_DIR, f"{name}.npy")
    
    if os.path.exists(path):
        try:
            existing = np.load(path)
            if existing.ndim == 1:
                existing = np.expand_dims(existing, axis=0)
            new_emb = np.expand_dims(embedding, axis=0)
            updated = np.vstack((existing, new_emb))
        except (ValueError, OSError) as e:
            print(f"[Error Persistencia] El archivo {path} estaba corrupto. Sobreescribiendo ({e}).")
            updated = np.expand_dims(embedding, axis=0)
    else:
        updated = np.expand_dims(embedding, axis=0)
        
    np.save(path, updated)
