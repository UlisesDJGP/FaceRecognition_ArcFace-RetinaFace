import os
import numpy as np
from insightface.app import FaceAnalysis
from insightface.app.common import Face

# =========================
# CARGA DEL MODELO
# =========================

def load_face_model(use_gpu=None):
    if use_gpu is None:
        use_gpu = os.getenv("USE_GPU", "0") == "1"

    providers = ["CUDAExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
    ctx_id = 0 if use_gpu else -1

    app = FaceAnalysis(
        name="buffalo_s",
        providers=providers,
        allowed_modules=['detection', 'recognition']
    )
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    return app

# =========================
# DOBLE BUFFER 4K 
# =========================

def detect_faces_4k_double_buffer(app, original_frame, small_frame):
    """
    Estrategia de Doble Buffer Avanzada
    Mapeo de Inferencia entre un frame reducido y un frame Ultra HD
    """
    # Deteccion rapida en el frame pequeño
    bboxes, kpss = app.det_model.detect(small_frame, max_num=0, metric='default')
    
    if bboxes.shape[0] == 0:
        return []
        
    # Calculo Dinámico de la Razón de Crecimiento
    ratio_y = original_frame.shape[0] / small_frame.shape[0]
    ratio_x = original_frame.shape[1] / small_frame.shape[1]
    
    faces = []
    
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i, 0:4].copy()
        det_score = bboxes[i, 4]
        kps = None
        if kpss is not None:
            kps = kpss[i].copy()
            
        # Mapeo Inverso
        bbox[0] *= ratio_x
        bbox[1] *= ratio_y
        bbox[2] *= ratio_x
        bbox[3] *= ratio_y
        
        if kps is not None:
            kps[:, 0] *= ratio_x
            kps[:, 1] *= ratio_y
            
        face = Face(bbox=bbox, kps=kps, det_score=det_score)
        
        # Importar dinámicamente nuestra FFI (Edge Native C++)
        from modules.kernel_ffi import extract_embedding_srf, srf_lib
        from insightface.utils import face_align
        
        if kps is not None:
            aligned_crop = face_align.norm_crop(original_frame, landmark=face.kps, image_size=112)
            
            if srf_lib is not None:
                # PATH RÁPIDO: Motor C++ ONNX nativo
                try:
                    embedding_512 = extract_embedding_srf(aligned_crop)
                    face.embedding = embedding_512
                except Exception as e:
                    print(f"[FFI Error] C++ falló la decodificación ONNX: {e}")
                    # Fallback a InsightFace Python puro
                    if 'recognition' in app.models:
                        app.models['recognition'].get(original_frame, face)
            else:
                # PATH SEGURO: InsightFace Python puro (si la DLL/SO no se compiló)
                if 'recognition' in app.models:
                    app.models['recognition'].get(original_frame, face)
                
        faces.append(face)
        
    return faces

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

        if isinstance(saved_embeddings, list):
            saved_embeddings = np.array(saved_embeddings)

        if saved_embeddings.ndim == 1:
            saved_embeddings = np.expand_dims(saved_embeddings, axis=0)

        for saved_embedding in saved_embeddings:

            saved_embedding = saved_embedding.flatten()
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