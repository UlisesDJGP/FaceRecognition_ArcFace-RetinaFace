import logging
import os
import sys

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np

from recognition_kernel import FaceExtractor
from storage import LocalStorage

# ─── Logging ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("face-register-api")

# ─── FastAPI App ────────────────────────────────────────────
app = FastAPI(
    title="Face Registration API",
    description="Backend para registro facial con InsightFace (ArcFace + RetinaFace)",
    version="1.0.0",
)

# Manejo de CORS para permitir peticiones desde el frontend HTML local
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Componentes Core ──────────────────────────────────────
logger.info("Cargando modelo InsightFace (buffalo_l) ...")
extractor = FaceExtractor()
logger.info("Modelo InsightFace cargado exitosamente.")

storage = LocalStorage(base_dir="/home/ulises/Documentos/FaceRecognition/modules/Saved")
logger.info("Storage local inicializado en: %s", os.path.abspath(storage.base_dir))


# ─── Endpoints ─────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "online", "service": "Face Registration API v1.0"}


@app.get("/api/v1/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": extractor.app is not None,
        "storage_path": os.path.abspath(storage.base_dir),
    }

@app.post("/api/v1/detect")
async def detect_face(file: UploadFile = File(...)):
    """
    Endpoint liviano de detección en tiempo real.
    Solo detecta y devuelve info del rostro — NO guarda nada.
    El frontend lo llama cada 500ms para mostrar el bounding box y el score.
    """
    image_bytes = await file.read()
    np_img = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if image is None:
        return {"face_found": False, "status": "Sin imagen", "det_score": 0.0, "bbox": None}

    try:
        faces = extractor.app.get(image)
    except Exception as e:
        logger.error("Error en detección rápida: %s", e)
        return {"face_found": False, "status": "Error de detección", "det_score": 0.0, "bbox": None}

    if len(faces) == 0:
        return {"face_found": False, "status": "Sin rostro", "det_score": 0.0, "bbox": None}

    if len(faces) > 1:
        return {"face_found": False, "status": "Múltiples rostros", "det_score": 0.0, "bbox": None}

    face = faces[0]
    det_score = float(face.det_score) if hasattr(face, "det_score") else 0.0
    bbox = [float(x) for x in face.bbox]  # [x1, y1, x2, y2]

    h, w = image.shape[:2]
    if det_score >= 0.75:
        status = f"Capturando... ✓ ({det_score:.2f})"
        good = True
    else:
        status = f"Baja calidad ({det_score:.2f})"
        good = False

    return {
        "face_found": True,
        "good_quality": good,
        "det_score": det_score,
        "status": status,
        "bbox": bbox,
        "img_width": w,
        "img_height": h,
    }


@app.post("/api/v1/register")
async def register_student(
    nombre: str = Form(...),
    matricula: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    files: list[UploadFile] = File(...)
):
    logger.info("Registro recibido — Matricula: %s, Nombre: %s, %d archivos", matricula, nombre, len(files))

    images = []
    for file in files:
        image_bytes = await file.read()
        np_img = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if image is not None:
            images.append(image)

    if not images:
        logger.warning("No se recibieron imágenes válidas para %s", matricula)
        raise HTTPException(status_code=400, detail="Los archivos enviados no contienen imágenes válidas.")

    logger.info("Se decodificaron %d frames para la ráfaga.", len(images))

    # Extraer características faciales promediando la ráfaga
    success, embedding, msg = extractor.extract_features_burst(images)

    if not success:
        logger.warning("Extracción fallida para %s: %s", matricula, msg)
        raise HTTPException(status_code=400, detail=msg)

    logger.info(
        "Matriz facial extraída para %s — shape: %s",
        matricula,
        embedding.shape,
    )

    # Guardar datos localmente (Solo el .npy, descartamos la foto en .jpg como solicitaste)
    saved = storage.save_student(matricula, nombre, email, password, embedding)

    if not saved:
        logger.error("Error fatal al guardar datos para %s", matricula)
        raise HTTPException(status_code=500, detail="Error interno al guardar los datos.")

    logger.info("✅ Registro completado para %s (%s)", matricula, nombre)
    return {
        "status": "success",
        "message": f"Usuario {nombre} registrado exitosamente. Paquete de {embedding.shape[0]} vectores faciales guardado.",
    }


# ─── Punto de entrada ─────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
