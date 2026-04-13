import logging
import numpy as np
import cv2

logger = logging.getLogger("face-register-api.kernel")

# ── Mismos parámetros que register-face.py ──────────────────────────────────
MIN_DET_SCORE = 0.75   # igual que register-face.py → PERSON_NAME / MIN_DET_SCORE
TARGET_SAMPLES = 20    # misma meta de muestras que SAMPLES = 20
# ────────────────────────────────────────────────────────────────────────────


class FaceExtractor:
    """
    Motor de extracción de características faciales usando InsightFace.
    Configurado para ser IDÉNTICO a register-face.py:
      - Modelo:     buffalo_s  (mismo que w600k_mbf.onnx local)
      - Embedding:  faces[0].embedding  (raw, sin normalizar → igual que local)
      - Det score:  0.75 mínimo         (igual que MIN_DET_SCORE = 0.75)
      - Salida:     array (N, 512)       (igual que save_embedding() acumulado)
    """

    def __init__(self, model_name: str = "buffalo_s", det_size: tuple = (640, 640)):
        try:
            from insightface.app import FaceAnalysis

            self.app = FaceAnalysis(name=model_name, providers=["CPUExecutionProvider"])
            self.app.prepare(ctx_id=0, det_size=det_size)
            logger.info(
                "FaceExtractor inicializado — modelo: %s, det_size: %s",
                model_name,
                det_size,
            )
        except Exception as e:
            logger.error("Error al cargar el modelo InsightFace: %s", e)
            raise RuntimeError(f"No se pudo inicializar InsightFace: {e}") from e

    def extract_features_burst(self, images: list[np.ndarray]):
        """
        Replica exacta de la lógica de register-face.py:
          - Itera sobre cada frame de la ráfaga
          - Exige exactamente 1 rostro por frame  (igual: if len(faces) == 1)
          - Exige det_score >= 0.75               (igual: if det_score >= MIN_DET_SCORE)
          - Usa faces[0].embedding (RAW)           (igual: embedding = faces[0].embedding)
          - Apila en matrix (N, 512)               (igual: lo que save_embedding() acumula)

        Returns:
            tuple: (success: bool, embeddings_matrix: np.ndarray | None, message: str)
        """
        valid_embeddings = []

        for img in images:
            if img is None or img.size == 0:
                continue

            try:
                faces = self.app.get(img)

                # Igual que register-face.py: solo procesa si hay UN solo rostro
                if len(faces) == 1:
                    main_face = faces[0]
                    det_score = float(main_face.det_score) if hasattr(main_face, "det_score") else -1.0

                    # Mismo umbral exacto: MIN_DET_SCORE = 0.75
                    if det_score >= MIN_DET_SCORE:
                        # ¡CLAVE! Usar .embedding (raw) igual que register-face.py
                        # NO usar .normed_embedding → recognizer.py espera el raw
                        embedding = main_face.embedding
                        if embedding is not None and len(embedding) == 512:
                            valid_embeddings.append(embedding)
                            logger.debug(
                                "Sample %d — det_score: %.3f",
                                len(valid_embeddings), det_score
                            )
                    else:
                        logger.debug("Frame descartado — det_score bajo: %.3f", det_score)

                elif len(faces) > 1:
                    logger.debug("Frame descartado — múltiples rostros: %d", len(faces))
                else:
                    logger.debug("Frame descartado — sin rostro detectado.")

            except Exception as e:
                logger.error("Error procesando frame de ráfaga: %s", e)

        if len(valid_embeddings) == 0:
            return (
                False,
                None,
                "No se pudo detectar un rostro válido (score ≥ 0.75) en ninguna de las fotos. "
                "Asegúrate de buena iluminación y que solo tú estés en cámara."
            )

        # Apilar embeddings en matriz (N, 512) — IDÉNTICO a lo que save_embedding() genera
        embeddings_matrix = np.array(valid_embeddings, dtype=np.float32)  # shape: (N, 512)

        logger.info(
            "Ráfaga completa — %d/%d muestras válidas → matriz shape: %s",
            len(valid_embeddings), len(images), embeddings_matrix.shape
        )
        return (
            True,
            embeddings_matrix,
            f"Rostro indexado ({len(valid_embeddings)} muestras válidas de {len(images)} frames)."
        )
