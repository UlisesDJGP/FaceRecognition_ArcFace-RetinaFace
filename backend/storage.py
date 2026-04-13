import hashlib
import json
import logging
import os
from datetime import datetime

import numpy as np

logger = logging.getLogger("face-register-api.storage")


class LocalStorage:
    """
    Almacenamiento local de datos de estudiantes registrados.
    Estructura de archivos:
        dataset/
        ├── database.json          # Índice global de usuarios
        ├── {matricula}/
        │   ├── {matricula}_face.jpg       # Imagen original de registro
        │   └── {matricula}_embedding.npy  # Vector 512D normalizado
    """

    def __init__(self, base_dir: str = "dataset"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

        self.db_path = os.path.join(self.base_dir, "database.json")
        if not os.path.exists(self.db_path):
            with open(self.db_path, "w") as f:
                json.dump({}, f)
            logger.info("Base de datos JSON creada en %s", self.db_path)

    def _hash_password(self, password: str) -> str:
        """Hash simple con SHA-256. En producción usar bcrypt/argon2."""
        return hashlib.sha256(password.encode("utf-8")).hexdigest()

    def student_exists(self, matricula: str) -> bool:
        """Verifica si un estudiante ya fue registrado."""
        try:
            with open(self.db_path, "r") as f:
                db = json.load(f)
            return matricula in db
        except Exception:
            return False

    def save_student(
        self,
        matricula: str,
        nombre: str,
        email: str,
        password: str,
        embedding: np.ndarray,
    ) -> bool:
        try:
            # Verificar si ya existe
            if self.student_exists(matricula):
                logger.warning(
                    "La matrícula %s ya está registrada. Sobreescribiendo...",
                    matricula,
                )

            # Guardar embedding como archivo numpy
            embedding_path = os.path.join(self.base_dir, f"{matricula}.npy")
            np.save(embedding_path, embedding)
            logger.info(
                "Embedding guardado: %s (shape: %s)", embedding_path, embedding.shape
            )

            # Actualizar base de datos JSON
            with open(self.db_path, "r") as f:
                db = json.load(f)

            db[matricula] = {
                "nombre": nombre,
                "email": email,
                "password_hash": self._hash_password(password),
                "embedding_file": os.path.abspath(embedding_path),
                "embedding_dim": int(embedding.shape[0]),
                "registered_at": datetime.now().isoformat(),
            }

            with open(self.db_path, "w") as f:
                json.dump(db, f, indent=4, ensure_ascii=False)

            logger.info(
                "Base de datos actualizada — total registros: %d", len(db)
            )
            return True

        except Exception as e:
            logger.error("Error al guardar datos para %s: %s", matricula, e, exc_info=True)
            return False
