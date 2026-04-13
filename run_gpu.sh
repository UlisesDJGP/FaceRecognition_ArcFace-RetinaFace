#!/usr/bin/env bash

# Detectar el directorio real en el que se encuentra este script
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$PROJECT_DIR"

# Detectar dinámicamente la versión de Python que vive en el venv
PY_VER=$("./venv/bin/python" -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')")
NVIDIA_DIR="./venv/lib/$PY_VER/site-packages/nvidia"

# Agregamos automáticamente todas las subcarpetas iterables de nvidia/lib al LD_LIBRARY_PATH
for dir in "$NVIDIA_DIR"/*/lib; do
    if [ -d "$dir" ]; then
        export LD_LIBRARY_PATH="$dir:$LD_LIBRARY_PATH"
    fi
done

echo "🔧 Entorno GPU configurado. Arrancando modelo FaceRecognition..."
export USE_GPU=1
./venv/bin/python main.py
