#!/usr/bin/env bash

# Detectar dinámicamente la versión de Python que vive en el venv
PY_VER=$("$PWD/venv/bin/python" -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')")
NVIDIA_DIR="$PWD/venv/lib/$PY_VER/site-packages/nvidia"

# Agregamos automáticamente todas las subcarpetas iterables de nvidia/lib al LD_LIBRARY_PATH
for dir in "$NVIDIA_DIR"/*/lib; do
    if [ -d "$dir" ]; then
        export LD_LIBRARY_PATH="$dir:$LD_LIBRARY_PATH"
    fi
done

echo "🔧 Entorno GPU configurado. Arrancando modelo FaceRecognition..."
export USE_GPU=1
./venv/bin/python main.py
