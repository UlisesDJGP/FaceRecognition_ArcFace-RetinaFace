#!/usr/bin/env bash

# Ruta base de las librerías Nvidia instaladas en el VENV
NVIDIA_DIR="$PWD/venv/lib/python3.14/site-packages/nvidia"

# Agregamos automáticamente todas las subcarpetas iterables de nvidia/lib al LD_LIBRARY_PATH
for dir in "$NVIDIA_DIR"/*/lib; do
    if [ -d "$dir" ]; then
        export LD_LIBRARY_PATH="$dir:$LD_LIBRARY_PATH"
    fi
done

echo "🔧 Entorno GPU configurado. Arrancando modelo FaceRecognition..."
export USE_GPU=1
./venv/bin/python main.py
