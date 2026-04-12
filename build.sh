#!/usr/bin/env bash
set -e

# ============================================================
# build.sh — Compilación portable para SRF_AR_System
# GPU Auto-Detect | Multi-Distro | Sin hardcodeos
# Compatible con: Arch, Garuda, Ubuntu, Fedora, Debian, etc.
# ============================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  SRF_AR_System — Build Script"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# --- Verificar entorno virtual --------------------------------
if [ ! -f "$PWD/venv/bin/python" ]; then
    echo ""
    echo "❌  ERROR: Entorno virtual no encontrado en ./venv/"
    echo "    Créalo primero:"
    echo "      python -m venv venv"
    echo "      source venv/bin/activate"
    echo "      pip install -r requeriments.txt"
    exit 1
fi

# --- Verificar CUDA Toolkit ----------------------------------
if ! command -v nvcc &> /dev/null; then
    echo ""
    echo "❌  ERROR: nvcc no encontrado. Instala CUDA Toolkit:"
    echo "    Arch/Garuda : sudo pacman -S cuda"
    echo "    Ubuntu      : sudo apt install nvidia-cuda-toolkit"
    echo "    Fedora      : sudo dnf install cuda"
    exit 1
fi

# --- Verificar CMake -----------------------------------------
if ! command -v cmake &> /dev/null; then
    echo ""
    echo "❌  ERROR: cmake no encontrado."
    echo "    Arch/Garuda : sudo pacman -S cmake"
    echo "    Ubuntu      : sudo apt install cmake"
    echo "    Fedora      : sudo dnf install cmake"
    exit 1
fi

# --- Mostrar info del sistema --------------------------------
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "No detectada (nvidia-smi no disponible)")
GPU_ARCH=$(nvcc --version | grep "release" | awk '{print $NF}' | tr -d ',')
PY_VER=$("$PWD/venv/bin/python" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
CORES=$(nproc)

echo ""
echo "  🖥️  GPU        : $GPU_NAME"
echo "  🔧 CUDA        : $GPU_ARCH (arquitectura detectada automáticamente)"
echo "  🐍 Python venv : $PY_VER"
echo "  ⚙️  CPU cores   : $CORES (compilación paralela)"
echo ""

# --- Compilar ------------------------------------------------
mkdir -p build
cd build

echo "🔨 Configurando CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

echo ""
echo "🔨 Compilando con $CORES núcleos en paralelo..."
make -j"$CORES"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✅ Compilación exitosa"
echo "  📦 Librerías generadas en: ./build/"
echo "     - libsrf_onnx.so   (ArcFace ONNX — optimizado para tu GPU)"
echo "     - libsrf_bridge.so (CUDA Kernel)"
echo "     - libcamera.so     (Captura V4L2)"
echo ""
echo "  Para ejecutar con GPU:"
echo "    bash run_gpu.sh"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
