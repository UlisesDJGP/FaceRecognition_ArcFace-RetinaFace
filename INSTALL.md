# SRF_AR_System — Guía de Instalación y Ejecución

> Sistema de reconocimiento facial basado en ArcFace + RetinaFace con motor de inferencia C++/ONNX.
> Compatible con cualquier distribución Linux y Windows. La GPU se auto-detecta sin configuraciones manuales.

---

## Índice

1. [Requisitos Comunes](#1-requisitos-comunes)
2. [Arch Linux / Garuda Linux](#2-arch-linux--garuda-linux)
3. [Ubuntu / Debian / Mint](#3-ubuntu--debian--mint)
4. [Fedora / RHEL / CentOS Stream](#4-fedora--rhel--centos-stream)
5. [openSUSE](#5-opensuse)
6. [Windows 10/11](#6-windows-1011)
7. [Uso del Sistema](#7-uso-del-sistema)
8. [Solución de Problemas](#8-solución-de-problemas)

---

## 1. Requisitos Comunes

Independientemente de la distribución, necesitas:

| Requisito | Versión mínima | Notas |
|---|---|---|
| **GPU NVIDIA** | Cualquiera (GTX 900+) | AMD/Intel → solo modo CPU |
| **Driver NVIDIA** | 520+ | Para CUDA 12.x |
| **Python** | 3.11+ | El proyecto detecta la versión automáticamente |
| **CMake** | 3.18+ | Para compilar el motor C++ |
| **Git** | Cualquiera | Para clonar el repositorio |

> [!NOTE]  
> El proyecto **no tiene ningún hardcodeo de GPU**. Una RTX 5060 o una GTX 1080 funcionan sin cambiar nada — CMake detecta la arquitectura CUDA (`sm_XX`) de tu tarjeta automáticamente al compilar.

---

## 2. Arch Linux / Garuda Linux

### 2.1 Instalar dependencias del sistema

```bash
sudo pacman -S --needed \
    python \
    cmake \
    git \
    cuda \
    cudnn \
    opencv \
    nvidia-utils \
    base-devel
```

> [!TIP]
> En Garuda, `cuda` puede estar en los repositorios de Chaotic-AUR. Si no aparece, instala con: `yay -S cuda`

### 2.2 Verificar que CUDA está disponible

```bash
nvcc --version
nvidia-smi
```

### 2.3 Clonar e instalar

```bash
git clone <url-del-repo>
cd FaceRecognition

# Crear entorno virtual
python -m venv venv
source venv/bin/activate

# Instalar dependencias Python
pip install -r requeriments.txt

# Compilar el motor C++ (auto-detecta tu GPU)
bash build.sh
```

### 2.4 Ejecutar

```bash
# Modo GPU (recomendado)
bash run_gpu.sh

# Modo CPU (sin NVIDIA)
source venv/bin/activate && python main.py
```

---

## 3. Ubuntu / Debian / Mint

### 3.1 Instalar driver NVIDIA y CUDA Toolkit

```bash
# Verificar GPU detectada
ubuntu-drivers devices

# Instalar driver recomendado automáticamente
sudo ubuntu-drivers autoinstall

# Reiniciar
sudo reboot
```

Después del reinicio, instalar CUDA Toolkit:

```bash
# Ubuntu 22.04 / 24.04
sudo apt update
sudo apt install -y \
    nvidia-cuda-toolkit \
    cmake \
    build-essential \
    git \
    python3 \
    python3-pip \
    python3-venv \
    libopencv-dev
```

> [!IMPORTANT]
> Ubuntu 22.04 incluye Python **3.10** por defecto. Ubuntu 24.04 incluye **3.12**.  
> El proyecto detecta la versión automáticamente — no necesitas cambiar nada.

### 3.2 Verificar instalación

```bash
nvcc --version     # Debe mostrar la versión CUDA
nvidia-smi         # Debe mostrar tu GPU
python3 --version  # Debe ser 3.10/3.11/3.12+
```

### 3.3 Clonar e instalar

```bash
git clone <url-del-repo>
cd FaceRecognition

# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias Python
pip install -r requeriments.txt

# Compilar el motor C++ (auto-detecta tu GPU)
bash build.sh
```

### 3.4 Ejecutar

```bash
# Modo GPU
bash run_gpu.sh

# Modo CPU
source venv/bin/activate && python3 main.py
```

---

## 4. Fedora / RHEL / CentOS Stream

> [!WARNING]
> Fedora usa `/usr/lib64/` para las librerías NVIDIA en lugar de `/usr/lib/`. El proyecto ya incluye este path en `CMakeLists.txt` — no necesitas configuración adicional.

### 4.1 Habilitar repositorios RPM Fusion y CUDA

```bash
# RPM Fusion (necesario para drivers NVIDIA libres/privativos)
sudo dnf install -y \
    https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm \
    https://download1.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm

# Driver NVIDIA
sudo dnf install -y akmod-nvidia xorg-x11-drv-nvidia-cuda

# Reiniciar
sudo reboot
```

### 4.2 Instalar dependencias de desarrollo

```bash
sudo dnf install -y \
    cmake \
    gcc-c++ \
    git \
    python3 \
    python3-pip \
    python3-virtualenv \
    opencv-devel \
    cuda-toolkit
```

> [!NOTE]
> En Fedora, `nvcc` puede estar en `/usr/local/cuda/bin/`. Si no está en PATH:
> ```bash
> export PATH=/usr/local/cuda/bin:$PATH
> echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
> ```

### 4.3 Clonar e instalar

```bash
git clone <url-del-repo>
cd FaceRecognition

python3 -m venv venv
source venv/bin/activate
pip install -r requeriments.txt
bash build.sh
```

### 4.4 Ejecutar

```bash
bash run_gpu.sh
```

---

## 5. openSUSE

### 5.1 Instalar dependencias

```bash
# Tumbleweed
sudo zypper install -y \
    cmake \
    gcc-c++ \
    git \
    python3 \
    python3-pip \
    python311-virtualenv \
    opencv-devel

# Driver NVIDIA (desde repositorio NVIDIA)
sudo zypper addrepo --refresh \
    https://download.nvidia.com/opensuse/tumbleweed \
    NVIDIA
sudo zypper install -y cuda
```

### 5.2 Clonar e instalar

```bash
git clone <url-del-repo>
cd FaceRecognition

python3 -m venv venv
source venv/bin/activate
pip install -r requeriments.txt
bash build.sh
```

---

## 6. Windows 10/11

> El directorio `Windows_Deploy/` contiene la versión adaptada para Windows con soporte DirectShow para cámaras.

### 6.1 Pre-requisitos

1. **Driver NVIDIA** — Descargar de [nvidia.com/drivers](https://www.nvidia.com/Download/index.aspx)
2. **CUDA Toolkit** — Descargar de [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
3. **CMake** — Descargar de [cmake.org](https://cmake.org/download/) (marcar "Add to PATH")
4. **Visual Studio 2022** (Community) — Con el componente **"Desarrollo de escritorio con C++"**
5. **Python 3.11+** — Descargar de [python.org](https://www.python.org/downloads/)
6. **Git** — Descargar de [git-scm.com](https://git-scm.com/download/win)

### 6.2 Instalar y compilar

Abrir `cmd` o `PowerShell` como **Administrador**:

```batch
cd Windows_Deploy

:: Crear entorno virtual
python -m venv venv
venv\Scripts\activate

:: Instalar dependencias Python
pip install -r requeriments.txt

:: Compilar el motor C++ (auto-detecta tu GPU NVIDIA)
build_kernel.bat
```

### 6.3 Ejecutar

```batch
:: Modo GPU
run_gpu.bat

:: Modo CPU (sin GPU NVIDIA)
venv\Scripts\python main.py
```

---

## 7. Uso del Sistema

### Registrar una persona

```bash
# Linux
source venv/bin/activate
python register_face.py

# Windows
venv\Scripts\python register_face.py
```

El sistema capturará **20 muestras** de embeddings faciales (512D cada una) y las guardará en `modules/Saved/`.

### Ejecutar reconocimiento en tiempo real

```bash
# Linux — GPU
bash run_gpu.sh

# Linux — CPU
source venv/bin/activate && python main.py

# Windows — GPU
run_gpu.bat
```

### Verificar asistencias registradas

```bash
cat attendance.csv
```

---

## 8. Solución de Problemas

### ❌ `nvcc: command not found`

```bash
# Arch
sudo pacman -S cuda

# Ubuntu
sudo apt install nvidia-cuda-toolkit

# Fedora
sudo dnf install cuda-toolkit
export PATH=/usr/local/cuda/bin:$PATH

# Verificar
nvcc --version
```

### ❌ `libonnxruntime.so not found`

El script `build.sh` lo detecta automáticamente. Si falla, verifica que el venv está activo antes de compilar:

```bash
source venv/bin/activate
pip show onnxruntime-gpu   # Debe mostrar info del paquete
bash build.sh
```

### ❌ `CUDA error: no kernel image is available`

Tu GPU tiene una arquitectura CUDA que el `.so` precompilado no soporta. Solución: recompila desde cero:

```bash
rm -rf build/
bash build.sh   # nvcc compilará con la arquitectura correcta de tu GPU
```

### ❌ Cámara en blanco y negro (solo Linux)

Los controles V4L2 de saturación pueden estar en valores bajos. El código los restaura automáticamente, pero si persiste:

```bash
# Verificar y forzar manualmente
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --set-ctrl=saturation=64,brightness=128
```

### ❌ `USE_GPU=0` — corre por CPU aunque tengo GPU

Asegúrate de usar `run_gpu.sh` (Linux) o `run_gpu.bat` (Windows), no `python main.py` directamente.

```bash
# Linux
bash run_gpu.sh

# Verificar que el entorno GPU está activo
echo $USE_GPU   # Debe imprimir: 1
```

---

## Tabla de Compatibilidad

| Distribución | Python Default | NVML Path | Estado |
|---|---|---|---|
| **Arch Linux** | 3.14 (bleeding edge) | `/usr/lib/` | ✅ Nativo |
| **Garuda Linux** | 3.14 (Arch-based) | `/usr/lib/` | ✅ Nativo |
| **Ubuntu 24.04** | 3.12 | `/usr/lib/x86_64-linux-gnu/` | ✅ Soportado |
| **Ubuntu 22.04** | 3.10 | `/usr/lib/x86_64-linux-gnu/` | ✅ Soportado |
| **Fedora 41** | 3.13 | `/usr/lib64/` | ✅ Soportado |
| **Debian 12** | 3.11 | `/usr/lib/x86_64-linux-gnu/` | ✅ Soportado |
| **openSUSE Tumbleweed** | 3.12 | `/usr/lib64/` | ✅ Soportado |
| **Windows 10/11** | Variable | `Windows_Deploy/` | ✅ Rama separada |

---

*Generado automáticamente. Para reportar problemas, consulta los logs en `system.log` y `logs/`.*
