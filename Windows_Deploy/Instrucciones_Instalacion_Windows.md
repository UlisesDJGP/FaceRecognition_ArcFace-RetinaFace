# Guía Experta de Despliegue en Windows (Face Recognition 4K Double Buffer)

Esta carpeta (`Windows_Deploy`) contiene el código de producción sincronizado con la versión de Linux. Incluye todas las optimizaciones implementadas durante la reestructuración: Thermal Throttling, validación TCP de cámara IP, fallback automático a webcam local, protección contra embeddings corruptos, y la estrategia de Doble Buffer 4K.

> **Nota:** La versión de Windows utiliza InsightFace Python puro para la extracción de embeddings ArcFace. La versión de Linux usa un motor C++ ONNX Runtime nativo (FFI) que es ~20-30% más rápido. Ambos producen vectores de 512 dimensiones idénticos y son compatibles entre sí (los archivos `.npy` se pueden compartir).

## Requisitos Previos y Entorno

### 1. Python
Instala **Python 3.10 o superior**. Durante la instalación **marca "Add Python to PATH"**.

### 2. NVIDIA CUDA (Solo si usarás GPU)
- Descarga e instala el **NVIDIA CUDA Toolkit 12.x** para Windows.
- Opcionalmente instala **cuDNN** compatible con CUDA 12 copiando los DLLs a `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin\`.

### 3. Visual Studio C++ Build Tools (Solo si InsightFace falla al instalar)
Si `pip install insightface` lanza errores de compilación (letras rojas de Cython), instala "Visual Studio C++ Build Tools" marcando "Desarrollo para el escritorio con C++".

---

## Montaje del Proyecto

Abre **PowerShell** o **CMD** y navega a la carpeta `Windows_Deploy`:

```bat
:: 1. Crear el entorno virtual
python -m venv venv

:: 2. Activar el entorno
venv\Scripts\activate

:: 3. Instalar dependencias
pip install -r requeriments.txt

:: 4. (Solo para GPU) Instalar ONNX Runtime GPU
pip install onnxruntime-gpu
```

---

## Ejecución

### Registrar Usuarios
```bat
python register_face.py
```
El sistema te pedirá el nombre de la persona. Asegúrate de estar bien iluminado y mueve la cabeza lentamente. Capturará 20 muestras.

> **Tip:** Para máxima precisión, registra cada persona **dos veces**: una sin lentes y otra con lentes (el sistema apilará todos los vectores automáticamente).

### Monitoreo Biométrico (Sistema Principal)
```bat
run_gpu.bat
```
Esto arranca `main.py` con `USE_GPU=1` habilitado.

---

## Características Sincronizadas con Linux

| Característica | Estado |
|---|---|
| Doble Buffer 4K (detección rápida + reconocimiento UHD) | ✅ |
| Thermal Throttling (protección GPU a 80°C) | ✅ |
| HUD visual de alerta térmica en pantalla | ✅ |
| Validación TCP de cámara IP (evita freeze de 12s) | ✅ |
| Fallback automático a webcam local (DirectShow) | ✅ |
| Protección contra embeddings nulos | ✅ |
| Protección contra archivos .npy corruptos | ✅ |
| Ventana deslizante de votación (anti-parpadeo) | ✅ |
| Tracking multi-rostro con IDs persistentes | ✅ |
| Cooldown de asistencia (evita registros duplicados) | ✅ |
| Motor C++ ONNX Runtime nativo (FFI) | ✅ (requiere compilar con `build_kernel.bat`) |

---

## Compilación del Motor C++ ONNX (Opcional pero Recomendado)

El motor C++ nativo acelera la extracción de embeddings ~20-30%. Si no lo compilas, el sistema funciona igual usando InsightFace Python puro.

**Requisitos:**
- CMake 3.18+ ([descargar](https://cmake.org/download/))
- Visual Studio C++ Build Tools (con "Desarrollo para escritorio con C++")

```bat
:: Compilar la DLL
build_kernel.bat
```

Esto genera `build\Release\srf_onnx.dll`. El sistema la detecta automáticamente al arrancar.

---

## Diferencias con Linux

1. **Cámara:** Windows usa `cv2.CAP_DSHOW` (DirectShow) en lugar de `cv2.CAP_V4L2`. No necesita ajustes de saturación V4L2.
2. **Motor C++:** Windows compila como `.dll`, Linux como `.so`. El código fuente (`srf_onnx_bridge.cpp`) es idéntico.
3. **nvidia-smi:** En Windows se ejecuta con `CREATE_NO_WINDOW` para evitar que aparezca una ventana CMD emergente cada 30 frames.
4. **Fallback inteligente:** Si la DLL no está compilada, el sistema cae automáticamente a InsightFace Python puro sin errores.

---

## Estructura de Archivos

```
Windows_Deploy/
├── main.py                 # Programa principal (monitoreo + asistencia)
├── register_face.py        # Registro de nuevos usuarios
├── run_gpu.bat             # Lanzador con GPU habilitada
├── build_kernel.bat        # Compilador del motor C++ ONNX
├── CMakeLists.txt          # Configuración de compilación C++
├── requeriments.txt        # Dependencias pip
├── CUDA/
│   └── srf_onnx_bridge.cpp # Motor C++ ONNX (código fuente compartido con Linux)
├── modules/
│   ├── __init__.py         # Exports del paquete
│   ├── kernel_ffi.py       # Puente FFI Python↔C++ (multiplataforma)
│   ├── camera.py           # Captura con TCP + DirectShow fallback
│   ├── recognizer.py       # Doble Buffer + ArcFace (C++ nativo o Python fallback)
│   ├── embeddings.py       # Persistencia .npy con protección anti-corrupción
│   ├── attendance.py       # Registro CSV con horario académico
│   └── logger.py           # Log de eventos con timestamp
└── Instrucciones_Instalacion_Windows.md  # Este archivo
```
