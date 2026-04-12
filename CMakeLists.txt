cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(SRF_AR_System LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# "native": nvcc detecta automáticamente la GPU instalada y compila con el sm_XX exacto.
# (sm_75 para T1000/RTX 20xx, sm_86 para RTX 30xx, sm_89 para RTX 40xx, sm_100 para RTX 50xx)
# Cada usuario obtiene el binario óptimo para su GPU sin editar nada.
set(CMAKE_CUDA_ARCHITECTURES native)

# =============================================================================
# DEPENDENCIAS DEL SISTEMA
# =============================================================================
# 1. NVIDIA Stack
find_package(CUDAToolkit REQUIRED)
find_library(NVML_LIBRARY NAMES nvidia-ml nvml REQUIRED
  PATHS
    /usr/lib                    # Arch Linux, Garuda
    /usr/lib64                  # Fedora, RHEL, openSUSE
    /usr/lib/x86_64-linux-gnu   # Ubuntu, Debian
    /opt/cuda/lib64             # CUDA instalado manualmente
    /usr/local/cuda/lib64       # CUDA Toolkit alternativo
)

# 3. Python3 (para detectar la ruta de ONNX Runtime dinámicamente)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# 2. Computer Vision Stack (OpenCV para V4L2)
find_package(OpenCV REQUIRED)

# =============================================================================
# TARGET 1: KERNEL DE INFERENCIA GPU (CUDA)
# =============================================================================
add_library(srf_bridge SHARED ${CMAKE_CURRENT_SOURCE_DIR}/FaceRecognition_ArcFace-RetinaFace/CUDA/srf_bridge.cu)
target_link_libraries(srf_bridge PRIVATE CUDA::cudart ${NVML_LIBRARY})
target_compile_options(srf_bridge PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -use_fast_math>)

# =============================================================================
# TARGET 2: DRIVER DE CAPTURA V4L2 (C++)
# =============================================================================
add_library(camera SHARED ${CMAKE_CURRENT_SOURCE_DIR}/FaceRecognition_ArcFace-RetinaFace/CUDA/camera_bridge.cpp)
# Enlazamos nuestro binario con la API de C++ de OpenCV
target_link_libraries(camera PRIVATE ${OpenCV_LIBS})
target_compile_options(camera PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra>)

# =============================================================================
# TARGET 3: INFERENCIA ONNX RUNTIME C++ (ArcFace Backend)
# =============================================================================
include(FetchContent)
FetchContent_Declare(
  onnxruntime_headers
  URL https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-x64-1.18.0.tgz
)
FetchContent_MakeAvailable(onnxruntime_headers)

add_library(srf_onnx SHARED ${CMAKE_CURRENT_SOURCE_DIR}/FaceRecognition_ArcFace-RetinaFace/CUDA/srf_onnx_bridge.cpp)

# Aseguramos apuntar los headers descargados (API C++ Nativa)
target_include_directories(srf_onnx PRIVATE ${onnxruntime_headers_SOURCE_DIR}/include)

# Detectar dinámicamente la ruta del .so de ONNX Runtime desde el intérprete de Python activo.
# Compatible con cualquier versión de Python (3.11, 3.12, 3.13, 3.14...) y cualquier distro.
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c
    "import onnxruntime, os; print(os.path.dirname(onnxruntime.__file__))"
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  OUTPUT_VARIABLE ONNXRUNTIME_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)
file(GLOB ONNXRUNTIME_SO_LIST "${ONNXRUNTIME_DIR}/capi/*libonnxruntime.so*")
list(GET ONNXRUNTIME_SO_LIST 0 ONNXRUNTIME_LIB)
message(STATUS "[SRF] ONNX Runtime detectado en: ${ONNXRUNTIME_LIB}")
target_link_libraries(srf_onnx PRIVATE ${ONNXRUNTIME_LIB})
target_compile_options(srf_onnx PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra -fPIC>)