cmake_minimum_required(VERSION 3.18)
project(FaceRecognition_SRF LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# "native": nvcc detecta automáticamente la GPU del sistema al compilar.
# (sm_75 para RTX 20xx/T1000, sm_86 para RTX 30xx, sm_89 para RTX 40xx, sm_100 para RTX 50xx)
# El usuario obtiene el binario óptimo para su GPU sin editar nada.
set(CMAKE_CUDA_ARCHITECTURES native)

# ============================================================
# ONNX Runtime: Detectar dinámicamente desde el intérprete Python del venv
# Compatible con cualquier versión de Python (3.11, 3.12, 3.13, 3.14...)
# ============================================================

# Detectar intérprete de Python del venv
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Obtener la ruta real del paquete onnxruntime instalado en el venv
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c
    "import onnxruntime, os; print(os.path.dirname(onnxruntime.__file__))"
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE ONNXRUNTIME_PKG_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)

set(ONNX_CAPI_DIR "${ONNXRUNTIME_PKG_DIR}/capi")
message(STATUS "[SRF] ONNX Runtime detectado en: ${ONNX_CAPI_DIR}")

if(NOT EXISTS "${ONNX_CAPI_DIR}")
    message(FATAL_ERROR "No se encontró ONNX Runtime en ${ONNX_CAPI_DIR}. Asegúrate de tener el venv con onnxruntime instalado.")
endif()

# FetchContent para obtener los headers C++ de ONNX Runtime
include(FetchContent)
FetchContent_Declare(
    onnxruntime_headers
    URL https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-x64-gpu-1.18.0.tgz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(onnxruntime_headers)

# ============================================================
# Target: srf_onnx (Motor C++ ONNX Bridge como DLL)
# ============================================================

add_library(srf_onnx SHARED CUDA/srf_onnx_bridge.cpp)

target_include_directories(srf_onnx PRIVATE
    "${onnxruntime_headers_SOURCE_DIR}/include"
)

# Enlazar con onnxruntime.dll del venv
target_link_directories(srf_onnx PRIVATE "${ONNX_CAPI_DIR}")
target_link_libraries(srf_onnx PRIVATE onnxruntime)

# Copiar la DLL de ONNX Runtime al directorio de build para que se encuentre en runtime
add_custom_command(TARGET srf_onnx POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNX_CAPI_DIR}/onnxruntime.dll"
        "$<TARGET_FILE_DIR:srf_onnx>"
    COMMENT "Copiando onnxruntime.dll al directorio de build..."
)
