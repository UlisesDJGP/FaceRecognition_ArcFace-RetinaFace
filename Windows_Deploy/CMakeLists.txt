cmake_minimum_required(VERSION 3.18)
project(FaceRecognition_SRF LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# ============================================================
# ONNX Runtime: Reutilizar los binarios del venv de Python
# En Windows: venv\Lib\site-packages\onnxruntime\capi\
# ============================================================

# Detectar automáticamente la ruta del venv
set(ONNX_CAPI_DIR "${CMAKE_SOURCE_DIR}/venv/Lib/site-packages/onnxruntime/capi")

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
