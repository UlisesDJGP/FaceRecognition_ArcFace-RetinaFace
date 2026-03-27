cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(SRF_AR_System LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_ARCHITECTURES 75)

# =============================================================================
# DEPENDENCIAS DEL SISTEMA
# =============================================================================
# 1. NVIDIA Stack
find_package(CUDAToolkit REQUIRED)
find_library(NVML_LIBRARY NAMES nvidia-ml nvml REQUIRED PATHS /usr/lib /usr/lib/x86_64-linux-gnu /opt/cuda/lib64)

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