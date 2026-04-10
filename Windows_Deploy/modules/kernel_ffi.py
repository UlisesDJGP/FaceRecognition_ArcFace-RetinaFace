import ctypes
import numpy as np
import os
import sys
import cv2

# Detectar plataforma para elegir la extensión correcta
if sys.platform == "win32":
    LIB_NAME = "srf_onnx.dll"
    LIB_SEARCH_DIRS = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../build/Release")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../build/Debug")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../build")),
    ]
else:
    LIB_NAME = "libsrf_onnx.so"
    LIB_SEARCH_DIRS = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../build")),
    ]

# Buscar la librería en los directorios posibles
lib_path = None
for search_dir in LIB_SEARCH_DIRS:
    candidate = os.path.join(search_dir, LIB_NAME)
    if os.path.exists(candidate):
        lib_path = candidate
        break

if lib_path is None:
    print(f"[FFI] Advertencia: No se encontró la librería C++ ONNX ({LIB_NAME}). Asegúrese de compilar con CMake.")
    srf_lib = None
else:
    if sys.platform == "win32":
        # En Windows: añadir el directorio de la DLL al PATH para resolver dependencias (onnxruntime.dll)
        dll_dir = os.path.dirname(lib_path)
        os.add_dll_directory(dll_dir)
        
        # También añadir el directorio de ONNX Runtime del venv
        onnx_capi = os.path.abspath(os.path.join(os.path.dirname(__file__), "../venv/Lib/site-packages/onnxruntime/capi"))
        if os.path.exists(onnx_capi):
            os.add_dll_directory(onnx_capi)
        
        srf_lib = ctypes.CDLL(lib_path)
    else:
        # En Linux: Precargar libonnxruntime en el espacio global de memoria
        onnx_binary_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../venv/lib/python3.14/site-packages/onnxruntime/capi/libonnxruntime.so.1.24.4"))
        
        if os.path.exists(onnx_binary_path):
            ctypes.CDLL(onnx_binary_path, mode=os.RTLD_GLOBAL)
        else:
            print(f"[FFI] Advertencia: No se encontró ONNX base en {onnx_binary_path}")

        srf_lib = ctypes.CDLL(lib_path)

    # ---------------------------------------------------------
    # FIRMAS FUNCIONALES DE C++
    # int32_t srf_init_onnx(const char* model_path)
    srf_lib.srf_init_onnx.argtypes = [ctypes.c_char_p]
    srf_lib.srf_init_onnx.restype = ctypes.c_int32

    # int32_t srf_extract_arcface_embedding(const float* host_roi_bgr, float* host_output_vector, int roi_width, int roi_height)
    srf_lib.srf_extract_arcface_embedding.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=3, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.c_int,
        ctypes.c_int
    ]
    srf_lib.srf_extract_arcface_embedding.restype = ctypes.c_int32


def init_srf_engine(model_path):
    """Inicializa la sesión C++ ONNX cargando el archivo .onnx"""
    if srf_lib is None: return False
    
    # Asegurar que la ruta sea bytes (C char*)
    b_path = model_path.encode('utf-8')
    res = srf_lib.srf_init_onnx(b_path)
    if res == 0:
        print("[FFI] Motor C++ ONNX Runtime Inicializado Exitosamente.")
        return True
    else:
        print("[FFI] Error al inicializar el Motor C++ ONNX.")
        return False

def extract_embedding_srf(crop_bgr):
    """
    Ingiere un parche (crop) en BGR del rostro (H, W, 3).
    Lo escala rápidamente a 112x112 en Python, y delega todo el 
    peso matemático NCHW + Inferencia ONNX al puente C++.
    Retorna un numpy array de [512] flotantes perfectos.
    """
    if srf_lib is None:
        raise RuntimeError("La librería SRF_ONNX no está cargada.")

    # ArcFace siempre requiere 112x112 fijos
    if crop_bgr.shape[0] != 112 or crop_bgr.shape[1] != 112:
        crop_bgr = cv2.resize(crop_bgr, (112, 112))

    # Convertir a RAW float32 memoria contigua para C++
    # ATENCION: NO DIVIDIMOS entre 127.5 aquí, C++ lo hará a velocidad nativa vectorial
    crop_float = np.ascontiguousarray(crop_bgr, dtype=np.float32)

    # Preparar el buffer de salida C++ de tamaño 512 puro (tamaño output final ArcFace)
    out_vector = np.zeros(512, dtype=np.float32)

    res = srf_lib.srf_extract_arcface_embedding(crop_float, out_vector, 112, 112)
    
    if res != 0:
        raise RuntimeError("Fallo interno en el C++ al inferir el Tensor ONNX.")

    return out_vector
