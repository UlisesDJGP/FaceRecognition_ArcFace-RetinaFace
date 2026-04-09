# CUDA & Mojo Blueprint: 4K Double Buffer Orchestration

Este documento sirve como hoja de ruta técnica para migrar la arquitectura de inferencia Python-OpenCV a la futura integración C++/CUDA y Mojo (`srf_bridge.cu` y `main.mojo`).

## 1. Justificación de la Estrategia

El envío de matrices UltraHD (4K: 3840x2160x3) directamente a los kernels convolucionales de detección en VRAM causa ahogamiento del bus PCIe y saturación de la memoria unificada. La arquitectura ideal (ya probada y perfilada en Python) implementa un Doble Buffer que escinde el flujo de inferencia en dos flujos desacoplados de resolución asimétrica.

## 2. Mapa de Concurrencia (Memory Pipeline)

En C++ (a través del TensorRT/CUDNN Backend), el flujo debe imitar precisamente el siguiente ciclo:

### A. Captura (Raw Buffer)
1. Extraer frame `UHD_Buffer` [3840, 2160, 3] en puntero C++.
2. Subir matriz asíncronamente a memoria de GPU `cudaMemcpyAsync`.

### B. Downsampling (Small Buffer)
1. Lanzar kernel CUDA `cv::cuda::resize` o `nppResize` reduciendo el clon a `Small_Buffer` [640, 360, 3].
2. Formato: `NHWC` (Interleaved) -> `NCHW` o lo que requiera TensorRT.

### C. Fase 1: Detección (RetinaFace Kernel)
1. Ejecutar Inferencia de Detección en TensorRT usando el `Small_Buffer`.
2. Salida: Vector de predicciones `bboxes_small` [N, 5] (x1, y1, x2, y2, score).

### D. Fase Matemática: Mapeo de Ejes (Escala Dinámica)
Se debe calcular un escalador flotante en tiempo real (evitar hardcodeos) antes del recorte:
```cpp
// En C++ / CUDA Kernel
float ratio_x = (float)UHD_Buffer.width / (float)Small_Buffer.width;
float ratio_y = (float)UHD_Buffer.height / (float)Small_Buffer.height;

for(int i=0; i < N; ++i) {
    UHD_bboxes[i].x1 = bboxes_small[i].x1 * ratio_x;
    UHD_bboxes[i].y1 = bboxes_small[i].y1 * ratio_y;
    UHD_bboxes[i].x2 = bboxes_small[i].x2 * ratio_x;
    UHD_bboxes[i].y2 = bboxes_small[i].y2 * ratio_y;
}
```

### E. Fase 2: Recorte (ROI Extraction) & Alineación
1. Utilizar un kernel de CUDA `CropAndResize` o transformar afinmente (`SimilarityTransform`) usando `UHD_bboxes` directamente encima del `UHD_Buffer`.
2. Salidas: Mini-buffers de [112, 112, 3] por cada rostro, con altísima fidelidad de pixeles.

### F. Fase 3: Reconocimiento (ArcFace Kernel)
1. Ingerir los mini-buffers empacados como un batch: Tensor de dimensiones `[N, 3, 112, 112]`.
2. Lanzar inferencia `ArcFace`.
3. Descargar embeddings de dimensión `[N, 512]` a Mojo.

## 3. Notas de Transpoblación a Mojo / FFI (`srf_bridge.cu`)

Para enviar comandos desde `main.mojo`, la interfaz FFI debe diseñarse como una máquina de estados:
```mojo
fn execute_double_buffer_cycle(frame_ptr: UnsafePointer[UInt8], width: Int, height: Int)
```
Internamente, `srf_bridge.cu` no debe interrumpir o devolver datos a Mojo hasta que la Fase 3 (Extracción ArcFace) retorne los embeddings `[512]`. Esto reduce fuertemente la latencia de cambios de contexto entre Mojo y C++.
