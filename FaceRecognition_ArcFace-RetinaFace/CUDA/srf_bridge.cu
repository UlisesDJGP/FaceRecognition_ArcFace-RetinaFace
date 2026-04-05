#include <cuda_runtime.h>
#include <nvml.h>
#include <iostream>

#define TILE_SIZE 16
#define TILE_VECTOR_DIM 64
#define GRID_SIZE 7
#define MAX_TEMP_THRESHOLD 80
#define SAFE_TEMP_THRESHOLD 70


// 1. KERNEL COMPUTACIONAL PURO (Ejecución en GPU)

__global__ void extract_tile_vector_kernel(
    const float* __restrict__ roi_input,     
    float* __restrict__ vector_map,        
    int roi_width,
    int roi_height
) {
    int tile_x = blockIdx.x;
    int tile_y = blockIdx.y;
    int thread_id = threadIdx.x;
    
    int tile_offset_x = tile_x * TILE_SIZE;
    int tile_offset_y = tile_y * TILE_SIZE;
    
    // Carga a memoria ultra-rápida
    __shared__ float tile_accum[3][TILE_SIZE][TILE_SIZE];
    
    for (int c = 0; c < 3; c++) {
        for (int i = thread_id; i < TILE_SIZE * TILE_SIZE; i += blockDim.x) {
            int local_y = i / TILE_SIZE;
            int local_x = i % TILE_SIZE;
            int global_x = tile_offset_x + local_x;
            int global_y = tile_offset_y + local_y;
            
            int global_idx = (c * roi_height + global_y) * roi_width + global_x;
            tile_accum[c][local_y][local_x] = roi_input[global_idx];
        }
    }
    __syncthreads();
    
    // Procesamiento SIMT en la GPU
    float features[TILE_VECTOR_DIM];
    for (int f = thread_id; f < TILE_VECTOR_DIM; f += blockDim.x) {
        features[f] = 0.0f;
        for (int c = 0; c < 3; c++) {
            for (int y = 1; y < TILE_SIZE - 1; y++) {
                for (int x = 1; x < TILE_SIZE - 1; x++) {
                    float val = tile_accum[c][y][x];
                    features[f] += val * __sinf((float)f * val); 
                }
            }
        }
    }
    __syncthreads();
    
    // Escritura coalescida protegida contra desbordamientos
    if (thread_id < TILE_VECTOR_DIM) {
        int tile_idx = tile_y * GRID_SIZE + tile_x;
        vector_map[tile_idx * TILE_VECTOR_DIM + thread_id] = features[thread_id];
    }
}

// -------------------------------------------------------------------
// 2. FFI PLANA PARA MOJO (Ejecución en Host)
// -------------------------------------------------------------------
extern "C" {
    
    // Llama esto UNA sola vez al iniciar tu aplicación Mojo
    int32_t srf_init_nvml() {
        return (nvmlInit() == NVML_SUCCESS) ? 0 : -1;
    }

    // Pipeline aplanado: Evalúa térmicas -> Asigna Memoria -> Ejecuta -> Retorna
    int32_t srf_process_roi_cuda(
        const float* host_roi_pixels, 
        float* host_output_vector,      
        int roi_width, 
        int roi_height,
        int vector_dim
    ) {
        // A. Evaluación Térmica Lineal (O(1))
        nvmlDevice_t nvml_device;
        if (nvmlDeviceGetHandleByIndex(0, &nvml_device) == NVML_SUCCESS) {
            unsigned int temp;
            nvmlDeviceGetTemperature(nvml_device, NVML_TEMPERATURE_GPU, &temp);
            
            // En producción, aquí puedes acoplar el flag para derivar el procesamiento al CPU
            // si la temperatura excede el límite crítico de forma sostenida.
            if (temp > MAX_TEMP_THRESHOLD) {
                std::cout << "[ALERTA] Throttling inminente. Temp: " << temp << "°C\n";
            }
        }

        // B. Gestión de Memoria
        size_t roi_bytes = roi_width * roi_height * 3 * sizeof(float);
        size_t out_bytes = vector_dim * sizeof(float);

        float *d_roi, *d_out;
        if (cudaMalloc(&d_roi, roi_bytes) != cudaSuccess || cudaMalloc(&d_out, out_bytes) != cudaSuccess) {
            return -1;
        }
        cudaMemcpy(d_roi, host_roi_pixels, roi_bytes, cudaMemcpyHostToDevice);
        
        // C. Lanzamiento Directo
        dim3 grid(GRID_SIZE, GRID_SIZE);
        dim3 block(128);
        extract_tile_vector_kernel<<<grid, block>>>(d_roi, d_out, roi_width, roi_height);

        // D. Retorno de Resultados y Limpieza
        cudaMemcpy(host_output_vector, d_out, out_bytes, cudaMemcpyDeviceToHost);
        cudaFree(d_roi);
        cudaFree(d_out);

        return 0; 
    }
}