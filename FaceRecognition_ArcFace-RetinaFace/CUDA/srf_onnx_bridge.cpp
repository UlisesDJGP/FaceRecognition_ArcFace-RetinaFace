#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <cmath>

// Singleton environment
Ort::Env* env = nullptr;
Ort::Session* session = nullptr;

extern "C" {

    int32_t srf_init_onnx(const char* model_path) {
        if (env != nullptr) return 0; // Already initialized

        try {
            // Inicializar entorno ONNX
            env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "FaceRecognition_srf");

            Ort::SessionOptions session_options;
            session_options.SetIntraOpNumThreads(1);
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

            // Activar backend CUDA Execution Provider
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = 0;
            session_options.AppendExecutionProvider_CUDA(cuda_options);

            // Cargar modelo
            session = new Ort::Session(*env, model_path, session_options);
            return 0;
        } catch (const std::exception& e) {
            std::cerr << "C++ ONNX Runtime Exception: " << e.what() << std::endl;
            return -1;
        }
    }

    int32_t srf_extract_arcface_embedding(
        const float* host_roi_bgr, 
        float* host_output_vector, 
        int roi_width, 
        int roi_height
    ) {
        if (session == nullptr) return -1; // No initialized
        if (roi_width != 112 || roi_height != 112) return -1; // ArcFace requirs 112x112

        try {
            // Normalizacion In-Place CPU (HWC [112x112x3] -> NCHW [1x3x112x112]) y ((p - 127.5)/127.5)
            // Esto ahorra incontables milisegundos a Python
            std::vector<float> input_tensor_values(1 * 3 * 112 * 112);
            int channel_stride = 112 * 112;

            for (int h = 0; h < 112; ++h) {
                for (int w = 0; w < 112; ++w) {
                    for (int c = 0; c < 3; ++c) {
                        // host_roi_bgr is (H, W, C) -> RGB OpenCV
                        float pixel = host_roi_bgr[(h * 112 + w) * 3 + c];
                        input_tensor_values[c * channel_stride + h * 112 + w] = (pixel - 127.5f) / 127.5f;
                    }
                }
            }

            Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

            std::vector<int64_t> input_shape = {1, 3, 112, 112};
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, 
                input_tensor_values.data(), 
                input_tensor_values.size(), 
                input_shape.data(), 
                input_shape.size()
            );

            Ort::AllocatorWithDefaultOptions allocator;
            Ort::AllocatedStringPtr input_name_ptr = session->GetInputNameAllocated(0, allocator);
            Ort::AllocatedStringPtr output_name_ptr = session->GetOutputNameAllocated(0, allocator);

            const char* input_names[] = {input_name_ptr.get()};
            const char* output_names[] = {output_name_ptr.get()};

            auto output_tensors = session->Run(
                Ort::RunOptions{nullptr}, 
                input_names, 
                &input_tensor, 
                1, 
                output_names, 
                1
            );

            float* floatarr = output_tensors.front().GetTensorMutableData<float>();
            
            // Output normal de ArcFace es de 512
            for(int i = 0; i < 512; i++){
                host_output_vector[i] = floatarr[i];
            }

            return 0;

        } catch (const std::exception& e) {
            std::cerr << "C++ Inferencia Fallida: " << e.what() << std::endl;
            return -1;
        }
    }
}
