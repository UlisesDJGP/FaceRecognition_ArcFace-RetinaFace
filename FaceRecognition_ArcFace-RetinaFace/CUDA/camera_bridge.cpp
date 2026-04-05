#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdint>
#include <cstring>

// 1. ALINEACIÓN ESTRICHA DE MEMORIA (ABI compatible con Mojo)
// El orden de las variables aquí es de vital importancia para el bitcast.
struct CameraHandle {
    uint8_t* frame_data;          // Offset 0: Puntero directo a los píxeles (Mojo lee esto)
    cv::VideoCapture* cap;        // Offset 8: Controlador V4L2
    cv::Mat* current_frame;       // Offset 16: Buffer temporal de OpenCV
    int width;
    int height;
};

// 2. INTERFAZ FORÁNEA (FFI) EXPORTADA EN C PURO
extern "C" {

    // Inicializa el hardware y reserva la memoria RAM
    void* init_hardware(const char* device_path, int width, int height) {
        CameraHandle* handle = new CameraHandle();
        handle->width = width;
        handle->height = height;
        
        // Forzamos el backend V4L2 nativo de Linux para menor latencia
        handle->cap = new cv::VideoCapture(device_path, cv::CAP_V4L2);
        
        if (!handle->cap->isOpened()) {
            std::cerr << " [ERROR] OpenCV no pudo abrir el dispositivo V4L2: " << device_path << std::endl;
            delete handle;
            return nullptr;
        }

        // Negociación de resolución de hardware
        handle->cap->set(cv::CAP_PROP_FRAME_WIDTH, width);
        handle->cap->set(cv::CAP_PROP_FRAME_HEIGHT, height);
        // Sugerir MJPG es crucial para anchos de banda 4K en USB
        handle->cap->set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));

        handle->current_frame = new cv::Mat();
        
        // Alojamiento en el Heap: Un buffer contiguo para RGB (3 canales)
        handle->frame_data = new uint8_t[width * height * 3];

        std::cout << " [SISTEMA] Hardware de cámara inicializado (" << device_path << ") a " << width << "x" << height << std::endl;
        return static_cast<void*>(handle);
    }

    // Captura un frame, lo procesa y lo deposita en la RAM alineada
    int capture_to_bridge(void* raw_handle) {
        if (!raw_handle) return -1;
        CameraHandle* handle = static_cast<CameraHandle*>(raw_handle);

        // Lectura de hardware
        handle->cap->read(*(handle->current_frame));
        if (handle->current_frame->empty()) {
            return -1; // Descarte de frame corrupto
        }

        // Transformación de espacio de color: OpenCV lee BGR, las redes neuronales exigen RGB.
        cv::Mat rgb_frame;
        cv::cvtColor(*(handle->current_frame), rgb_frame, cv::COLOR_BGR2RGB);

        // Fallback de seguridad: Si la cámara ignoró la resolución (muy común en webcams baratas)
        // forzamos el redimensionamiento por software para no desbordar la memoria de Mojo.
        if (rgb_frame.cols != handle->width || rgb_frame.rows != handle->height) {
            cv::resize(rgb_frame, rgb_frame, cv::Size(handle->width, handle->height));
        }

        // Copia plana de bytes al bloque maestro (Zero-Copy hacia Mojo)
        std::memcpy(handle->frame_data, rgb_frame.data, handle->width * handle->height * 3);

        return 0; // Código de salida exitoso
    }

    // Prevención de Fugas de Memoria (Memory Leaks)
    void shutdown_hardware(void* raw_handle) {
        if (!raw_handle) return;
        CameraHandle* handle = static_cast<CameraHandle*>(raw_handle);
        
        if (handle->cap && handle->cap->isOpened()) {
            handle->cap->release();
        }
        
        delete handle->cap;
        delete handle->current_frame;
        delete[] handle->frame_data;
        delete handle;
        
        std::cout << " [SISTEMA] Hardware de cámara liberado de forma segura." << std::endl;
    }

}