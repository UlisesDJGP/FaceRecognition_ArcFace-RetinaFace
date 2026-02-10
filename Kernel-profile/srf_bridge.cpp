#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include <atomic>
#include <immintrin.h>

enum BufferState { EMPTY, WRITING, READY, READING };

extern "C" {
    struct RawFrame {
        uint8_t* data;
        int width;
        int height;
        size_t size;
        int fd; // File descriptor de la cámara
        std::atomic<BufferState> state;
    };

    /**
     * Inicializa el hardware de la cámara Sony 4K vía V4L2.
     */
    RawFrame* init_hardware(const char* device, int w, int h) {
        int fd = open(device, O_RDWR);
        if (fd < 0) { perror("Error abriendo cámara"); return nullptr; }

        // Configuración de formato 4K (MJPEG o YUYV según la cámara Sony)
        struct v4l2_format fmt = {0};
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        fmt.fmt.pix.width = w;
        fmt.fmt.pix.height = h;
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV; // Ajustable según sensor
        fmt.fmt.pix.field = V4L2_FIELD_NONE;

        if (ioctl(fd, VIDIOC_S_FMT, &fmt) < 0) {
            perror("Error configurando resolución 4K");
            close(fd);
            return nullptr;
        }

        RawFrame* frame = (RawFrame*)malloc(sizeof(RawFrame));
        frame->width = w;
        frame->height = h;
        frame->size = (size_t)w * h * 2; // YUYV usa 2 bytes por píxel
        frame->fd = fd;
        // Memoria alineada para que Mojo procese con SIMD sin penalización
        frame->data = (uint8_t*)_mm_malloc(frame->size, 64);
        frame->state.store(EMPTY);

        return frame;
    }

    /**
     * Captura de alta velocidad: Inyecta el frame directamente en el buffer alineado.
     */
    int capture_to_bridge(RawFrame* frame) {
        if (frame->state.load() == READING) return -1;

        frame->state.store(WRITING);
        
        // En producción, aquí usamos read() o mmap para llenar frame->data
        // El driver de la Sony 4K deposita los bytes aquí:
        if (read(frame->fd, frame->data, frame->size) < 0) {
            frame->state.store(EMPTY);
            return -2;
        }

        frame->state.store(READY);
        return 0;
    }

    void shutdown_hardware(RawFrame* frame) {
        if (frame) {
            close(frame->fd);
            if (frame->data) _mm_free(frame->data);
            free(frame);
        }
    }
}