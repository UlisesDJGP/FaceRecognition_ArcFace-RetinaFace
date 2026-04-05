from sys.ffi import DLHandle
from memory import UnsafePointer
from srf_bridge_interface import SRFCudaBridge, HardwareBridge

alias DTYPE = DType.uint8

struct SRF_Orchestrator:
    var hardware_bridge: HardwareBridge
    var cuda_bridge: SRFCudaBridge
    var width: Int
    var height: Int

    fn __init__(out self, camera_lib_path: String, cuda_lib_path: String, w: Int, h: Int) raises:
        print("[SISTEMA] Inicializando Orquestador de Producción SRF-AR...")
        self.width = w
        self.height = h
        self.cuda_bridge = SRFCudaBridge(cuda_lib_path)
        print("[SISTEMA] Conectando con hardware de video (/dev/video0)...")
        self.hardware_bridge = HardwareBridge(camera_lib_path, "/dev/video0", w, h)

    fn process_stream_cycle(self) raises:
        var roi_elements = 112 * 112 * 3
        var normalized_roi_ptr = UnsafePointer[Float32].alloc(roi_elements)
        
        print("\n ¡INICIANDO TRANSMISIÓN EN VIVO! (Capturando 100 frames...)")
        print(" Pasa tu mano frente a la cámara para ver cómo cambian los valores de luz reales.\n")
        
        # Bucle de 100 frames para mantener la cámara encendida
        for frame_count in range(300):
            var status = self.hardware_bridge.update() 
            
            if status == 0:
                var raw_cam_ptr = self.hardware_bridge.get_data_ptr()
                
                # Extraemos UN solo píxel al azar (por ejemplo, el índice 50000) de tu cámara real
                # Si el cuarto está oscuro, será cercano a 0.0. Si hay luz, subirá.
                var pixel_real = raw_cam_ptr[50000].cast[DType.float32]() / 255.0
                
                print("[FRAME ", frame_count, "] Valor de luz del píxel físico: ", pixel_real)
                
                # Zero-Copy a la GPU (Aún con la matemática simulada en CUDA)
                var vector_map = self.cuda_bridge.extract_vector_map(normalized_roi_ptr, 512)
                vector_map.free()
            else:
                print(" [STREAM] Error al leer el frame de la cámara.")
                break
                
        normalized_roi_ptr.free()
        print("\n Transmisión finalizada. Hardware liberado.")

    fn __del__(owned self):
        pass

fn main():
    print("================================================================")
    print(" PRUEBA DE REALIDAD: ORQUESTADOR CONTINUO ")
    print("================================================================")
    
    var cuda_lib_path = "/home/ulises/Documentos/FaceRecognition/build/libsrf_bridge.so"
    var camera_lib_path = "/home/ulises/Documentos/FaceRecognition/build/libcamera.so" 
    
    try:
        # Iniciamos a 1280x720 (Cuando tengas la Sony 4K, solo cambiaremos estos números a 3840, 2160)
        var orchestrator = SRF_Orchestrator(camera_lib_path, cuda_lib_path, 1280, 720)
        orchestrator.process_stream_cycle()
    except e:
        print(" [ERROR CRÍTICO] Excepción: ", e)