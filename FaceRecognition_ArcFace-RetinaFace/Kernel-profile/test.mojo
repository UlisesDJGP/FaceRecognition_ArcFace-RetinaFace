from srf_bridge_interface import SRFCudaBridge
from memory import UnsafePointer

fn main():
    print("================================================================")
    print("🧪 PRUEBA DE AISLAMIENTO: KERNEL CUDA (MOCK DATA)")
    print("================================================================")
    
    var cuda_lib = "/home/ulises/Documentos/FaceRecognition/build/libsrf_bridge.so"
    
    try:
        print("[1/3] Conectando con la gráfica NVIDIA...")
        var cuda_bridge = SRFCudaBridge(cuda_lib)
        
        print("[2/3] Generando puntero falso (Mock ROI) de 112x112x3...")
        var num_elements = 112 * 112 * 3
        var mock_roi = UnsafePointer[Float32].alloc(num_elements)
        
        for i in range(num_elements):
            mock_roi[i] = 0.5 
            
        print("[3/3] Ejecutando Zero-Copy a la VRAM y lanzando hilos CUDA...")
        var vector_map = cuda_bridge.extract_vector_map(mock_roi, 512)
        
        print("\n✅ ¡ÉXITO! El kernel procesó la imagen simulada.")
        print("Muestra de los primeros 5 valores del vector extraído:")
        for i in range(5):
            print("  -> Feature[", i, "]: ", vector_map[i])
            
        mock_roi.free()
        vector_map.free()
            
    except e:
        print("\n❌ [ERROR CRÍTICO] La ejecución falló: ", e)