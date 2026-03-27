from sys.ffi import DLHandle
from memory import UnsafePointer

struct HardwareBridge:
    var _lib: DLHandle
    var _raw_frame: UnsafePointer[NoneType]
    var device_path: String

    fn __init__(out self, lib_path: String, dev: String, w: Int, h: Int) raises:
        self._lib = DLHandle(lib_path)
        self.device_path = dev
        
        var init_fn = self._lib.get_function[fn(UnsafePointer[UInt8], Int, Int) -> UnsafePointer[NoneType]]("init_hardware")
        self._raw_frame = init_fn(dev.as_bytes().unsafe_ptr(), w, h)
        
        if not self._raw_frame:
            print("❌ ERROR CRÍTICO: No se pudo inicializar el hardware Sony 4K.")

    fn update(self) -> Int:
        var cap_fn = self._lib.get_function[fn(UnsafePointer[NoneType]) -> Int]("capture_to_bridge")
        return cap_fn(self._raw_frame)

    fn get_data_ptr(self) -> UnsafePointer[UInt8]:
        return self._raw_frame.bitcast[UnsafePointer[UInt8]]()[0]

    fn __del__(owned self):
        var close_fn = self._lib.get_function[fn(UnsafePointer[NoneType]) -> None]("shutdown_hardware")
        close_fn(self._raw_frame)

struct SRFCudaBridge:
    var _lib: DLHandle
    var _init_nvml: fn() -> Int32
    var _process_roi_cuda: fn(UnsafePointer[Float32], UnsafePointer[Float32], Int32, Int32, Int32) -> Int32
    
    fn __init__(out self, lib_path: String) raises:
        print("[SISTEMA] Cargando librería dinámica CUDA...")
        self._lib = DLHandle(lib_path)
        
        self._init_nvml = self._lib.get_function[fn() -> Int32]("srf_init_nvml")
        self._process_roi_cuda = self._lib.get_function[fn(UnsafePointer[Float32], UnsafePointer[Float32], Int32, Int32, Int32) -> Int32]("srf_process_roi_cuda")
        
        var status = self._init_nvml()
        if status != 0:
            print("⚠️ [ADVERTENCIA] NVML falló. Operando sin métricas de temperatura.")
        else:
            print("✅ [NVML] Gestor térmico de NVIDIA anclado correctamente.")

    fn extract_vector_map(self, roi_ptr: UnsafePointer[Float32], vector_dim: Int32 = 512) -> UnsafePointer[Float32]:
        var vector_map = UnsafePointer[Float32].alloc(Int(vector_dim))
        
        var status = self._process_roi_cuda(roi_ptr, vector_map, 112, 112, vector_dim)
        
        if status != 0:
            print("❌ [ERROR] Falla de segmentación o ejecución en el kernel CUDA.")
            
        return vector_map