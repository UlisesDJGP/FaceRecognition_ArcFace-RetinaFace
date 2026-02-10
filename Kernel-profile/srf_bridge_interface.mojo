from sys.ffi import DLHandle
from memory import UnsafePointer

struct HardwareBridge:
    var _lib: DLHandle
    var _raw_frame: UnsafePointer[NoneType]
    var device_path: String

    fn __init__(inout self, lib_path: String, dev: String, w: Int, h: Int):
        self._lib = DLHandle(lib_path)
        self.device_path = dev
        
        let init_fn = self._lib.get_function[fn(UnsafePointer[UInt8], Int, Int) -> UnsafePointer[NoneType]]("init_hardware")
        
        # Pasamos el path del dispositivo (ej: /dev/video0)
        self._raw_frame = init_fn(dev.as_bytes().as_ptr(), w, h)
        
        if not self._raw_frame:
            print("ERROR CRÍTICO: No se pudo inicializar el hardware Sony 4K.")

    fn update(self) -> Int:
        """Llama a la captura física de la cámara."""
        let cap_fn = self._lib.get_function[fn(UnsafePointer[NoneType]) -> Int]("capture_to_bridge")
        return cap_fn(self._raw_frame)

    fn get_data_ptr(self) -> UnsafePointer[UInt8]:
        """Extrae el puntero de datos alineado para el Kernel de Mojo."""
        # El puntero a datos es el primer miembro de la estructura RawFrame
        return self._raw_frame.bitcast[UnsafePointer[UnsafePointer[UInt8]]]()[]

    fn __del__(owned self):
        let close_fn = self._lib.get_function[fn(UnsafePointer[NoneType]) -> None]("shutdown_hardware")
        close_fn(self._raw_frame)