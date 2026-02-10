from sys.ffi import DLHandle
from memory import UnsafePointer
from algorithm import parallelize

# Alias para claridad estructural
alias DTYPE = DType.uint8

struct SRF_Orchestrator:
    var _lib: DLHandle
    var _frame_ptr: UnsafePointer[NoneType]
    var kernel: SRFKernel
    var width: Int
    var height: Int

    fn __init__(inout self, path: String, w: Int, h: Int):
        self._lib = DLHandle(path)
        self.width = w
        self.height = h
        
        # Inicialización del bridge C++
        let create_fn = self._lib.get_function[fn(Int, Int) -> UnsafePointer[NoneType]]("create_aligned_frame")
        self._frame_ptr = create_fn(w, h)
        
        # Kernel configurado para 16 tiles (4x4) en imagen 4K
        self.kernel = SRFKernel(w, h, 4, 4)

    fn process_stream_cycle(inout self) raises:
        """
        Ciclo de vida del procesamiento: Ingesta segura -> Kernel de Particionado.
        """
        let capture_fn = self._lib.get_function[fn(UnsafePointer[NoneType]) -> Int]("capture_frame")
        
        # Intento de captura
        let status = capture_fn(self._frame_ptr)
        
        if status == 0:
            # Casteo seguro del puntero de datos
            # Accedemos al miembro 'data' del struct RawFrame (offset 0 en este caso)
            let data_ptr = self._frame_ptr.bitcast[UnsafePointer[UnsafePointer[DTYPE]]]()[]
            
            # Ejecución del kernel paralelizado
            self.kernel.process_frame(data_ptr)
        else:
            # El sistema está ocupado o el hardware no respondió
            pass

    fn __del__(owned self):
        let destroy_fn = self._lib.get_function[fn(UnsafePointer[NoneType]) -> None]("destroy_frame")
        destroy_fn(self._frame_ptr)