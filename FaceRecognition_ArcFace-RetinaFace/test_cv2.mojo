from python import Python
from memory import UnsafePointer

fn main() raises:
    Python.add_to_path(".")
    var helper = Python.import_module("display_helper")
    print("Python module loaded successfully!")
    
    var ptr = UnsafePointer[UInt8].alloc(1280 * 720 * 3)
    # Fill a bit of it
    for i in range(100):
        ptr[i] = 255
        
    var key = helper.show_frame(Int(ptr), 720, 1280, 0.42)
    print("Show frame executed! Key:", key)
