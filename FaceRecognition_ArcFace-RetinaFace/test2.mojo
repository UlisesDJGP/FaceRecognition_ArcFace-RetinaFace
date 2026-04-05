from memory import UnsafePointer

fn main():
    var a = UnsafePointer[UInt8].alloc(1)
    a[0] = 42
    var b = UnsafePointer[UnsafePointer[UInt8]].alloc(1)
    b[0] = a
    
    var opaque = b.bitcast[NoneType]()
    
    # We want to get `a` back from `opaque`.
    var recovered = opaque.bitcast[UnsafePointer[UInt8]]()[0]
    
    print("Recovered:", recovered[0])
