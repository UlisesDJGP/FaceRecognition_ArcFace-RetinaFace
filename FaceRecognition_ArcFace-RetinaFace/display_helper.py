import numpy as np
import ctypes
import cv2

def show_frame(ptr_int, h, w, light_val):
    c_ptr = ctypes.cast(ptr_int, ctypes.POINTER(ctypes.c_uint8))
    arr = np.ctypeslib.as_array(c_ptr, shape=(h, w, 3))
    
    # Make a copy since we don't want to modify raw buffer and cause segfaults
    # if it's read-only, etc.
    display_arr = arr.copy()
    
    cv2.putText(display_arr, f"Light: {light_val}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # We won't actually imshow here in the test because we run headless
    # But we can test if cv2 loads
    return 0
