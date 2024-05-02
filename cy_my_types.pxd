import numpy as np
cimport numpy as np
cimport cython

ctypedef np.int16_t image_t
ctypedef np.uint8_t color_t

DTYPE = np.uint8
ctypedef np.uint8_t DTYPE_t

DTYPEF32 = np.float32
ctypedef np.float32_t DTYPEF32_t
