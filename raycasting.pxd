import numpy as np
cimport numpy as np
cimport cython

from libc.math cimport floor, ceil, sqrt, fabs, round
from cython.parallel import prange
from cy_my_types cimport image_t, color_t

cdef void surf_raycasting(image_t[:, :, :], color_t[:, :, :], color_t[:, :], int, int, int, int) nogil
