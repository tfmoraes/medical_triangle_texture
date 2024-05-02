import numpy as np
cimport numpy as np
cimport cython

from libc.math cimport floor, ceil, sqrt, fabs, round
from cython.parallel import prange
from cy_my_types cimport image_t

@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.cdivision(True)
cdef inline double get_LUT_value(double v, int ww, int wl) nogil:
    cdef double _min = wl - 0.5 - (ww - 1.0)/2.0
    cdef double _max = wl - 0.5 + (ww - 1.0)/2.0

    if v <= _min:
        return 0
    elif  v >= _max:
        return 255
    else:
        return ((v - (wl - 0.5))/(ww - 1.0) + 0.5) * 255


@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.cdivision(True)
cdef void surf_raycasting(image_t[:, :, :] volume, color_t[:, :, :] image, color_t[:, :] clut, int ww, int wl, int x, int y) nogil:
    cdef int dz, dy, dx
    dz = volume.shape[0]
    dy = volume.shape[1]
    dx = volume.shape[2]

    cdef double alphai = 0.0
    cdef double alpha = 0.0
    cdef double alphaj = 0.0

    cdef int s

    cdef double maxv = -9999

    cdef double cr, cg, cb
    cr = 0
    cg = 0
    cb = 0

    for z in xrange(dz):
        if 0 <= x <= (dx-1) and 0 <= y <= (dy-1) and 0 <= z <= (dz-1):
            # print init, end, vx, vy, vz
            # print n0[0], n0[1], n0[2], inx, iny, inz

            gv = get_LUT_value(volume[z, y, x], ww, wl)
            # if gv > maxv:
                # maxv = gv

            # if  maxv == 255:
                # break
            alpha = gv / 255.0

            cr = (clut[<int>gv, 0]/255.0)*alpha + cr * (1 - alpha)
            cg = (clut[<int>gv, 1]/255.0)*alpha + cg * (1 - alpha)
            cb = (clut[<int>gv, 2]/255.0)*alpha + cb * (1 - alpha)

            alphaj = alpha + (1 - alpha)*alphai
            alphai = alphaj

            # if alphai >= 1.0:
                # break

    image[y, x, 0] = <np.uint8_t>(255 * cr)
    image[y, x, 1] = <np.uint8_t>(255 * cg)
    image[y, x, 2] = <np.uint8_t>(255 * cb)
    # maxv = get_LUT_value(maxv, ww, wl)
    # image[y, x, 0] = clut[<int>maxv, 0]
    # image[y, x, 1] = clut[<int>maxv, 1]
    # image[y, x, 2] = clut[<int>maxv, 2]
