#http://en.wikipedia.org/wiki/Local_maximum_intensity_projection
import numpy as np
cimport numpy as np
cimport cython

from libc.math cimport floor, ceil, sqrt, fabs
from cython.parallel import prange

from cy_my_types cimport image_t, DTYPEF32_t, color_t
from raycasting cimport surf_raycasting

cdef image_t NULL_VALUE = -32768


@cython.boundscheck(False) # turn of bounds-checking for entire function
def lmip(np.ndarray[image_t, ndim=3] image, int axis, image_t tmin,
         image_t tmax, np.ndarray[image_t, ndim=2] out):
    cdef image_t max
    cdef int start
    cdef int sz = image.shape[0]
    cdef int sy = image.shape[1]
    cdef int sx = image.shape[2]

    # AXIAL
    if axis == 0:
        for x in xrange(sx):
            for y in xrange(sy):
                max = image[0, y, x]
                if max >= tmin and max <= tmax:
                    start = 1
                else:
                    start = 0
                for z in xrange(sz):
                    if image[z, y, x] > max:
                        max = image[z, y, x]

                    elif image[z, y, x] < max and start:
                        break
                    
                    if image[z, y, x] >= tmin and image[z, y, x] <= tmax:
                        start = 1

                out[y, x] = max

    #CORONAL
    elif axis == 1:
        for z in xrange(sz):
            for x in xrange(sx):
                max = image[z, 0, x]
                if max >= tmin and max <= tmax:
                    start = 1
                else:
                    start = 0
                for y in xrange(sy):
                    if image[z, y, x] > max:
                        max = image[z, y, x]

                    elif image[z, y, x] < max and start:
                        break
                    
                    if image[z, y, x] >= tmin and image[z, y, x] <= tmax:
                        start = 1

                out[z, x] = max

    #CORONAL
    elif axis == 2:
        for z in xrange(sz):
            for y in xrange(sy):
                max = image[z, y, 0]
                if max >= tmin and max <= tmax:
                    start = 1
                else:
                    start = 0
                for x in xrange(sx):
                    if image[z, y, x] > max:
                        max = image[z, y, x]

                    elif image[z, y, x] < max and start:
                        break
                    
                    if image[z, y, x] >= tmin and image[z, y, x] <= tmax:
                        start = 1

                out[z, y] = max


cdef image_t get_colour(image_t vl, image_t wl, image_t ww):
    cdef image_t out_colour
    cdef image_t min_value = wl - (ww / 2)
    cdef image_t max_value = wl + (ww / 2)
    if vl < min_value:
        out_colour = min_value
    elif vl > max_value:
        out_colour = max_value
    else:
        out_colour = vl

    return out_colour

@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.cdivision(True)
cdef float get_opacity(image_t vl, image_t wl, image_t ww) nogil:
    cdef float out_opacity
    cdef image_t min_value = wl - (ww / 2)
    cdef image_t max_value = wl + (ww / 2)
    if vl < min_value:
        out_opacity = 0.0
    elif vl > max_value:
        out_opacity = 1.0
    else:
        out_opacity = 1.0/(max_value - min_value) * (vl - min_value)

    return out_opacity

@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.cdivision(True)
cdef float get_opacity_f32(DTYPEF32_t vl, image_t wl, image_t ww) nogil:
    cdef float out_opacity
    cdef image_t min_value = wl - (ww / 2)
    cdef image_t max_value = wl + (ww / 2)
    if vl < min_value:
        out_opacity = 0.0
    elif vl > max_value:
        out_opacity = 1.0
    else:
        out_opacity = 1.0/(max_value - min_value) * (vl - min_value)

    return out_opacity


@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.cdivision(True)
def mida(np.ndarray[image_t, ndim=3] image, int axis, image_t wl,
         image_t ww, np.ndarray[image_t, ndim=2] out):
    cdef int sz = image.shape[0]
    cdef int sy = image.shape[1]
    cdef int sx = image.shape[2]

    cdef image_t min = image.min()
    cdef image_t max = image.max()
    cdef image_t vl

    cdef image_t min_value = wl - (ww / 2)
    cdef image_t max_value = wl + (ww / 2)

    cdef float fmax=0.0
    cdef float fpi
    cdef float dl
    cdef float bt

    cdef float alpha
    cdef float alpha_p = 0.0
    cdef float colour
    cdef float colour_p = 0

    cdef int x, y, z

    # AXIAL
    if axis == 0:
        for x in prange(sx, nogil=True):
            for y in xrange(sy):
                fmax = 0.0
                alpha_p = 0.0
                colour_p = 0.0
                for z in xrange(sz):
                    vl = image[z, y, x]
                    if vl != NULL_VALUE:
                        fpi = 1.0/(max - min) * (vl - min)
                        if fpi > fmax:
                            dl = fpi - fmax
                            fmax = fpi
                        else:
                            dl = 0.0

                        bt = 1.0 - dl
                        
                        colour = fpi
                        alpha = get_opacity(vl, wl, ww)
                        colour = (bt * colour_p) + (1 - bt * alpha_p) * colour * alpha
                        alpha = (bt * alpha_p) + (1 - bt * alpha_p) * alpha

                        colour_p = colour
                        alpha_p = alpha

                        if alpha >= 1.0:
                            break


                #out[y, x] = <image_t>((max_value - min_value) * colour + min_value)
                out[y, x] = <image_t>((max - min) * colour + min)


    #CORONAL
    elif axis == 1:
        for z in prange(sz, nogil=True):
            for x in xrange(sx):
                fmax = 0.0
                alpha_p = 0.0
                colour_p = 0.0
                for y in xrange(sy):
                    vl = image[z, y, x]
                    if vl != NULL_VALUE:
                        fpi = 1.0/(max - min) * (vl - min)
                        if fpi > fmax:
                            dl = fpi - fmax
                            fmax = fpi
                        else:
                            dl = 0.0

                        bt = 1.0 - dl
                        
                        colour = fpi
                        alpha = get_opacity(vl, wl, ww)
                        colour = (bt * colour_p) + (1 - bt * alpha_p) * colour * alpha
                        alpha = (bt * alpha_p) + (1 - bt * alpha_p) * alpha

                        colour_p = colour
                        alpha_p = alpha

                        if alpha >= 1.0:
                            break

                    out[z, x] = <image_t>((max - min) * colour + min)

    #AXIAL
    elif axis == 2:
        for z in prange(sz, nogil=True):
            for y in xrange(sy):
                fmax = 0.0
                alpha_p = 0.0
                colour_p = 0.0
                for x in xrange(sx):
                    vl = image[z, y, x]
                    if vl != NULL_VALUE:
                        fpi = 1.0/(max - min) * (vl - min)
                        if fpi > fmax:
                            dl = fpi - fmax
                            fmax = fpi
                        else:
                            dl = 0.0

                        bt = 1.0 - dl
                        
                        colour = fpi
                        alpha = get_opacity(vl, wl, ww)
                        colour = (bt * colour_p) + (1 - bt * alpha_p) * colour * alpha
                        alpha = (bt * alpha_p) + (1 - bt * alpha_p) * alpha

                        colour_p = colour
                        alpha_p = alpha

                        if alpha >= 1.0:
                            break

                    out[z, y] = <image_t>((max - min) * colour + min)



@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.cdivision(True)
cdef inline void finite_difference(image_t[:, :, :] image,
                              int x, int y, int z, float h, float *g) nogil:
    cdef int px, py, pz, fx, fy, fz

    cdef int sz = image.shape[0]
    cdef int sy = image.shape[1]
    cdef int sx = image.shape[2]

    cdef float gx, gy, gz

    if x == 0:
        px = 0
        fx = 1
    elif x == sx - 1:
        px = x - 1
        fx = x
    else:
        px = x - 1
        fx = x + 1

    if y == 0:
        py = 0
        fy = 1
    elif y == sy - 1:
        py = y - 1
        fy = y
    else:
        py = y - 1
        fy = y + 1

    if z == 0:
        pz = 0
        fz = 1
    elif z == sz - 1:
        pz = z - 1
        fz = z
    else:
        pz = z - 1
        fz = z + 1

    gx = (image[z, y, fx] - image[z, y, px]) / (2*h)
    gy = (image[z, fy, x] - image[z, py, x]) / (2*h)
    gz = (image[fz, y, x] - image[pz, y, x]) / (2*h)

    g[0] = gx
    g[1] = gy
    g[2] = gz



@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.cdivision(True)
cdef inline float calc_fcm_itensity(image_t[:, :, :] image,
                      int x, int y, int z, float n, float* dir) nogil:
    cdef float g[3]
    finite_difference(image, x, y, z, 1.0, g)
    cdef float gm = sqrt(g[0]*g[0] + g[1]*g[1] + g[2]*g[2])
    cdef float d = g[0]*dir[0] + g[1]*dir[1] + g[2]*dir[2]
    cdef float sf = (1.0 - fabs(d/gm))**n
    #alpha = get_opacity_f32(gm, wl, ww)
    cdef float vl = gm * sf
    return vl

@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.cdivision(True)
def fast_countour_mip(np.ndarray[image_t, ndim=3] image,
                      float n,
                      int axis,
                      image_t wl, image_t ww,
                      int tmip,
                      np.ndarray[image_t, ndim=2] out):
    cdef int sz = image.shape[0]
    cdef int sy = image.shape[1]
    cdef int sx = image.shape[2]
    cdef float gm
    cdef float alpha
    cdef float sf
    cdef float d

    cdef float* g
    cdef float* dir = [ 0, 0, 0 ]

    cdef image_t[:, :, :] vimage = image
    cdef np.ndarray[image_t, ndim=3] tmp = np.empty_like(image)

    cdef image_t min = image.min()
    cdef image_t max = image.max()
    cdef float fmin = <float>min
    cdef float fmax = <float>max
    cdef float vl
    cdef image_t V

    cdef int x, y, z

    if axis == 0:
        dir[2] = 1.0
    elif axis == 1:
        dir[1] = 1.0
    elif axis == 2:
        dir[0] = 1.0

    for z in prange(sz, nogil=True):
        for y in range(sy):
            for x in range(sx):
                vl = calc_fcm_itensity(vimage, x, y, z, n, dir)
                tmp[z, y, x] = <image_t>vl

    cdef image_t tmin = tmp.min()
    cdef image_t tmax = tmp.max()

    #tmp = ((max - min)/<float>(tmax - tmin)) * (tmp - tmin) + min

    if tmip == 0:
        out[:] = tmp.max(axis)
    elif tmip == 1:
        lmip(tmp, axis, 700, 3033, out)
    elif tmip == 2:
        mida(tmp, axis, wl, ww, out)


@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.cdivision(True)
def raycasting(image_t[:, :, :] image, color_t[:, :] clut, image_t wl, image_t ww, color_t[:, :, :] out):
    cdef int sz = image.shape[0]
    cdef int sy = image.shape[1]
    cdef int sx = image.shape[2]
    
    cdef int x, y

    for y in prange(sy, nogil=True):
        for x in xrange(sx):
            surf_raycasting(image, out, clut, ww, wl, x, y)
