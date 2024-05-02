import numpy as np
cimport numpy as np
cimport cython

from libc.math cimport floor, ceil, sqrt, fabs, round
from cython.parallel import prange
# from constants cimport NULL_VALUE

from cy_my_types cimport image_t
from interpolation cimport interpolate, tricubicInterpolate

cdef image_t NULL_VALUE = -32768

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
cdef inline void uv2bar(double x1, double y1,
                        double x2, double y2,
                        double x3, double y3,
                        double x, double y, np.float64_t[3] bar) nogil:
    bar[0] = ((y2-y3)*(x-x3) + (x3-x2)*(y-y3))/((y2-y3)*(x1-x3) + (x3-x2)*(y1-y3))
    bar[1] = ((y3-y1)*(x-x3) + (x1-x3)*(y-y3))/((y2-y3)*(x1-x3) + (x3-x2)*(y1-y3))
    bar[2] = 1.0 - bar[0] - bar[1]


@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.cdivision(True)
cdef void raycasting(image_t[:, :, :] volume, np.uint8_t[:, :, :] image, np.uint8_t[:, :] clut, int ww, int wl, int x, int y, double vx, double vy, double vz, double inx, double iny, double inz, np.float64_t[:] spacing, double offset, int nsteps) nogil:
    cdef int dz, dy, dx
    dz = volume.shape[0]
    dy = volume.shape[1]
    dx = volume.shape[2]


    cdef double[3] end
    cdef double[3] init #= np.zeros(shape=(3), dtype='float64')

    init[0] = vx
    init[1] = vy
    init[2] = vz

    end[0] =  vx + (offset)*inx
    end[1] =  vy + (offset)*iny
    end[2] =  vz + (offset)*inz

    cdef double step = offset / (nsteps)

    cdef double px, py, pz

    cdef double alphai = 0.0
    cdef double alpha = 0.0
    cdef double alphaj = 0.0

    cdef int s

    cdef double maxv = -9999

    cdef double cr, cg, cb
    cr = 0
    cg = 0
    cb = 0

    for s in xrange(nsteps + 1):
        # vx = (1.0 - 1.0/s)*init[0] + end[0]*1.0/s
        # vy = (1.0 - 1.0/s)*init[1] + end[1]*1.0/s
        # vz = (1.0 - 1.0/s)*init[2] + end[2]*1.0/s

        px = (init[0] + inx*step*s) 
        py = (init[1] + iny*step*s) 
        pz = (init[2] + inz*step*s) 

        if 0 <= px <= (dx-1) and 0 <= py <= (dy-1) and 0 <= pz <= (dz-1):

            # print init, end, vx, vy, vz
            # print n0[0], n0[1], n0[2], inx, iny, inz

            gv = get_LUT_value(interpolate(volume, px, py, pz), ww, wl)
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


@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.cdivision(True)
cdef void raycasting_mida(image_t[:, :, :] volume, np.uint8_t[:, :, :] image, np.uint8_t[:, :] clut, int ww, int wl, int x, int y, double vx, double vy, double vz, double inx, double iny, double inz, np.float64_t[:] spacing, double offset, int nsteps) nogil:
    cdef int dz, dy, dx
    dz = volume.shape[0]
    dy = volume.shape[1]
    dx = volume.shape[2]


    cdef double[3] end
    cdef double[3] init #= np.zeros(shape=(3), dtype='float64')

    init[0] = vx + offset/2.0 * inx
    init[1] = vy + offset/2.0 * iny
    init[2] = vz + offset/2.0 * inz

    end[0] =  vx - offset/2.0*inx
    end[1] =  vy - offset/2.0*iny
    end[2] =  vz - offset/2.0*inz

    cdef double step = offset / (nsteps)

    cdef double px, py, pz

    cdef double alphai = 0.0
    cdef double alpha = 0.0
    cdef double alphaj = 0.0

    cdef int s

    cdef double maxv = -9999

    for s in xrange(nsteps + 1):
        # vx = (1.0 - 1.0/s)*init[0] + end[0]*1.0/s
        # vy = (1.0 - 1.0/s)*init[1] + end[1]*1.0/s
        # vz = (1.0 - 1.0/s)*init[2] + end[2]*1.0/s

        px = (init[0] - inx*step*s) 
        py = (init[1] - iny*step*s) 
        pz = (init[2] - inz*step*s) 

        if 0 <= px <= (dx-1) and 0 <= py <= (dy-1) and 0 <= pz <= (dz-1):

            # print init, end, vx, vy, vz
            # print n0[0], n0[1], n0[2], inx, iny, inz

            gv = get_LUT_value(interpolate(volume, px, py, pz), ww, wl)
            # if gv > maxv:
                # maxv = gv

            # if  maxv == 255:
                # break
            alpha = gv / 255.0

            image[y, x, 0] = <np.uint8_t>(image[y, x, 0] + clut[<int>gv, 0] * alpha * (1 - alphai))
            image[y, x, 1] = <np.uint8_t>(image[y, x, 1] + clut[<int>gv, 1] * alpha * (1 - alphai))
            image[y, x, 2] = <np.uint8_t>(image[y, x, 2] + clut[<int>gv, 2] * alpha * (1 - alphai))

            alphaj = (1 - alphai)*alpha + alphai
            alphai = alphaj

            if alphai >= 1.0:
                break

@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.cdivision(True)
cdef void raycasting2hf(image_t[:, :, :] volume, image_t[:, :, :] image, np.uint8_t[:, :] clut, int ww, int wl, int x, int y, double vx, double vy, double vz, double inx, double iny, double inz, np.float64_t[:] spacing, double offset, int nsteps) nogil:
    cdef int dz, dy, dx
    dz = volume.shape[0]
    dy = volume.shape[1]
    dx = volume.shape[2]


    cdef double[3] end
    cdef double[3] init #= np.zeros(shape=(3), dtype='float64')

    init[0] = vx
    init[1] = vy
    init[2] = vz

    end[0] =  vx + (-offset)*inx
    end[1] =  vy + (-offset)*iny
    end[2] =  vz + (-offset)*inz

    cdef double step = offset / (nsteps)

    cdef double px, py, pz

    cdef double alphai = 0.0
    cdef double alpha = 0.0
    cdef double alphaj = 0.0

    cdef int s

    cdef double maxv = -9999

    cdef double cr, cg, cb
    cr = 0
    cg = 0
    cb = 0

    for s in xrange(nsteps):
        # vx = (1.0 - 1.0/s)*init[0] + end[0]*1.0/s
        # vy = (1.0 - 1.0/s)*init[1] + end[1]*1.0/s
        # vz = (1.0 - 1.0/s)*init[2] + end[2]*1.0/s

        px = (init[0] + inx*step*s) / spacing[0]
        py = (init[1] + iny*step*s) / spacing[1]
        pz = (init[2] + inz*step*s) / spacing[2]

        if 0 <= px <= (dx-1) and 0 <= py <= (dy-1) and 0 <= pz <= (dz-1):
            image[s, y, x] = <image_t>tricubicInterpolate(volume, px, py, pz)
        else:
            image[s, y, x] = NULL_VALUE
            # volume[<int>(pz), <int>(py), <int>( px )] = s * 100



@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.cdivision(True)
def generate_tcoords(np.float64_t[:, :] vertices, np.int32_t[:, :] faces, image_t[:, :, :] volume, np.float64_t[:] spacing, int offset=2, int dim=5000):
    cdef int n = faces.shape[0]
    cdef int nx = <int>sqrt(n)
    cdef int ny = <int>(ceil(n / <float>(nx)))

    cdef int d = dim
    cdef double dtx = d / <float>(nx)
    cdef double dty = d / <float>(ny)

    cdef np.ndarray[np.int16_t, ndim=2] image = np.zeros(shape=(d, d), dtype='int16')
    cdef np.ndarray[np.uint8_t, ndim=3] tnormals = np.zeros(shape=(d, d, 3), dtype='uint8')
    cdef np.ndarray[np.float64_t, ndim=2] tcoords = np.zeros(shape=(faces.shape[0], 6), dtype='float64')
    cdef int tmin = -2000
    cdef int tmax = 5000

    cdef int tc = 0
    cdef np.float64_t[3] bar

    cdef np.float64_t[:] v0, v1, v2
    cdef double gv
    cdef np.uint8_t g
    cdef int i, j
    cdef int x, y

    cdef int dz, dy, dx
    dz = volume.shape[0]
    dy = volume.shape[1]
    dx = volume.shape[2]

    cdef to_return = 0

    cdef int c0x, c0y, c1x, c1y, c2x, c2y
    cdef double vx, vy, vz

    cdef double tnx, tny, tnz, tnn
    cdef double hx, hy, hz
    hx = 1.0 #spacing[0]
    hy = 1.0 #spacing[1]
    hz = 1.0 #spacing[2]

    for tc in xrange(n):
        # for i in xrange(nx):
        i = tc % nx
        j = tc / nx
        c0x = <int>(i * dtx + offset)
        c0y = <int>(j * dty + offset)

        c1x = <int>(c0x + dtx - offset)
        c1y = <int>(c0y)

        c2x = <int>((c0x + c1x) / 2.0)
        c2y = <int>(c0y + dty - offset)

        tcoords[tc, 0] = <float>(c0x)/d
        tcoords[tc, 1] = 1-<float>(c0y)/d
        tcoords[tc, 2] = <float>(c1x)/d
        tcoords[tc, 3] = 1-<float>(c1y)/d
        tcoords[tc, 4] = <float>(c2x)/d
        tcoords[tc, 5] = 1-<float>(c2y)/d

        # print ">>>>>>>>>", tc

        v0 = vertices[faces[tc, 0]]
        v1 = vertices[faces[tc, 1]]
        v2 = vertices[faces[tc, 2]]

        for y in xrange(c0y-1, c2y+1):
            for x in xrange(c0x-1, c1x+1):
                uv2bar(tcoords[tc,0], tcoords[tc,1],
                       tcoords[tc,2], tcoords[tc,3],
                       tcoords[tc,4], tcoords[tc,5],
                       <float>(x)/d, 1-<float>(y)/d,
                       bar)


                vx = (bar[0]*v0[0] + bar[1]*v1[0] + bar[2]*v2[0]) / spacing[0]
                vy = (bar[0]*v0[1] + bar[1]*v1[1] + bar[2]*v2[1]) / spacing[1]
                vz = (bar[0]*v0[2] + bar[1]*v1[2] + bar[2]*v2[2]) / spacing[2]


                gv = (interpolate(volume, vx, vy, vz) + interpolate(volume, vx+hx, vy, vz) + interpolate(volume, vx-hx, vy, vz) + interpolate(volume, vx, vy+hy, vz) + interpolate(volume, vx, vy-hy, vz) + interpolate(volume, vx, vy, vz+hz)  + interpolate(volume, vx, vy, vz - hz)) / 7.0 
                tnx = (interpolate(volume, vx + hx, vy, vz) - interpolate(volume, vx - hx, vy, vz))/(2.0 * hx)
                tny = (interpolate(volume, vx, vy + hy, vz) - interpolate(volume, vx, vy - hy, vz))/(2.0 * hy)
                tnz = (interpolate(volume, vx, vy, vz + hz) - interpolate(volume, vx, vy, vz - hz))/(2.0 * hz)
                tnn = sqrt(tnx*tnx + tny*tny + tnz*tnz)
                # g = <np.uint8_t>((gv - tmin) * (255.0/(tmax - tmin)))

                image[y, x] = <np.int16_t>gv
                tnormals[y, x, 0] = <np.uint8_t>(((tnx / tnn) + 1.0) * 255.0/2.0)
                tnormals[y, x, 1] = <np.uint8_t>(((tny / tnn) + 1.0) * 255.0/2.0)
                tnormals[y, x, 2] = <np.uint8_t>(((tnz / tnn) + 1.0) * 255.0/2.0)

                # print tnx/tnn, tny/tnn, tnz/tnn, tnormals[y, x, 0], tnormals[y, x, 1], tnormals[y, x, 2]

                # image[y, x, 0] = g
                # image[y, x, 1] = g
                # image[y, x, 2] = g

            # tc+=1
            # if tc == n:
                # to_return = 1
                # break

        # if to_return:
            # break

    print "returned"
    return tcoords, image, tnormals


@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.cdivision(True)
cdef void _generate_tcoords_color(np.float64_t[:, :] vertices, np.float64_t[:, :] normals, np.int32_t[:, :] faces, image_t[:, :, :] volume, np.float64_t[:] spacing, int ww, int wl, np.uint8_t[:, :] clut, int offset, int dim, np.float64_t[:, :] tcoords, np.uint8_t[:, :, :] image, np.uint8_t[:, :, :] tnormals, int tc, int nx, double dtx, double dty, int d) nogil:

    cdef double tnx, tny, tnz, tnn
    cdef double hx, hy, hz
    hx = 1.0 #spacing[0]
    hy = 1.0 #spacing[1]
    hz = 1.0 #spacing[2]

    cdef np.float64_t[3] bar
    cdef double v0x, v0y, v0z
    cdef double v1x, v1y, v1z
    cdef double v2x, v2y, v2z

    # cdef np.float64_t[:] n0, n1, n2
    cdef double n0x, n0y, n0z
    cdef double n1x, n1y, n1z
    cdef double n2x, n2y, n2z
    cdef double gv
    cdef np.uint8_t g
    cdef int i, j
    cdef int x, y

    cdef int c0x, c0y, c1x, c1y, c2x, c2y
    cdef double vx, vy, vz
    cdef double px, py, pz
    cdef double inx, iny, inz, inn

    i = tc % nx
    j = tc / nx
    c0x = <int>(i * dtx + offset)
    c0y = <int>(j * dty + offset)

    c1x = <int>(c0x + dtx - offset)
    c1y = <int>(c0y)

    c2x = <int>((c0x + c1x) / 2.0)
    c2y = <int>(c0y + dty - offset)

    tcoords[tc, 0] = <float>(c0x)/d
    tcoords[tc, 1] = 1-<float>(c0y)/d
    tcoords[tc, 2] = <float>(c1x)/d
    tcoords[tc, 3] = 1-<float>(c1y)/d
    tcoords[tc, 4] = <float>(c2x)/d
    tcoords[tc, 5] = 1-<float>(c2y)/d


    v0x = vertices[faces[tc, 0], 0]
    v0y = vertices[faces[tc, 0], 1]
    v0z = vertices[faces[tc, 0], 2]

    v1x = vertices[faces[tc, 1], 0]
    v1y = vertices[faces[tc, 1], 1]
    v1z = vertices[faces[tc, 1], 2]

    v2x = vertices[faces[tc, 2], 0]
    v2y = vertices[faces[tc, 2], 1]
    v2z = vertices[faces[tc, 2], 2]

    n0x = normals[faces[tc, 0], 0]
    n0y = normals[faces[tc, 0], 1]
    n0z = normals[faces[tc, 0], 2]

    n1x = normals[faces[tc, 1], 0]
    n1y = normals[faces[tc, 1], 1]
    n1z = normals[faces[tc, 1], 2]

    n2x = normals[faces[tc, 2], 0]
    n2y = normals[faces[tc, 2], 1]
    n2z = normals[faces[tc, 2], 2]


    for y in xrange(c0y-1, c2y+1):
        for x in xrange(c0x-1, c1x+1):
            uv2bar(tcoords[tc,0], tcoords[tc,1],
                   tcoords[tc,2], tcoords[tc,3],
                   tcoords[tc,4], tcoords[tc,5],
                   <float>(x)/d, 1-<float>(y)/d,
                   bar)


            vx = (bar[0]*v0x + bar[1]*v1x + bar[2]*v2x)
            vy = (bar[0]*v0y + bar[1]*v1y + bar[2]*v2y)
            vz = (bar[0]*v0z + bar[1]*v1z + bar[2]*v2z)

            px = vx/spacing[0]
            py = vy/spacing[1]
            pz = vz/spacing[2]

            inx = (bar[0]*n0x + bar[1]*n1x + bar[2]*n2x)
            iny = (bar[0]*n0y + bar[1]*n1y + bar[2]*n2y)
            inz = (bar[0]*n0z + bar[1]*n1z + bar[2]*n2z)
            inn = sqrt(inx*inx + iny*iny + inz*inz)
            inx = inx / inn
            iny = iny / inn
            inz = inz / inn

            tnx = (interpolate(volume, px + hx, py, pz) - interpolate(volume, px - hx, py, pz))/(2.0 * hx)
            tny = (interpolate(volume, px, py + hy, pz) - interpolate(volume, px, py - hy, pz))/(2.0 * hy)
            tnz = (interpolate(volume, px, py, pz + hz) - interpolate(volume, px, py, pz - hz))/(2.0 * hz)

            tnn = sqrt(tnx*tnx + tny*tny + tnz*tnz)
            # g = <np.uint8_t>((gv - tmin) * (255.0/(tmax - tmin)))
            gv = get_LUT_value(interpolate(volume, px, py, pz), ww, wl)

            image[y, x, 0] = clut[<np.uint8_t>gv, 0]
            image[y, x, 1] = clut[<np.uint8_t>gv, 1]
            image[y, x, 2] = clut[<np.uint8_t>gv, 2]

            tnormals[y, x, 0] = <np.uint8_t>(((tnx / tnn) + 1.0) * 255.0/2.0)
            tnormals[y, x, 1] = <np.uint8_t>(((tny / tnn) + 1.0) * 255.0/2.0)
            tnormals[y, x, 2] = <np.uint8_t>(((tnz / tnn) + 1.0) * 255.0/2.0)

            raycasting_mida(volume, image, clut, ww, wl, x, y, px, py, pz, inx, iny, inz, spacing, 50.0, 50)


@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.cdivision(True)
def generate_tcoords_color(np.float64_t[:, :] vertices, np.float64_t[:, :] normals, np.int32_t[:, :] faces, image_t[:, :, :] volume, np.float64_t[:] spacing, int ww, int wl, np.uint8_t[:, :] clut, int offset=2, int dim=5000):
    cdef int n = faces.shape[0]
    cdef int nx = <int>sqrt(n)
    cdef int ny = <int>(ceil(n / <float>(nx)))

    cdef int d = dim
    cdef double dtx = d / <float>(nx)
    cdef double dty = d / <float>(ny)

    cdef np.ndarray[np.uint8_t, ndim=3] image = np.zeros(shape=(d, d, 3), dtype='uint8')
    cdef np.ndarray[np.uint8_t, ndim=3] tnormals = np.zeros(shape=(d, d, 3), dtype='uint8')
    cdef np.ndarray[np.float64_t, ndim=2] tcoords = np.zeros(shape=(faces.shape[0], 6), dtype='float64')

    cdef int tc = 0

    for tc in xrange(n):
        print ">>>>>>>>>", tc
        _generate_tcoords_color(vertices, normals, faces, volume, spacing, ww, wl, clut, offset, dim, tcoords, image, tnormals, tc, nx, dtx, dty, d)

    print "returned"
    return tcoords, image, tnormals


@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.cdivision(True)
cdef void _generate_tcoords_hf(np.float64_t[:, :] vertices, np.float64_t[:, :] normals, np.int32_t[:, :] faces, image_t[:, :, :] volume, np.float64_t[:] spacing, int ww, int wl, np.uint8_t[:, :] clut, int offset, int dim, np.float64_t[:, :] tcoords, image_t[:, :, :] image, np.uint8_t[:, :, :] tnormals, int tc, int nx, double dtx, double dty, int d, int nslices) nogil:

    cdef double tnx, tny, tnz, tnn
    cdef double hx, hy, hz
    hx = 1.0 #spacing[0]
    hy = 1.0 #spacing[1]
    hz = 1.0 #spacing[2]

    cdef np.float64_t[3] bar
    cdef double v0x, v0y, v0z
    cdef double v1x, v1y, v1z
    cdef double v2x, v2y, v2z

    # cdef np.float64_t[:] n0, n1, n2
    cdef double n0x, n0y, n0z
    cdef double n1x, n1y, n1z
    cdef double n2x, n2y, n2z
    cdef double gv
    cdef np.uint8_t g
    cdef int i, j
    cdef int x, y

    cdef int c0x, c0y, c1x, c1y, c2x, c2y
    cdef double vx, vy, vz
    cdef double px, py, pz
    cdef double inx, iny, inz, inn

    i = tc % nx
    j = tc / nx
    c0x = <int>(i * dtx + offset)
    c0y = <int>(j * dty + offset)

    c1x = <int>(c0x + dtx - offset)
    c1y = <int>(c0y)

    c2x = <int>((c0x + c1x) / 2.0)
    c2y = <int>(c0y + dty - offset)

    tcoords[tc, 0] = <float>(c0x)/d
    tcoords[tc, 1] = 1-<float>(c0y)/d
    tcoords[tc, 2] = <float>(c1x)/d
    tcoords[tc, 3] = 1-<float>(c1y)/d
    tcoords[tc, 4] = <float>(c2x)/d
    tcoords[tc, 5] = 1-<float>(c2y)/d


    v0x = vertices[faces[tc, 0], 0]
    v0y = vertices[faces[tc, 0], 1]
    v0z = vertices[faces[tc, 0], 2]

    v1x = vertices[faces[tc, 1], 0]
    v1y = vertices[faces[tc, 1], 1]
    v1z = vertices[faces[tc, 1], 2]

    v2x = vertices[faces[tc, 2], 0]
    v2y = vertices[faces[tc, 2], 1]
    v2z = vertices[faces[tc, 2], 2]

    n0x = normals[faces[tc, 0], 0]
    n0y = normals[faces[tc, 0], 1]
    n0z = normals[faces[tc, 0], 2]

    n1x = normals[faces[tc, 1], 0]
    n1y = normals[faces[tc, 1], 1]
    n1z = normals[faces[tc, 1], 2]

    n2x = normals[faces[tc, 2], 0]
    n2y = normals[faces[tc, 2], 1]
    n2z = normals[faces[tc, 2], 2]


    for y in xrange(c0y-1, c2y+1):
        for x in xrange(c0x-1, c1x+1):
            uv2bar(tcoords[tc,0], tcoords[tc,1],
                   tcoords[tc,2], tcoords[tc,3],
                   tcoords[tc,4], tcoords[tc,5],
                   <float>(x)/d, 1-<float>(y)/d,
                   bar)


            vx = (bar[0]*v0x + bar[1]*v1x + bar[2]*v2x)
            vy = (bar[0]*v0y + bar[1]*v1y + bar[2]*v2y)
            vz = (bar[0]*v0z + bar[1]*v1z + bar[2]*v2z)

            px = vx#/spacing[0]
            py = vy#/spacing[1]
            pz = vz#/spacing[2]

            inx = (bar[0]*n0x + bar[1]*n1x + bar[2]*n2x)
            iny = (bar[0]*n0y + bar[1]*n1y + bar[2]*n2y)
            inz = (bar[0]*n0z + bar[1]*n1z + bar[2]*n2z)
            inn = sqrt(inx*inx + iny*iny + inz*inz)
            inx = inx / inn
            iny = iny / inn
            inz = inz / inn

            tnx = (interpolate(volume, px + hx, py, pz) - interpolate(volume, px - hx, py, pz))/(2.0 * hx)
            tny = (interpolate(volume, px, py + hy, pz) - interpolate(volume, px, py - hy, pz))/(2.0 * hy)
            tnz = (interpolate(volume, px, py, pz + hz) - interpolate(volume, px, py, pz - hz))/(2.0 * hz)

            tnn = sqrt(tnx*tnx + tny*tny + tnz*tnz)
            # g = <np.uint8_t>((gv - tmin) * (255.0/(tmax - tmin)))
            gv = get_LUT_value(interpolate(volume, px, py, pz), ww, wl)

            # image[y, x, 0] = clut[<np.uint8_t>gv, 0]
            # image[y, x, 1] = clut[<np.uint8_t>gv, 1]
            # image[y, x, 2] = clut[<np.uint8_t>gv, 2]

            tnormals[y, x, 0] = <np.uint8_t>(((tnx / tnn) + 1.0) * 255.0/2.0)
            tnormals[y, x, 1] = <np.uint8_t>(((tny / tnn) + 1.0) * 255.0/2.0)
            tnormals[y, x, 2] = <np.uint8_t>(((tnz / tnn) + 1.0) * 255.0/2.0)

            raycasting2hf(volume, image, clut, ww, wl, x, y, px, py, pz, inx, iny, inz, spacing, 5.0, nslices)


@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.cdivision(True)
def generate_tcoords_hf(np.float64_t[:, :] vertices, np.float64_t[:, :] normals, np.int32_t[:, :] faces, image_t[:, :, :] volume, np.float64_t[:] spacing, int ww, int wl, np.uint8_t[:, :] clut, int offset=2, int dim=5000, int nslices=5):
    cdef int n = faces.shape[0]
    cdef int nx = <int>sqrt(n)
    cdef int ny = <int>(ceil(n / <float>(nx)))

    cdef int d = dim
    cdef double dtx = d / <float>(nx)
    cdef double dty = d / <float>(ny)

    cdef np.ndarray[image_t, ndim=3] image = np.zeros(shape=(nslices, d, d), dtype='int16')
    cdef np.ndarray[np.uint8_t, ndim=3] tnormals = np.zeros(shape=(d, d, 3), dtype='uint8')
    cdef np.ndarray[np.float64_t, ndim=2] tcoords = np.zeros(shape=(faces.shape[0], 6), dtype='float64')

    image[:] = NULL_VALUE

    cdef int tc = 0

    for tc in xrange(n):
        print ">>>>>>>>>", tc
        _generate_tcoords_hf(vertices, normals, faces, volume, spacing, ww, wl, clut, offset, dim, tcoords, image, tnormals, tc, nx, dtx, dty, d, nslices)

    print "returned"
    return tcoords, image, tnormals
