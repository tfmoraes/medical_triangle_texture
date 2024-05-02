import colorsys
import itertools
import random
import sys
import plistlib

import numpy as np
import vtk

from optparse import OptionParser

from scipy.misc import imsave
from skimage.draw import polygon
from vtk.util import numpy_support

import converters
import interpolate
from ply_writer import PlyWriter

def parse_cmd_line():
    parser = OptionParser()
    parser.add_option('-i', '--input', dest='input',
                      help='PLY input')

    parser.add_option('-t', '--texture', dest='texture',
                      help='memmap file')

    parser.add_option('-o', '--output', dest='output',
                      help='PLY output')

    parser.add_option('-c', '--clut', dest='clut',
                      help='clut')

    parser.add_option('-s', '--spacing', dest='spacing', type="float", nargs=3,
                      default = (1, 1, 1), help='Spacing x, y and z')

    parser.add_option('-d', '--dimensions', dest='dimensions', type="int", nargs=3,
                      default = (100, 100, 100), help='Dimension x, y and z')

    parser.add_option('-p', '--dtexture', dest='dtexture', type="int",
                      default = 5000, help='Texture dimension in pixels')

    parser.add_option('--ww', dest='ww', type='int', default=255)
    parser.add_option('--wl', dest='wl', type='int', default=255)

    options, args = parser.parse_args()
    print options
    return options


def read_ply(fname):
    r = vtk.vtkPLYReader()
    r.SetFileName(fname)
    r.Update()

    n = vtk.vtkPolyDataNormals()
    n.SetInput(r.GetOutput())
    n.ComputePointNormalsOn()
    n.Update()

    c = vtk.vtkCleanPolyData()
    c.SetInput(n.GetOutput())
    c.Update()

    vertices = numpy_support.vtk_to_numpy(c.GetOutput().GetPoints().GetData())
    normals = numpy_support.vtk_to_numpy(c.GetOutput().GetPointData().GetNormals())
    faces = numpy_support.vtk_to_numpy(c.GetOutput().GetPolys().GetData())
    faces.shape = -1, 4
    faces = faces[:, 1::]

    return vertices, normals, faces


def uv2bar(p1, p2, p3, f):
    f1 = p1 - f
    f2 = p2 - f
    f3 = p3 - f

    a = np.linalg.norm(np.cross(p1-p2, p1-p3))
    a1 = np.linalg.norm(np.cross(f2, f3)) / a
    a2 = np.linalg.norm(np.cross(f3, f1)) / a
    a3 = np.linalg.norm(np.cross(f1, f2)) / a

    return a1, a2, a3

def generate_tcoords(vertices, faces, volume, spacing, offset=2):
    n = faces.shape[0]
    nx = int(n ** 0.5)
    ny = int(np.ceil(n / float(nx)))

    d = 10000
    dtx = d / float(nx)
    dty = d / float(ny)

    image = np.zeros(shape=(d, d, 3), dtype='uint8')
    tcoords = np.zeros(shape=(faces.shape[0], 6), dtype='float64')
    tmin = volume.min()
    tmax = volume.max()

    tc = 0
    for j in xrange(ny):
        for i in xrange(nx):
            c0x = int(i * dtx + offset)
            c0y = int(j * dty + offset)

            c1x = int(c0x + dtx - offset)
            c1y = int(c0y)

            c2x = int((c0x + c1x) / 2)
            c2y = int(c0y + dty - offset)

            tcoords[tc] = (float(c0x)/d, 1-float(c0y)/d, float(c1x)/d, 1-float(c1y)/d, float(c2x)/d, 1-float(c2y)/d)

            # Drawing polygon
            #  print tc, n, nx, ny, uv2bar(np.array((tcoords[tc][0], tcoords[tc][1])),
                                       #  np.array((tcoords[tc][2], tcoords[tc][3])),
                                       #  np.array((tcoords[tc][4], tcoords[tc][5])),
                                       #  np.array((tcoords[tc][0], tcoords[tc][1])))
            rr, cc = polygon(np.array((c0y, c1y, c2y)), np.array((c0x, c1x, c2x)))
            sys.stdout.write('\r %.3f' % (100*float(tc)/n))
            sys.stdout.flush()

            for y, x in  zip(rr, cc):
                a, b, c = uv2bar(np.array((tcoords[tc][0], tcoords[tc][1])),
                                       np.array((tcoords[tc][2], tcoords[tc][3])),
                                       np.array((tcoords[tc][4], tcoords[tc][5])),
                                       np.array((float(x)/d, 1-float(y)/d)))
                v0 = vertices[faces[tc, 0]]
                v1 = vertices[faces[tc, 1]]
                v2 = vertices[faces[tc, 2]]

                v = a*v0 + b*v1 + c*v2
                vx, vy, vz = v / spacing

                gv = interpolate(volume, vx, vy, vz)
                g = (gv - tmin) * (255.0/(tmax - tmin))

                image[y, x, 0] = g
                image[y, x, 1] = g
                image[y, x, 2] = g

            #  r,g,b = colorsys.hsv_to_rgb(tc / float(n-1), 0.5, 0.5)
            #  image[rr, cc, 0] = r * 255
            #  image[rr, cc, 1] = g * 255
            #  image[rr, cc, 2] = b * 255

            tc+=1
            if tc == n:
                return tcoords, image

    return tcoords, image

def main():
    options = parse_cmd_line()
    ifile = options.input
    ofile = options.output
    tfile = options.texture
    otfile = ofile.replace('.ply', '.png')
    tnfile = ofile.replace('.ply', '_n.png')
    dim = options.dimensions[::-1]
    dtexture=options.dtexture
    spacing = np.array(options.spacing, dtype='float64')
    cfile = options.clut
    ww = options.ww
    wl = options.wl

    p = plistlib.readPlist(cfile)
    clut = np.array(zip(p['Red'], p['Green'], p['Blue']), dtype='uint8')


    tdata = np.memmap(tfile, shape=dim, dtype='int16')
    vertices, normals, faces = read_ply(ifile)
    print vertices.shape, normals.shape, faces.shape
    tcoords, image, tnormals = interpolate.generate_tcoords_color(vertices.astype('float64'), normals.astype('float64'), faces.astype('int32'), tdata, spacing, ww, wl, clut, dim=dtexture)
    imsave(otfile, image[::-1, :])
    #  vimg = converters.to_vtk(image, spacing, 0, 'AXIAL')
    #  w = vtk.vtkPNGWriter()
    #  w.SetFileName(otfile)
    #  w.SetInput(vimg)
    #  w.Write()

    imsave(tnfile, tnormals[::-1, :])

    w = PlyWriter(ofile)
    w.from_faces_vertices_list(faces, vertices, normals, tcoords, otfile)

if __name__ == "__main__":
    main()
