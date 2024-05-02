#!/usr/bin/env python
# -*- coding: UTF-8 -*-

class PlyWriter(object):
    def __init__(self, filename):
        self.filename = filename

    def _write_header(self, ply_file, n_vertices, n_faces, tfile):
        ply_file.write('ply\n')
        ply_file.write('format ascii 1.0\n')
        ply_file.write('comment VCGLIB generated\n')
        ply_file.write('comment TextureFile %s\n' % tfile)
        ply_file.write('element vertex %d\n' % n_vertices)
        ply_file.write('property float x\n')
        ply_file.write('property float y\n')
        ply_file.write('property float z\n')
        ply_file.write('property float nx\n')
        ply_file.write('property float ny\n')
        ply_file.write('property float nz\n')
        ply_file.write('element face %d\n' % n_faces)
        ply_file.write('property list uchar int vertex_indices\n')
        ply_file.write('property list uchar float texcoord\n')
        ply_file.write('property uchar red\n')
        ply_file.write('property uchar green\n')
        ply_file.write('property uchar blue\n')
        ply_file.write('property uchar alpha\n')
        ply_file.write('end_header\n')


    def from_corner_table(self, ct):
        with file(self.filename, 'w') as ply_file:
            self._write_header(ply_file, len(ct.vertices), len(ct.V)/3)
            for v in ct.vertices.values():
                ply_file.write(' '.join(['%f' % i for i in v]) + '\n')

            for c_id in xrange(0, len(ct.V), 3):
                cn = ct.next_corner(c_id)
                cp = ct.previous_corner(c_id)
                ply_file.write('3 %d %d %d\n' % (ct.V[c_id], ct.V[cn], ct.V[cp]))

    def from_faces_vertices_list(self, faces, vertices, normals, tcoords, tfile):
        with file(self.filename, 'w') as ply_file:
            self._write_header(ply_file, len(vertices), len(faces), tfile)

            for (x, y, z), (nx, ny, nz) in zip(vertices, normals):
                ply_file.write('%f %f %f %f %f %f\n' % (x, y, z, nx, ny, nz))

            for (vx, vy, vz), (t0x, t0y, t1x, t1y, t2x, t2y) in zip(faces, tcoords):
                ply_file.write('3 %d %d %d 6 %f %f %f %f %f %f 9 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8\n' % (vx, vy, vz, t0x, t0y, t1x, t1y, t2x, t2y))
