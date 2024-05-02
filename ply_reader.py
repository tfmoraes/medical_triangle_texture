#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys

EVENT_VERTEX = 1
EVENT_FACE = 2
EVENT_HEADER = 3

class PlyReader(object):
    def __init__(self, filename):
        self.filename = filename
        self.vertices = []
        self.normals = []
        self.faces = []
        self.tcoords = []
        
    def read(self):
        with open(self.filename, 'r') as ply_file:
            for line in ply_file:
                # reading header
                if line.startswith('element vertex'):
                    n_vertex = int(line.split()[2])
                elif line.startswith('element face'):
                    n_faces = int(line.split()[2])
                elif line.startswith('end_header'):
                    break

            # reading vertex
            v_id = 0
            for line in ply_file:
                vertex = [float(v.replace(',', '.')) for v in line.split()][:3]
                normal = [float(v.replace(',', '.')) for v in line.split()][3:6]
                self.vertices.append(vertex)
                self.normals.append(normal)
                v_id += 1
                sys.stdout.write('\rReading Vertices: %.2f%%' % ((100.0*v_id)/n_vertex))
                sys.stdout.flush()
                if v_id == n_vertex:
                    break

            print

            # reading faces
            f_id = 0
            for line in ply_file:
                face = [int(v) for v in line.split()[1:4]]
                tcoord = [float(v.replace(',', '.')) for v in line.split()][5:11]
                self.faces.append(face)
                self.tcoords.append(tcoord)
                f_id += 1
                sys.stdout.write('\rReading faces: %.2f%%' % ((100.0*f_id)/n_faces))
                sys.stdout.flush()
                if f_id == n_faces:
                    break

            print

def main():
    import sys

    r = PlyReader(sys.argv[1])
    r.read()

if __name__ == "__main__":
    main()
