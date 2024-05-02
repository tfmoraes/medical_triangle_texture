import numpy as np
import sys
import wx
import vtk
from vtk.util import numpy_support

import ply_reader
import volume_viewer
import clut_imagedata

class Window(wx.Frame):
    def __init__(self):
        """TODO: to be defined1. """
        wx.Frame.__init__(self, None, -1, "Alinhador")
        self._init_gui()
        self.Show()

    def _init_gui(self):
        self.viewer = volume_viewer.Viewer(self)

        vw3_sizer = wx.BoxSizer(wx.HORIZONTAL)
        vw3_sizer.Add(self.viewer, 1, wx.EXPAND)

        vsizer = wx.BoxSizer(wx.HORIZONTAL)
        vsizer.Add(vw3_sizer, 1, wx.EXPAND)


        self.spin_wl = wx.SpinCtrl(self, value='127.0', min=-2000, max=2000)
        self.spin_ww = wx.SpinCtrl(self, value='255.0', min=-5000, max=5000)

        btn_save = wx.Button(self, -1, 'Salvar')

        csizer = wx.GridBagSizer(4, 3)

        csizer.Add(wx.StaticText(self, -1, "WW\&WL"), (0, 0))
        csizer.Add(self.spin_wl, (1, 0))
        csizer.Add(self.spin_ww, (1, 1))

        csizer.Add(btn_save, (3, 0))

        self.csizer = csizer

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(vsizer, 1, wx.EXPAND)
        sizer.Add(csizer, 0, wx.EXPAND)

        self.spin_wl.Bind(wx.EVT_SPINCTRL, self.OnWLWW)
        self.spin_ww.Bind(wx.EVT_SPINCTRL, self.OnWLWW)
        btn_save.Bind(wx.EVT_BUTTON, self.OnSave)

        self.SetSizerAndFit(sizer)

    def load_model(self, mesh, texture):
        self.viewer.add_polydata(mesh, texture)

        #  dx, dy, dz = texture.GetDimensions()
        #  timage = numpy_support.vtk_to_numpy(texture.GetPointData().GetScalars())
        #  timage.shape = dz, dy, dx

        #  histogram, s = np.histogram(timage, bins=(timage.max() - timage.min()))

        #  self.clut = clut_imagedata.CLUTImageDataWidget(self, -1, histogram, timage.min(), timage.max())
        #  self.csizer.Add(self.clut, (2, 0), (1, 3))
        #  self.Layout()

        #  self.clut.Bind(clut_imagedata.EVT_CLUT_NODE_CHANGED, self.OnClut)

    def OnWLWW(self, evt):
        ww = self.spin_ww.GetValue()
        wl = self.spin_wl.GetValue()
        self.viewer._use = 'WWWL'
        self.viewer.set_wlww(wl, ww)

    def OnClut(self, evt):
        self.viewer.nodes = evt.GetNodes()
        ww = self.spin_ww.GetValue()
        wl = self.spin_wl.GetValue()
        self.viewer._use = 'widget'
        self.viewer.set_wlww(wl, ww)

    def OnSave(self, evt):
        self.viewer.save_obj('/tmp/manolo')


def load_model(viewer):
    r = ply_reader.PlyReader(sys.argv[1])
    r.read()

    svertices = []
    snormals = []
    stcoords = []

    triangles = vtk.vtkCellArray()

    for n, (v0, v1, v2) in enumerate(r.faces):
        svertices.append(r.vertices[v0])
        svertices.append(r.vertices[v1])
        svertices.append(r.vertices[v2])

        snormals.append(r.normals[v0])
        snormals.append(r.normals[v1])
        snormals.append(r.normals[v2])

        stcoords.append((r.tcoords[n][0], 1-r.tcoords[n][1]))
        stcoords.append((r.tcoords[n][2], 1-r.tcoords[n][3]))
        stcoords.append((r.tcoords[n][4], 1-r.tcoords[n][5]))

        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, n*3+0)
        triangle.GetPointIds().SetId(1, n*3+1)
        triangle.GetPointIds().SetId(2, n*3+2)
        triangles.InsertNextCell(triangle)

    vertices = np.array(svertices)
    normals = np.array(snormals)
    tcoords = np.array(stcoords)

    points = vtk.vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(vertices))

    mesh = vtk.vtkPolyData()
    mesh.SetPoints(points)
    mesh.SetPolys(triangles)

    vnormals = numpy_support.numpy_to_vtk(normals)
    vnormals.SetNumberOfComponents(3)
    mesh.GetPointData().SetNormals(vnormals)

    vtcoords = numpy_support.numpy_to_vtk(tcoords)
    vtcoords.SetNumberOfComponents(2)
    mesh.GetPointData().SetTCoords(vtcoords)

    #  print "normals"
    #  normals = vtk.vtkPolyDataNormals()
    #  normals.SetInput(mesh)
    #  normals.ComputeCellNormalsOn()
    #  normals.ComputePointNormalsOn()
    #  normals.Update()

    #  print "Smooth"
    #  smoothFilter = vtk.vtkSmoothPolyDataFilter()
    #  smoothFilter.SetInputConnection(normals.GetOutputPort())
    #  smoothFilter.SetNumberOfIterations(5)
    #  smoothFilter.Update()

    pd = vtk.vtkPolyData()
    pd.DeepCopy(mesh)

    #  mapper = vtk.vtkPolyDataMapper()
    #  mapper.SetInput(pd)
    #  mapper.ScalarVisibilityOff()


    pr = vtk.vtkPNGReader()
    pr.SetFileName(sys.argv[2])
    pr.Update()

    #  texture = vtk.vtkTexture()
    #  texture.SetInput(pr.GetOutput())
    #  texture.InterpolateOn()

    #  prop = vtk.vtkProperty()
    #  prop.SetInterpolationToPhong()

    #  actor = vtk.vtkActor()
    #  actor.SetMapper(mapper)
    #  actor.SetTexture(texture)
    #  actor.SetProperty(prop)

    #  ren = vtk.vtkRenderer()
    #  ren.AddActor(actor)

    viewer.load_model(pd, pr.GetOutput())


def run_app():
    app = wx.App()
    window = Window()
    app.SetTopWindow(window)
    window.viewer.set_colorscheme(sys.argv[3])
    wx.CallLater(1, load_model, window)
    app.MainLoop()


def main():
    run_app()

if __name__ == "__main__":
    main()
