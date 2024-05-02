#!/usr/bin/env python
# -*- coding: utf-8 -*-

#--------------------------------------------------------------------------
# Software:     InVesalius - Software de Reconstrucao 3D de Imagens Medicas
# Copyright:    (C) 2001  Centro de Pesquisas Renato Archer
# Homepage:     http://www.softwarepublico.gov.br
# Contact:      invesalius@cti.gov.br
# License:      GNU - GPL 2 (LICENSE.txt/LICENCA.txt)
#--------------------------------------------------------------------------
#    Este programa e software livre; voce pode redistribui-lo e/ou
#    modifica-lo sob os termos da Licenca Publica Geral GNU, conforme
#    publicada pela Free Software Foundation; de acordo com a versao 2
#    da Licenca.
#
#    Este programa eh distribuido na expectativa de ser util, mas SEM
#    QUALQUER GARANTIA; sem mesmo a garantia implicita de
#    COMERCIALIZACAO ou de ADEQUACAO A QUALQUER PROPOSITO EM
#    PARTICULAR. Consulte a Licenca Publica Geral GNU para obter mais
#    detalhes.
#--------------------------------------------------------------------------

import itertools
import os
import plistlib
import sys

import numpy as np
import wx
import vtk
from vtk.wx.wxVTKRenderWindowInteractor import wxVTKRenderWindowInteractor
from vtk.util import numpy_support
from wx.lib.pubsub import pub as Publisher

import matplotlib.pylab as plt

from skimage.draw import ellipse, polygon
from scipy.misc import imsave

import constants as const
import converters

import mips

class Viewer(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent, size=wx.Size(320, 320))
        self.SetBackgroundColour(wx.Colour(0, 0, 0))
        self.colorscheme = None
        self._init_gui()

        self.mtype = 1
        self.ww = 750
        self.wl = 750
        self.nodes = []
        self._use = 'WWWL'
        self._nslice = 0

    def _init_gui(self):
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.style = style

        interactor = wxVTKRenderWindowInteractor(self, -1, size=self.GetSize())
        interactor.SetInteractorStyle(style)
        self.interactor = interactor

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(interactor, 1, wx.EXPAND)
        self.sizer = sizer
        self.SetSizer(sizer)
        self.Layout()

        # It would be more correct (API-wise) to call interactor.Initialize() and
        # interactor.Start() here, but Initialize() calls RenderWindow.Render().
        # That Render() call will get through before we can setup the
        # RenderWindow() to render via the wxWidgets-created context; this
        # causes flashing on some platforms and downright breaks things on
        # other platforms.  Instead, we call widget.Enable().  This means
        # that the RWI::Initialized ivar is not set, but in THIS SPECIFIC CASE,
        # that doesn't matter.
        interactor.Enable(1)

        ren = vtk.vtkRenderer()
        ren.SetLayer(0)
        #  ren.LightFollowCameraOn()
        ren.SetBackground( .6,.6,.75)
        self.ren = ren
        ren.EraseOn()

        lk  = vtk.vtkLightKit()
        lk.AddLightsToRenderer(ren)

        ren2 = vtk.vtkRenderer()
        ren2.SetInteractive(0)
        ren2.SetLayer(1)
        ren2.EraseOff()
        self.ren2 = ren2

        #  cam = ren2.GetActiveCamera()
        #  cam.ParallelProjectionOn()

        #  self.set_view_angle(const.VOL_ISO)

        #  interactor.GetRenderWindow().SetNumberOfLayers(2)
        interactor.GetRenderWindow().AddRenderer(ren)
        #  interactor.GetRenderWindow().AddRenderer(ren2)

    def add_polydata(self, polydata, image):
        self.polydata = polydata

        m = vtk.vtkPolyDataMapper()
        m.SetInputData(polydata)
        m.ScalarVisibilityOff()

        #  cimage = self.do_ww_wl(image)

        texture = vtk.vtkTexture()
        texture.SetInputData(image)
        texture.InterpolateOn()

        self.texture = texture
        self.image = image

        prop = vtk.vtkProperty()
        prop.SetInterpolationToPhong()

        actor = vtk.vtkActor()
        actor.SetMapper(m)
        actor.SetTexture(texture)
        actor.SetProperty(prop)
        #  actor.SetProperty(vp)

        self.ren.AddActor(actor)

        self.set_view_angle(const.VOL_ISO)
        self.ren.ResetCamera()
        self.ren.ResetCameraClippingRange()
        self.interactor.Render()
        return actor

    def add_polydata_hf(self, polydata, image):
        self.polydata = polydata

        dx, dy, dz = image.GetDimensions()
        timage = numpy_support.vtk_to_numpy(image.GetPointData().GetScalars())
        timage.shape = dz, dy, dx

        self.timage = timage

        m = vtk.vtkPolyDataMapper()
        m.SetInputData(polydata)
        m.ScalarVisibilityOff()

        self.image = image

        prop = vtk.vtkProperty()
        prop.SetInterpolationToPhong()

        actor = vtk.vtkActor()
        actor.SetMapper(m)
        actor.SetProperty(prop)
        #  actor.SetProperty(vp)
        self.actor = actor

        self.gen_texture()


        self.ren.AddActor(actor)

        self.set_view_angle(const.VOL_ISO)
        self.ren.ResetCamera()
        self.ren.ResetCameraClippingRange()
        self.interactor.Render()
        return actor

    def add_volume(self, img, spacing, color, isovalue=127, reset_camera=True):
        #  vimg = converters.to_vtk((img - img.min()).astype('uint16'), spacing, 0, 'AXIAL')
        vimg = converters.to_vtk(img, spacing, 0, 'AXIAL')

        cf = vtk.vtkColorTransferFunction()
        cf.RemoveAllPoints()
        cf.AddRGBPoint(0.0, *color)
        cf.AddRGBPoint(255.0, *color)
        #  cf.AddRGBPoint(2000.0, *color)

        pf = vtk.vtkPiecewiseFunction()
        pf.RemoveAllPoints()
        pf.AddPoint(0.0, 0.0)
        pf.AddPoint(isovalue, 0.0)

        vp= vtk.vtkVolumeProperty()
        vp.SetColor(cf)
        vp.SetScalarOpacity(pf)
        vp.ShadeOn()
        vp.SetInterpolationTypeToLinear()

        isosurfaceFunc = vtk.vtkVolumeRayCastIsosurfaceFunction()
        isosurfaceFunc.SetIsoValue(isovalue)

        volumeRayCastMapper = vtk.vtkVolumeRayCastMapper()
        volumeRayCastMapper.SetInputData(vimg)
        volumeRayCastMapper.SetVolumeRayCastFunction(isosurfaceFunc)

        actor = vtk.vtkVolume()
        actor.SetMapper(volumeRayCastMapper)
        actor.SetProperty(vp)

        self.ren.AddActor(actor)

        if reset_camera:
            self.set_view_angle(const.VOL_ISO)

            self.ren.ResetCamera()
            self.ren.ResetCameraClippingRange()

        self.interactor.Render()

        return actor

    def add_volume_rc(self, img, spacing, color, isovalue=127, reset_camera=True):
        #  vimg = converters.to_vtk((img - img.min()).astype('uint16'), spacing, 0, 'AXIAL')
        vimg = converters.to_vtk(img, spacing, 0, 'AXIAL')

        cf = vtk.vtkColorTransferFunction()
        cf.RemoveAllPoints()
        #  cf.AddRGBPoint(0.0, *color)
        #  cf.AddRGBPoint(255.0, *color)
        #  cf.AddRGBPoint(2000.0, *color)

        pf = vtk.vtkPiecewiseFunction()
        pf.RemoveAllPoints()
        #  pf.AddPoint(0.0, 0.0)
        #  pf.AddPoint(isovalue, 1.0)

        vp= vtk.vtkVolumeProperty()
        vp.SetColor(cf)
        vp.SetScalarOpacity(pf)
        vp.ShadeOn()
        vp.SetInterpolationTypeToLinear()

        raycasting_function = vtk.vtkVolumeRayCastCompositeFunction()
        raycasting_function.SetCompositeMethodToInterpolateFirst()
        #  isosurfaceFunc.SetIsoValue(isovalue)
 
        volumeRayCastMapper = vtk.vtkFixedPointVolumeRayCastMapper()
        volumeRayCastMapper.SetInputData(vimg)
        #  volumeRayCastMapper.SetVolumeRayCastFunction(raycasting_function)
        volumeRayCastMapper.SetBlendModeToComposite()


        actor = vtk.vtkVolume()
        actor.SetMapper(volumeRayCastMapper)
        actor.SetProperty(vp)

        self.ren.AddActor(actor)

        self.cf = cf
        self.pf = pf

        if reset_camera:
            self.set_view_angle(const.VOL_ISO)

            self.ren.ResetCamera()
            self.ren.ResetCameraClippingRange()

        self.interactor.Render()

        self.vimg = vimg
        return actor

    def remove_volume(self, actor):
        self.ren.RemoveActor(actor)
        del actor

    def set_colorscheme(self, fname):
        p = plistlib.readPlist(fname)
        self.nodes = zip(p['Red'], p['Green'], p['Blue'])

    def do_ww_wl(self, image):
        if self.nodes:

            if self._use == 'widget':
                snodes = sorted(self.nodes)
                lut  = vtk.vtkColorTransferFunction()

                for n in self.nodes:
                    r, g, b = n.colour
                    lut.AddRGBPoint(n.value, r/255.0, g/255.0, b/255.0)

                lut.Build()

                colorer = vtk.vtkImageMapToColors()
                colorer.SetLookupTable(lut)
                colorer.SetInputData(image)
                colorer.SetOutputFormatToRGB()
                colorer.Update()
            else:
                lut = vtk.vtkWindowLevelLookupTable()
                lut.SetWindow(self.ww)
                lut.SetLevel(self.wl)
                lut.Build()
                for i, (r, g, b) in enumerate(self.nodes):
                    lut.SetTableValue(i, r/255.0, g/255.0, b/255.0, 1.0)

                colorer = vtk.vtkImageMapToColors()
                colorer.SetInputData(image)
                colorer.SetLookupTable(lut)
                colorer.SetOutputFormatToRGB()
                colorer.Update()
        else:
            colorer = vtk.vtkImageMapToWindowLevelColors()
            colorer.SetInputData(image)
            colorer.SetWindow(self.ww)
            colorer.SetLevel(self.wl)
            colorer.SetOutputFormatToRGB()
            colorer.Update()

        return colorer.GetOutput()

    def gen_texture(self):
        timage = self.timage

        dz, dy, dx = timage.shape
        spacing = (1, 1, 1)

        ww = self.ww
        wl = self.wl

        mimage = np.zeros(shape=(dy, dx), dtype='int16')
        if self.mtype == 7:
            mimage = np.zeros(shape=(dy, dx, 3), dtype='uint8')
            #  print np.array(self.nodes, dtype='uint8')
            mips.raycasting(timage, np.array(self.nodes, dtype='uint8'), self.wl, self.ww, mimage)
            print 'Aqui, manolO!'
            imsave('/tmp/fatia.png', mimage[::-1])
            #  cimage = converters.to_vtk_color(mimage, spacing, 0, 'AXIAL')
            r = vtk.vtkPNGReader()
            r.SetFileName('/tmp/fatia.png')
            r.Update()

            cimage = r.GetOutput()
            print cimage
        else:
            if self.mtype == 0:
                print "Slices"
                mimage[:] = timage[self._nslice, :, :]
            elif self.mtype == 1:
                mimage[:] = timage[1::].max(0)
            elif self.mtype == 2:
                mimage[:] = timage.min(0)
            elif self.mtype == 3:
                mimage[:] = timage.mean(0)
            elif self.mtype == 4:
                mips.mida(timage, 0, wl, ww, mimage)
            elif self.mtype == 5:
                mips.fast_countour_mip(timage,
                              1.0,
                              0,
                              wl, ww,
                              0, #tmip
                              mimage)
            elif self.mtype == 6:
                mips.fast_countour_mip(timage,
                              0.1,
                              0,
                              wl, ww,
                              2, #tmip
                              mimage)

            #  mimage[:] = timage.max(0)

            #  imsave('/tmp/fatia.png', mimage)
            vimg = converters.to_vtk(mimage, spacing, 0, 'AXIAL')
            cimage = self.do_ww_wl(vimg)

            print ">>>>", mimage.max(), mimage.min()
            print ">>>", vimg.GetScalarRange()
            print ">>>", cimage.GetScalarRange()

            self.mimage = mimage


        texture = vtk.vtkTexture()
        texture.SetInputData(cimage)
        texture.InterpolateOn()
        self.actor.SetTexture(texture)

        self.texture = texture

    def set_wlww(self, wl, ww):
        self.wl = wl
        self.ww = ww
        #  cimage = self.do_ww_wl(self.image)
        #  self.texture.SetInput(cimage)
        self.gen_texture()
        self.interactor.Render()

    def do_colour_mask(self, imagedata):
        scalar_range = int(imagedata.GetScalarRange()[1])
        r,g,b = 0, 1, 0

        # map scalar values into colors
        lut_mask = vtk.vtkLookupTable()
        lut_mask.SetNumberOfColors(256)
        lut_mask.SetHueRange(const.THRESHOLD_HUE_RANGE)
        lut_mask.SetSaturationRange(1, 1)
        lut_mask.SetValueRange(0, 255)
        lut_mask.SetRange(0, 255)
        lut_mask.SetNumberOfTableValues(256)
        lut_mask.SetTableValue(0, 0, 0, 0, 0.0)
        lut_mask.SetTableValue(1, 0, 0, 0, 0.0)
        lut_mask.SetTableValue(2, 0, 0, 0, 0.0)
        lut_mask.SetTableValue(127, 0, 0, 0, 0.5)
        lut_mask.SetTableValue(255, 1, 0, 0, 0.5)
        lut_mask.SetRampToLinear()
        lut_mask.Build()
        # self.lut_mask = lut_mask

        # map the input image through a lookup table
        img_colours_mask = vtk.vtkImageMapToColors()
        img_colours_mask.SetLookupTable(lut_mask)
        img_colours_mask.SetOutputFormatToRGBA()
        img_colours_mask.SetInputData(imagedata)
        #  img_colours_mask.Update()
        # self.img_colours_mask = img_colours_mask

        return img_colours_mask

    def set_camera_pos(self, pos):
        coord_camera = np.array(pos)
        #coord_camera = np.array(bases.FlipX(coord_camera))

        cam = self.ren.GetActiveCamera()

        if self.initial_foco is None:
            self.initial_foco = np.array(cam.GetFocalPoint())

        cam_initialposition = np.array(cam.GetPosition())
        cam_initialfoco = np.array(cam.GetFocalPoint())

        cam_sub = cam_initialposition - cam_initialfoco
        cam_sub_norm = np.linalg.norm(cam_sub)
        vet1 = cam_sub/cam_sub_norm

        vet2 = coord_camera + cam_sub

        #cam_sub_novo = coord_camera - self.initial_foco
        #cam_sub_novo_norm = np.linalg.norm(cam_sub_novo)
        #vet2 = cam_sub_novo/cam_sub_novo_norm
        #vet2 = vet2*cam_sub_norm + coord_camera

        #if self.is_inside_model(vet2):
            #vet2 *= -1

        cam.SetFocalPoint(coord_camera)
        cam.SetPosition(vet2)


    def set_view_angle(self, view):
        cam = self.ren.GetActiveCamera()
        cam.SetFocalPoint(0, 0, 0)

        xv, yv, zv = const.VOLUME_POSITION[const.AXIAL][0][view]
        xp, yp, zp = const.VOLUME_POSITION[const.AXIAL][1][view]

        cam.SetViewUp(xv, yv, zv)
        cam.SetPosition(xp, yp, zp)

        self.ren.ResetCameraClippingRange()
        self.ren.ResetCamera()
        self.interactor.Render()

    def save_obj(self, fname):
        w = vtk.vtkOBJExporter()
        w.SetInput(self.interactor.GetRenderWindow())
        w.SetFilePrefix(fname)
        w.Write()

        cimage = self.texture.GetInput()
        wi = vtk.vtkPNGWriter()
        wi.SetInputData(cimage)
        wi.SetFileName('/tmp/manolo.png')
        wi.Write()
        print "Salvo!"
