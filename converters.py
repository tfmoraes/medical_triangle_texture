# -*- coding: utf-8 -*-,
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

import numpy
import vtk
from vtk.util import numpy_support

def to_vtk(n_array, spacing, slice_number, orientation):
    try:
        dz, dy, dx = n_array.shape
    except ValueError:
        dy, dx = n_array.shape
        dz = 1

    v_image = numpy_support.numpy_to_vtk(n_array.flat)

    extent = (0, dx -1, 0, dy -1, 0,  dz - 1)

    # Generating the vtkImageData
    image = vtk.vtkImageData()
    image.SetOrigin(0, 0, 0)
    image.SetSpacing(spacing)
    image.SetDimensions(dx, dy, dz)
    # SetNumberOfScalarComponents e SetScalrType foi substituido por
    # AllocateScalars
    #  image.SetNumberOfScalarComponents(1)
    #  image.SetScalarType(numpy_support.get_vtk_array_type(n_array.dtype))
    image.AllocateScalars(numpy_support.get_vtk_array_type(n_array.dtype), 1)
    image.SetExtent(extent)
    image.GetPointData().SetScalars(v_image)
    # Não existe mais update
    #  image.Update()
    #  image.UpdateInformation()

    image_copy = vtk.vtkImageData()
    image_copy.DeepCopy(image)
    # Não existe mais update
    #  image_copy.Update()

    #  print image_copy.GetDimensions(), dx, dy, dz
    #  print image_copy

    return image_copy

def to_vtk_color(n_array, spacing, slice_number, orientation):
    try:
        dy, dx, dz = n_array.shape
    except ValueError:
        dy, dx = n_array.shape
        dz = 1

    v_image = numpy_support.numpy_to_vtk(n_array.flat)

    extent = (0, dx -1, 0, dy -1, 0, 0)

    # Generating the vtkImageData
    image = vtk.vtkImageData()
    image.SetOrigin(0, 0, 0)
    image.SetSpacing(1,1,1)
    image.SetNumberOfScalarComponents(3)
    image.SetDimensions(dx, dy, 1)
    image.SetExtent(extent)
    image.SetScalarType(numpy_support.get_vtk_array_type(n_array.dtype))
    #  image.AllocateScalars()
    image.GetPointData().SetScalars(v_image)
    image.Update()
    image.UpdateInformation()

    image_copy = vtk.vtkImageData()
    image_copy.DeepCopy(image)
    image_copy.Update()

    return image_copy
