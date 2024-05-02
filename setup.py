import os
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy as np

ext_modules = [ Extension("interpolation", ["interpolation.pyx"],
                          include_dirs =  [np.get_include()],
                          extra_compile_args=['-fopenmp', '-O3', '-march=native', '-ffast-math', '-funroll-loops'],
                          extra_link_args=['-fopenmp'],),

               Extension("raycasting", ["raycasting.pyx"],
                          include_dirs =  [np.get_include()],
                          extra_compile_args=['-fopenmp', '-O3', '-march=native', '-ffast-math', '-funroll-loops'],
                          extra_link_args=['-fopenmp'],),

               Extension("gen_texture", ["gen_texture.pyx"],
                         include_dirs =  [np.get_include()],
                         extra_compile_args=['-fopenmp', '-O3', '-march=native', '-ffast-math', '-funroll-loops'],
                         extra_link_args=['-fopenmp'],),

               Extension("mips", ["mips.pyx"],
                         include_dirs =  [np.get_include()],
                         extra_compile_args=['-fopenmp', '-O3', '-march=native', '-ffast-math', '-funroll-loops'],
                         extra_link_args=['-fopenmp'],),

               ]


setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules,
)
