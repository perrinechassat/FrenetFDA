from setuptools import setup, Extension
from Cython.Build import cythonize
import sys
import os
import numpy
# from distutils.core import setup
# from distutils.extension import Extension

# sys.path.insert(1, 'Sources/')

# def get_c_sources(folder):
#     """Find all C/C++ source files in the `folder` directory."""
#     allowed_extensions = [".c", ".pyx"]
#     sources = []
#     for root, dirs, files in os.walk(folder):
#         for name in files:
#             ext = os.path.splitext(name)[1]
#             if ext in allowed_extensions:
#                 sources.append(os.path.join(root, name))
#     return sources

#  include_dirs=["Sources"],
#  sources=get_c_sources("Source"),

# print(get_c_sources("Sources"))

# extensions = [
# 	Extension(name="optimum_reparam_N_curvatures",
#            sources=get_c_sources("Source"),
# 	    include_dirs=["Sources"],
# 	),
# ]

extensions = [
	Extension(name="optimum_reparam_N_curvatures",
           sources=["Sources/optimum_reparam_N_curvatures.pyx", "Sources/DynamicProgrammingQ2_C.c", "Sources/dp_nbhd_C.c", "Sources/dp_grid_C.c"],
	    include_dirs=[numpy.get_include()],
	),
]

setup(
    name="FrenetFDA",
    ext_modules=cythonize(extensions),
    version="1.0",
    packages=["FrenetFDA", "FrenetFDA.processing_Euclidean_curve", 'FrenetFDA.processing_Euclidean_curve.unified_estimate_Frenet_state_space',
              "FrenetFDA.processing_Frenet_path", "FrenetFDA.shape_analysis", "FrenetFDA.utils", "FrenetFDA.utils.Lie_group"],
    # package_data = { 'FrenetFDA': ['Sources/optimum_reparam_N_curvatures.pyx', "Sources/dp_grid_C.h", "Sources/dp_nbhd_C.h", "Sources/cDPQ_C.pxd"]},
    # include_package_data = True,
    setup_requires=["cython", "numpy"],
    install_requires=["cython", "numpy"],
)


