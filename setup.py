from setuptools import setup, Extension
import numpy
from Cython.Build import cythonize

extensions = [
	Extension(name="optimum_reparamN2_C",
	    sources=["Sources/optimum_reparamN2_C.pyx", "Sources/DynamicProgrammingQ2_C.c",
        "Sources/dp_grid_C.c", "Sources/dp_nbhd_C.c"],
	    include_dirs=[numpy.get_include()],
	    language="c"
	),
]

setup(
    name="FrenetFDA",
    ext_modules=cythonize(extensions),
    version="1.0",
    packages=["FrenetFDA", "FrenetFDA.processing_Euclidean_curve", 'FrenetFDA.processing_Euclidean_curve.unified_estimate_Frenet_state_space',
              "FrenetFDA.processing_Frenet_path", "FrenetFDA.shape_analysis", "FrenetFDA.utils", "FrenetFDA.utils.Lie_group"],
)