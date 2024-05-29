from setuptools import setup
from setuptools.extension import Extension
import numpy as np

## Metadata
project_name = 'fretbursts'

from Cython.Build import cythonize
ext_modules = [Extension("burstsearch_c",
                         [project_name + "/phtools/burstsearch_c.pyx"]),
               Extension("phrates_c",
                         [project_name + "/phtools/phrates_cy.pyx"],
                          include_dirs = ["."],)]

## Configure setup.py commands


setup(
      include_dirs = [np.get_include()],
      ext_modules = cythonize(ext_modules),
      include_package_data = True,
      packages = ['fretbursts', 'fretbursts.utils', 'fretbursts.fit',
                  'fretbursts.phtools', 'fretbursts.dataload'],
      )
