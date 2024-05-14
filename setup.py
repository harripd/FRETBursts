from setuptools import setup
from setuptools.extension import Extension
import numpy as np
import versioneer

## Metadata
project_name = 'fretbursts'

from Cython.Distutils import build_ext
ext_modules = [Extension("burstsearch_c",
                         [project_name + "/phtools/burstsearch_c.pyx"]),
               Extension("phrates_c",
                         [project_name + "/phtools/phrates_cy.pyx"],
                          include_dirs = ["."],)]

## Configure setup.py commands
cmdclass = versioneer.get_cmdclass()
cmdclass.update(build_ext=build_ext)


setup(name = project_name,
      version = versioneer.get_version(),
      cmdclass = cmdclass,
      include_dirs = [np.get_include()],
      ext_modules = ext_modules,
      include_package_data = True,
      packages = ['fretbursts', 'fretbursts.utils', 'fretbursts.fit',
                  'fretbursts.phtools', 'fretbursts.dataload',
                  'fretbursts.tests'],
      )
