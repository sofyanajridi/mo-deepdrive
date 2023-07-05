from setuptools import setup

setup(name='mo-deepdrive',
      version='0.0.1',
      # And any other dependencies we need
      install_requires=['gym', 'numpy', 'scipy', 'arcade', 'loguru',
                        'python-box', 'numba', 'matplotlib',
                        'retry', 'dataclasses'],
      py_modules = []
      )
