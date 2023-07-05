from setuptools import setup, find_packages


setup(name='mo-deepdrive',
      version='0.0.1',
      # And any other dependencies we need
      install_requires=['gym', 'numpy', 'scipy', 'arcade', 'loguru',
                        'python-box', 'numba', 'matplotlib',
                        'retry', 'dataclasses'],
      packages=find_packages()
      )
