# Setup file for packaging pyopmat

import os
import sys
import platform

from setuptools import setup, find_packages

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding = 'utf-8') as f:
  long_description = f.read()

setup (
    # Name of the project
    name = 'pyoptmat',
    # Version
    version = '1.2.0',
    # One line-description
    description = "Statistical inference for material models",
    # README
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    # Project webpage
    url='https://github.com/Argonne-National-Laboratory/pyoptmat',
    # Author
    author='Argonne National Laboratory',
    # email
    author_email = 'messner@anl.gov',
    # Various things for pypi
    classifiers=[
      'Intended Audience :: Science/Research',
      'License :: OSI Approved :: MIT License',
      'Programming Language :: Python :: 3',
      'Operating System :: OS Independent'
      ],
    # Which version of python is needed
    python_requires='>=3.6',
    # Keywords to help find
    keywords='materials inference modeling',
    # It is pure python
    zip_safe=True,
    # Get the python files
    packages=find_packages(),
    # Python dependencies
    install_requires=[
      'torch',
      'numpy',
      'scipy',
      'pyro-ppl',
      'tqdm',
      'matplotlib',
      'netCDF4',
      'xarray',
      'xarray[io]'
      ]
)
