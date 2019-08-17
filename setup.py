"""Setup.py script for the ebcpy-framework"""

import setuptools
import pip
from packaging import version

install_requires = ['numpy',
                    'scipy',
                    'pandas',
                    'matplotlib',
                    'h5py',
                    'SALib',
                    'pydot',
                    'cmake',
                    'modelicares'
                    ]
setup_requires = install_requires.copy()  # Add all open-source packages to setup-requires
#install_requires.append('dlib')

setuptools.setup(name='ebcpy',
                 version='0.1',
                 description='EBC Python Library used as a collection of useful '
                             'functions for different python modules of the '
                             'E.On Insttitute for Energy Efficien Buildings and Indoor '
                             'Climate',
                 url='not set yet',
                 author='RWTH Aachen University, E.ON Energy Research Center, Institute\
                 of Energy Efficient Buildings and Indoor Climate',
                 # Specify the Python versions you support here. In particular, ensure
                 # that you indicate whether you support Python 2, Python 3 or both.
                 classifiers=['Programming Language :: Python :: 3.5',
                              'Programming Language :: Python :: 3.6',
                              'Programming Language :: Python :: 3.7', ],
                 packages=setuptools.find_packages(exclude=['img']),
                 setup_requires=setup_requires,
                 install_requires=install_requires,
                 )
