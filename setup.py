"""Setup.py script for the ebcpy-framework"""

import setuptools

INSTALL_REQUIRES = ['numpy',
                    'scipy',
                    'pandas',
                    'matplotlib',
                    'h5py',
                    'cmake',
                    'modelicares',
                    'sklearn',
                    'tables',
                    'psutil',
                    'pyyaml'
                    'fmpy'
                    ]
SETUP_REQUIRES = INSTALL_REQUIRES.copy()  # Add all open-source packages to setup-requires
#INSTALL_REQUIRES.append('dlib')  # Disabled until this is open-source

setuptools.setup(name='ebcpy',
                 version='0.1.6',
                 description='EBC Python Library used as a collection of useful '
                             'functions for different python modules of the '
                             'E.On Insttitute for Energy Efficien Buildings and Indoor '
                             'Climate',
                 url='not set yet',
                 author='RWTH Aachen University, E.ON Energy Research Center, Institute\
                 of Energy Efficient Buildings and Indoor Climate',
                 # Specify the Python versions you support here. In particular, ensure
                 # that you indicate whether you support Python 2, Python 3 or both.
                 classifiers=['Programming Language :: Python :: 3.6',
                              'Programming Language :: Python :: 3.7', ],
                 packages=setuptools.find_packages(exclude=['img']),
                 setup_requires=SETUP_REQUIRES,
                 install_requires=INSTALL_REQUIRES,
                 )
