"""Setup.py script for the ebcpy-framework"""

import setuptools

EXTRAS_REQUIRE = {
    'full': [
        'openpyxl>=3.0.5',
        'xlrd>=2.0.1',
        'pymoo>=0.4.2',
    ]
}

INSTALL_REQUIRES = [
    'numpy>=1.19.5',
    'matplotlib>=3.3.4',
    'scipy>=1.5.4',
    'pandas>=1.1.5',
    'tables>=3.6.1',
    'sklearn~=0.0',
    'fmpy>=0.2.27',
    'pydantic>=1.8.2',
    'h5py>=3.3.0'
]
SETUP_REQUIRES = INSTALL_REQUIRES.copy()  # Add all open-source packages to setup-requires

setuptools.setup(
    name='ebcpy',
    version='0.2.0',
    description='EBC Python Library used as a collection of useful '
                'functions for different python modules of the '
                'E.On Insttitute for Energy Efficien Buildings and Indoor '
                'Climate',
    url='not set yet',  # TODO: Set when on pypi
    author='RWTH Aachen University, E.ON Energy Research Center, Institute\
                 of Energy Efficient Buildings and Indoor Climate',
    # Specify the Python versions you support here. In particular, ensure
    # that you indicate whether you support Python 2, Python 3 or both.
    classifiers=['Programming Language :: Python :: 3.7',
                 'Programming Language :: Python :: 3.8'],
    packages=setuptools.find_packages(exclude=['img']),
    extras_require=EXTRAS_REQUIRE,
    setup_requires=SETUP_REQUIRES,
    install_requires=INSTALL_REQUIRES,
)
