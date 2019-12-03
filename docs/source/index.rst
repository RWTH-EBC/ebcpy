.. ebcpy documentation master file, created by
   sphinx-quickstart on Thu Jul 11 08:20:35 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

About ebcpy
-------------------

**ebcpy** is the official python package of the Institute for Energy and Building Climate
(EBC) from the RWTH Aachen. While the main use-case is to provide a common set of
interfaces for different challenges concerning building-simulations, one of the
use-cases below may lead to a direct use of this package.

- Process pandas.DataFrames or numpy arrays with functions for many different, typical problems
in the sector of Building-Simulations
- Provide different API`s to control modelica (or dymola) simulations
- Use a collection of functions to process or alter modelica-specific files such as .mat-files
or dsfinal.txt / dsin.txt files


Installation
-------------------

For installation use pip. Run ``pip install -e "Path/to/this/repository"``

If environment variables are not set properly, try more explicit command in Windows shell:

``C:\Path\to\pythonDirectory\python.exe -c "import pip" & C:\Path\to\pythonDirectory\python.exe -m pip install -e C:\Path\to\this\repository``

Be aware of forward slashes (for python) and backslashes (for Windows). You might need to encompass paths in inverted commas (") in order to handle spaces.


.. toctree::
   :maxdepth: 2

   tutorial
   data_types
   preprocessor
   optimizer
   simulationapi


Version History
-------------------

v0.1.0: Implemented necessary features
v0.1.1: Fixed bugs necessary to work with AixCaliBuHa and EnSTATS and refactor functions based on feedback
v0.1.2:
- Move CalibrationClass to AixCaliBuHa
- Add scipy.optimize.differential_evolution as an optimizer


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
