.. ebcpy documentation master file, created by
   sphinx-quickstart on Thu Jul 11 08:20:35 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

About ebcpy
-------------------

**ebcpy** is the official python package of the Institute for Energy and Building Climate
(EBC) from the RWTH Aachen. The main use-case is to provide a common set of
interfaces for different challenges concerning building-simulations.  one of the
Find below possible challenges that may lead you to use this package.:

- Easily load and process Time-Series-Data into a pd.DataFrame and further pre-process it using various functions
- (Co-)Simulate models in the simulation-language Modelica either trough our fmu-api or dymola-api
- Optimize results of the simulation through a given objective-function using different open-source python frameworks and solvers for optimization.
- Process pandas.DataFrames or numpy arrays with functions for many different, typical problems in the sector of Building-Simulations
- Use a collection of functions to process or alter modelica-specific files such as .mat-files or dsfinal.txt / dsin.txt files

Check out our tutorial to fully understand the different classes and functions in this repo.
Always contribute if you see room for improvement by raising and issues

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
   simulationapi
   optimizer
   modelica


Version History
-------------------

v0.1.0: Implemented necessary features for use in AixCaliBuHa and EnSTATS
v0.1.1: Fixed bugs necessary to work with AixCaliBuHa and EnSTATS and refactor functions based on feedback
v0.1.2:
- Move CalibrationClass to AixCaliBuHa
- Add scipy.optimize.differential_evolution as an optimizer


Version History
-------------------

v0.1

- v0.1.0:
   - Implemented necessary features to run together with AixCaliBuHa and EnSTATS
- v0.1.1:
   - Fixed bugs necessary to work with AixCaliBuHa and EnSTATS and refactor functions based on feedback
- v0.1.2:
   - Move CalibrationClass to AixCaliBuHa
   - Add scipy.optimize.differential_evolution as an optimizer
- v0.1.3:
   - Move conversion.py to utils and make preprocessing a direct module
   - Introduce current_best_iterate as a parameter for optimization to ensure that the best solution to a problem is still saved even if an iteration step causes an error
   - Make interrupt of optimization through Keyboard-Interrupt possible
   - Adjust Goals functions to make slicing of multiple time-intervals possible.
- v0.1.4:
   - Create a tutorial with juypter-notebook
   - Introduce MultiIndex (from pandas) and make TimeSeriesData extend of the standard DataFrame
   - Adjust all classes, most notably Goals. This class will go into AixCaliBuHa, as it is only relevant for Calibrations.
- v0.1.5:
   - Remove dlib and PyQt5 from setup.py and delete TunerParas.show()
   - Refactor Optimizer so the framework parameter is only necessary to call optimize()
- v0.1.6:
   - Issue23: Change conversion functions to correctly handle multiheaders
   - Issue24: Add converters and correctly inherit from pd.DataFame in TimeSeriesData
   - Add functions for configuration using yaml
   - Add regex functions for extraction of modelica variables
   - Add option to directly return results as a dataframe using simulateMultiResultsModel.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
