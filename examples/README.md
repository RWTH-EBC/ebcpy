# Examples

This folder contains several example files which help with the understanding of ebcpy.

## Getting started

While these examples should run in any IDE, we advise using PyCharm.
Before being able to run these examples, be sure to:

1. Create a clean environment of python (We support 3.8 to 3.12). In Anaconda run: `conda create -n py313_ebcpy python=3.13`
2. Activate the environment in your terminal. In Anaconda run: `activate py312_ebcpy` 
3. Clone the repository by running `git clone https://github.com/RWTH-EBC/ebcpy`
4. Clone the BESMod in order to use the models: `git clone https://github.com/RWTH-EBC/BESMod` and install it as described in the BESMod Readme with AixLib
5. Install the library using `pip install ebcpy`.
   In order to execute everything, install the full version using `pip install ebcpy[full]`

## What can I learn in the examples?

### `e1_time_series_data_example.py`

1. Learn how to use time series data
2. Understand why we use `TimeSeriesAccessor`
3. Get to know the different processing functions

### `e2_fmu_example.py`

1. Learn how to use the `FMU_API`
2. Understand model variables
3. Learn how to change variables to store (`result_names`)
4. Learn how to change parameters of a simulation
5. Learn how to change inputs of a simulation
6. Learn how to run simulations in parallel

### `e3_0_simple_dymola_example.py`

1. Learn a common workflow for Dymola simulation studies using ebcpy
2. Understand how to run parameter studies across multiple model variants
3. Learn how to use model name modifiers
4. Learn how to post-process simulation results into usable formats

### `e3_dymola_example.py`

1. Learn how to use the `DymolaAPI`
2. Learn the different result options of the simulation
3. Learn how to convert inputs into the Dymola format
4. Learn advanced options for the `DymolaAPI`

### `e4_optimization_example.py`

1. Learn how to create a custom `Optimizer` class
2. Learn the different optimizer frameworks
3. See the difference in optimization when using newton-based methods and evolutionary algorithms.
   The difference is, that newton based methods (like L-BFGS-B) are vastly faster in both convex and
   concave problems, but they are not guaranteed to find the global minimum and can get stock in local optima. 
   Evolutionary algorithms (like the genetic algorithm) are substantially slower, 
   but they can overcome local optima, as shown in the concave examples.

### `e5_modifier_example.py`

1. Learn how to use the `DymolaAPI`
2. Learn how to dynamically modify structural parameters in the model
3. Learn how to redeclare models dynamically in the main model
