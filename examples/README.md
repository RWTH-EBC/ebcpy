# Examples

This folder contains several example files which help with the understanding of ebcpy.

## Getting started

While these examples should run in any IDE, we advise using PyCharm.
Before being able to run these examples, be sure to:

1. Create a clean environment of python 3.7 or 3.8. In Anaconda run: `conda create -n py38_ebcpy python=3.8`
2. Activate the environment in your terminal. In Anaconda run: `activate py38_ebcpy` 
3. Clone the repository by running `git clone https://github.com/RWTH-EBC/ebcpy`
4. Clone the AixLib in order to use the models: `git clone https://github.com/RWTH-EBC/AixLib`
   Also check if you're on development using `cd AixLib && git status && cd ..`
5. Install the library using `pip install -e ebcpy`.
   In order to execute everything, install the full version using `pip install -e ebcpy[full]`

## What can I learn in the examples?

### `e1_time_series_data_example.py`

1. Learn how to use `TimeSeriesData`
2. Understand why we use `TimeSeriesData`
3. Get to know the different processing functions

### `e2_fmu_example.py`

1. Learn how to use the `FMU_API`
2. Understand model variables
3. Learn how to change variables to store (`result_names`)
4. Learn how to change parameters of a simulation
5. Learn how to change inputs of a simulation
6. Learn how to run simulations in parallel

### `e3_dymola_example.py`

1. Learn how to use the `DymolaAPI`
2. Learn the different result options of the simulation
3. Learn how to convert inputs into the Dymola format

### `e4_optimization_example.py`

1. Learn how to create a custom `Optimizer` class
2. Learn the different optimizer frameworks
3. Learn the usage of `StatisticsAnalyzer`
4. Understand the motivation behing `AixCaliBuHA`
