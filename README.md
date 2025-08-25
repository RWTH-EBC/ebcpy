![E.ON EBC RWTH Aachen University](https://raw.githubusercontent.com/RWTH-EBC/ebcpy/master/docs/EBC_Logo.png)

[![DOI](https://joss.theoj.org/papers/10.21105/joss.03861/status.svg)](https://doi.org/10.21105/joss.03861)
[![pylint](https://rwth-ebc.github.io/ebcpy/master/pylint/pylint.svg )](https://rwth-ebc.github.io/ebcpy/master/pylint/pylint.html)
[![documentation](https://rwth-ebc.github.io/ebcpy/master/docs/doc.svg)](https://rwth-ebc.github.io/ebcpy/master/docs/index.html)
[![coverage](https://rwth-ebc.github.io/ebcpy/master/coverage/badge.svg)](https://rwth-ebc.github.io/ebcpy/master/coverage)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![build](https://rwth-ebc.github.io/ebcpy/master/build/build.svg)](https://rwth-ebc.github.io/ebcpy/master/build/build.svg)


# ebcpy

This **PY**thon package provides generic functions and classes commonly
used for the analysis and optimization of **e**nergy systems, **b**uildings and indoor **c**limate (**EBC**).

Key features are:

* `SimulationAPI`'s
* Optimization wrapper
* Useful loading of time series data and time series data accessor for DataFrames
* Pre-/Postprocessing
* Modelica utilities

It was developed together with `AixCaliBuHA`, a framework for an automated calibration of dynamic building and HVAC models. During this development, we found several interfaces relevant to further research. We thus decoupled these interfaces into `ebcpy` and used the framework, for instance in the design optimization of heat pump systems ([link](https://www.sciencedirect.com/science/article/abs/pii/S0196890421010645?via%3Dihub)).

# Installation

To install, simply run
```
pip install ebcpy
```

In order to use all optional dependencies (e.g. `pymoo` optimization), install via:

```
pip install ebcpy[full]
```

If you encounter an error with the installation of `scikit-learn`, first install `scikit-learn` separatly and then install `ebcpy`:

```
pip install scikit-learn
pip install ebcpy
```

If this still does not work, we refer to the troubleshooting section of `scikit-learn`: https://scikit-learn.org/stable/install.html#troubleshooting. Also check [issue 23](https://github.com/RWTH-EBC/ebcpy/issues/23) for updates.

In order to help development, install it as an egg:

```
git clone https://github.com/RWTH-EBC/ebcpy
pip install -e ebcpy
```

# How to get started?

We recommend running our jupyter-notebook to be guided through a **helpful tutorial**.  
For this, run the following code:
```
# If jupyter is not already installed:
pip install jupyter
# Go into your ebcpy-folder (cd \path_to_\ebcpy) or change the path to tutorial.ipynb and run:
jupyter notebook tutorial\tutorial.ipynb
```

Or, clone this repo and look at the examples\README.md file.
Here you will find several examples to execute.

# How to cite ebcpy

Please use the following metadata to cite `ebcpy` in your research:

```
@article{Wuellhorst2022,
  doi = {10.21105/joss.03861},
  url = {https://doi.org/10.21105/joss.03861},
  year = {2022},
  publisher = {The Open Journal},
  volume = {7},
  number = {72},
  pages = {3861},
  author = {Fabian Wüllhorst and Thomas Storek and Philipp Mehrfeld and Dirk Müller},
  title = {AixCaliBuHA: Automated calibration of building and HVAC systems},
  journal = {Journal of Open Source Software}
}
```

# Time series data
Note that we use steamline time series data based on a `pd.DataFrame`
using a common function and the accessor `tsd`. 
The aim is to make tasks like loading different filetypes or common functions
more convenient, while conserving the powerful tools of the DataFrame.
Just a example intro here:

```python
>>> from ebcpy.data_types import load_time_series_data
>>> df = load_time_series_data(r"path_to_a_supported_file")

# From Datetime to float
df.tsd.to_float_index()
# From float to datetime
df.tsd.to_datetime_index()
# To clean your data and create a common frequency:
df.tsd.clean_and_space_equally(desired_freq="1s")
```

# Documentation
Visit our official [Documentation](https://rwth-ebc.github.io/ebcpy/master/docs/index.html).

# Problems or questions?
Please [raise an issue here](https://github.com/RWTH-EBC/ebcpy/issues/new).

For other inquires, please contact [ebc-tools@eonerc.rwth-aachen.de](mailto:ebc-tools@eonerc.rwth-aachen.de).
