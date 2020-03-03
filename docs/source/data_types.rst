Data types
=====================

This module provides classes used in every package or module of this framework.
The correct understanding of the usage of this classes is a key.

.. _time-series-data:
Time Series Data
------------------

For different purposes in this framework, time-series data is needed.
Simulation output trajectories which have to be matched to real test-bench data.
Both data is time-depended. The module `pandas.DataFrame` provides useful and efficient functions for time-series-data.
However, the dataframe is extended for an easier loading of result-files and for easy use of multi-indexing.


.. autoclass:: ebcpy.data_types.TimeSeriesData
   :members:


.. _tuner-parameter:
Tuner Parameters
------------------

As the goal of a optimization is to tune parameters in order to match some simulated data to measured data or minimize some costs,
therefore a class for such **tuner parameters** is necessary:

.. autoclass:: ebcpy.data_types.TunerParas
   :members:

.. _goals:
Goals
------------------

To build the objective for minimizing the differences betweeen simulation and measurement, one has to specify **Goals**:


.. autoclass:: ebcpy.data_types.Goals
   :members:
