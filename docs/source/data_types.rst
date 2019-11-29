Data types
=====================

This module provides classes used in every package or module of this framework.
The correct understanding of the usage of this classes is a key to use the framework optimally.

.. _time-series-data:
Time Series Data
------------------

For different purposes in this framework, time-series data is needed.

In any calibration, three types of time-series data is necessary:

- **MeasTargetData**: The trajectories measured in an experiment. The calibration tries to fit some trajectories onto these measured ones.
- **SimTargetData**: These simulated trajectories compare directly to MeasTargetData
- **MeasInputData**: Measured input data are trajectories used to control the model during the simulation and establish the same environment as during the experiment. Examples are ambient temperatures or compressor on/off signal.

As all three above mentioned type of data are trajectories (time-series-data), the classes all extend from **TimeSeriesData**. A pandas.DataFrame object is used to handle the data.


.. autoclass:: ebcpy.data_types.TimeSeriesData
   :members:

.. autoclass:: ebcpy.data_types.MeasTargetData
   :members:

.. autoclass:: ebcpy.data_types.SimTargetData
   :members:

.. autoclass:: ebcpy.data_types.MeasInputData
   :members:

.. _tuner-parameter:
Tuner Parameters
------------------

As the goal of a calibration is to tune parameters in order to match some simulated data to measured data,
**tuner parameters** are necessary:

.. autoclass:: ebcpy.data_types.TunerParas
   :members:

.. _goals:
Goals
------------------

To build the objective for minimizing the differences betweeen simulation and measurement, one has to specify **Goals**:


.. autoclass:: ebcpy.data_types.Goals
   :members:
