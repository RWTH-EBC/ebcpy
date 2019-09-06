"""
Example file for the data_types module. The usage of classes inside
the data_types module should be clear when looking at the examples.
If not, please raise an issue.
"""

import os
from ebcpy import data_types


def setup_tuner_paras():
    """
    Example setup of tuner parameters.

    The parameter names are based on the model TestModel from
    the package AixCalTest. Open the model in Modelica to see other
    possible tuner parameters or have a look at the example on how
    to :meth:`find all tuner parameters<ebcpy.examples.dymola_api_example.example_dymola_api>`.

    The bounds object is optional, however highly recommend
    for calibration or optimization in general. As soon as you
    tune parameters with different units, such as Capacity and
    heat conductivity, the solver will fail to find good solutions.

    :return: Tuner parameter class
    :rtype: data_types.TunerParas

    Example:

    >>> tuner_paras = setup_tuner_paras()
    >>> print(tuner_paras)
                    initial_value     max      min    scale
    names
    C                     5000.00  6000.0  4000.00  2000.00
    m_flow_2                 0.02     0.1     0.01     0.09
    heatConv_a             200.00   300.0    10.00   290.00
    """
    tuner_paras = data_types.TunerParas(names=["C", "m_flow_2", "heatConv_a"],
                                        initial_values=[5000, 0.02, 200],
                                        bounds=[(4000, 6000), (0.01, 0.1), (10, 300)])

    return tuner_paras


def setup_goals():
    """
    Example setup of the Goals object.
    First, some simulated and measured target data is loaded from the
    example data.
    Then the goals object is instantiated. Please refer to the
    Goals documentation on the meaning of the parameters.


    :return: Goals object
    :rtype: data_types.Goals

    Example:

    >>> goals = setup_goals()
    >>> dif = goals.eval_difference(statistical_measure="RMSE")
    >>> print(round(dif, 3))
    1.055
    """

    # Load example simTargetData and measTargetData:
    _filepath = os.path.dirname(__file__)
    sim_target_data = data_types.SimTargetData(_filepath + "//simTargetData.mat")
    meas_target_data = data_types.MeasTargetData(_filepath + "//measTargetData.mat")

    # Setup the goals object
    goals = data_types.Goals(meas_target_data,
                             sim_target_data,
                             meas_columns=["heater.heatPorts[1].T", "heater1.heatPorts[1].T"],
                             sim_columns=["heater.heatPorts[1].T", "heater1.heatPorts[1].T"],
                             weightings=[0.7, 0.3])
    return goals


def setup_calibration_classes():
    """
    Example setup of a list calibration classes.
    The measured data of the setup_goals example can
    be segmentized into two classes. You can either use
    classes from the segmentizer package or manually define
    classes of interest to you. In this example the we have
    a manual segmentation, as the example is fairly small.

    :return: List of calibration classes
    :rtype: list
    """
    # Define the basic time-intervals and names for the calibration-classes:
    calibration_classes = [
        data_types.CalibrationClass(name="Heat up", start_time=0, stop_time=200),
        data_types.CalibrationClass(name="stationary", start_time=200, stop_time=400),
        data_types.CalibrationClass(name="cool down", start_time=400, stop_time=600),
    ]
    # Load the tuner parameters and goals
    tuner_paras = setup_tuner_paras()
    goals = setup_goals()
    # Set the tuner parameters and goals to all classes:
    for cal_class in calibration_classes:
        cal_class.set_tuner_paras(tuner_paras)
        cal_class.set_goals(goals)

    return calibration_classes
