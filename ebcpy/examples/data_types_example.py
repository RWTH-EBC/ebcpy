"""
Example file for the data_types module. The usage of classes inside
the data_types module should be clear when looking at the examples.
If not, please raise an issue.
"""

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
