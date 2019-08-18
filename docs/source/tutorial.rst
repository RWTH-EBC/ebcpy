How to use ebcpy
=====================

As this framework supports multiple stages in the process of calibration,
this small tutorial also separates the different tasks into separate examples.
However, as the interfaces between these steps are cohesive throughout the whole framework,
you can basically create a script running everything in one go.

Data types
----------

As the interfaces between the different packages are quite important, we will start with some
examples on how to setup the classes.

.. automodule:: ebcpy.examples.data_types_example
    :members:

Pre-processing
----------------
This module can basically be used at different places in this framework.
Typical use-cases are mentioned in the classes the function are used for.
For example the function :meth:`~ebcpy.preprocessing.conversion.convert_hdf_to_mat`
is used in the class :meth:`~ebcpy.data_types.MeasInputData` as mat-files are needed for
simulation in Modelica.

All functions in the pre-processing module should have a doctest. We refer to the example
in this doctest for a better understanding of the functions. If you don't understand the behaviour of
a function or the meaning, please raise an issue.
