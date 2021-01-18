.. _preprocessing:
Preprocessing
===================

This general overview may help you find the function you need:

- Remove duplicate rows by averaging the values (build_average_on_duplicate_rows)
- Convert any integer or float index into a datetime index (convert_index_to_datetime_index)
- Resample a given time-series on a given frequency (clean_and_space_equally_time_series)
- Apply a low-pass-filter (low_pass_filter)
- Apply a moving average to flatten disturbances in your measured data (moving_average)
- Convert e.g. an electrical power signal into a binary control signal (on-off) based on a threshold (create_on_off_signal)
- Find the number of lines without any values in it (number_lines_totally_na)
- Split a data-set into training and test set according to cross-validation(cross_validation)

All functions in the pre-processing module should have a doctest. We refer to the example
in this doctest for a better understanding of the functions. If you don't understand
the behaviour of a function or the meaning, please raise an issue.


.. automodule:: ebcpy.preprocessing
   :members:

.. _conversion:
Conversion
------------------
.. automodule:: ebcpy.utils.conversion
   :members:
