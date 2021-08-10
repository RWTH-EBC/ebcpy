"""
Goals of this part of the examples:
 1. Learn how to use `TimeSeriesData`
2. Understand why we use `TimeSeriesData`
3. Get to know the different processing functions
"""
# Start by importing all relevant packages
import pathlib
import numpy as np
import matplotlib.pyplot as plt
# Imports from ebcpy
from ebcpy import TimeSeriesData


def main():
    """
    This example has no arguments
    """
    # First get the path with relevant input files:
    basepath = pathlib.Path(__file__).parents[1].joinpath("tutorial", "data")

    # ######################### Instantiation of TimeSeriesData ##########################
    # First we open an .hdf. Be carful, you have to pass a key!
    tsd_hdf = TimeSeriesData(basepath.joinpath('measuredData.hdf'), key='test')
    print(tsd_hdf)
    # Now a simulation result file (.mat)
    tsd_mat = TimeSeriesData(basepath.joinpath('simulatedData.mat'))
    print(tsd_mat)
    # Now a .csv. .xlsx works as well.
    tsd_csv = TimeSeriesData(basepath.joinpath('excelData.csv'))
    print(tsd_csv)
    # Or construct like any pandas DataFrame
    tsd_df = TimeSeriesData({"A": np.random.rand(100), "B": np.random.rand(100)})
    print(tsd_df)
    # ######################### Why do we use TimeSeriesData? ##########################
    # As you may have guessed, TimeSeriesData is just a plain old DataFrame.
    # It inherits the standard one and adds functionalities used typically on
    # energy related time series.
    print("TimeSeriesData inherits from", TimeSeriesData.__base__)

    # ######################### Processing TimeSeriesData ##########################
    # TODO Describe what is happening
    # Index changing:
    print(tsd_hdf.index)
    tsd_hdf.to_float_index(offset=0)
    print(tsd_hdf.index)
    tsd_hdf.to_datetime_index(unit_of_index="s")
    print(tsd_hdf.index)
    # Some filter options
    tsd_hdf.low_pass_filter(crit_freq=0.1, filter_order=2,
                            variable="measured_T", new_tag="lowPass2")
    print(tsd_hdf)
    tsd_hdf.moving_average(window=50, variable="measured_T",
                           tag="raw", new_tag="MovingAverage")
    print(tsd_hdf)
    for tag in tsd_hdf.get_tags(variable="measured_T")[::-1]:
        plt.plot(tsd_hdf.loc[:, ("measured_T", tag)], label=tag)
    plt.legend()

    # How-to re-sample your data:
    tsd_hdf_ref = tsd_hdf.copy()  # Create a savecopy to later reference the change.
    # Call the function. Desired frequency is a string (s: seconds), 60: 60 seconds.
    # Play around with this value to see what happens.
    tsd_hdf.clean_and_space_equally(desired_freq="60s")
    plt.figure()
    plt.plot(tsd_hdf_ref.loc[:, ("measured_T", "raw")], label="Reference", color="blue")
    plt.plot(tsd_hdf.loc[:, ("measured_T", "raw")], label="Resampled", color="red")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
