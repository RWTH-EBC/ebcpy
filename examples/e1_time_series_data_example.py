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


def main(with_plot=True):
    """
    Arguments of this example:

    :param bool with_plot:
        Show the plot at the end of the script. Default is True.
    """
    # First get the path with relevant input files:
    basepath = pathlib.Path(__file__).parents[1].joinpath("tutorial", "data")
    # Note: We often use pathlib. If you're not familiar and want to learn more,
    # just search for any of the many tutorials available online.

    # ######################### Instantiation of TimeSeriesData ##########################
    # First we open a simulation result file (.mat)
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
    # Index changing:
    print(tsd_csv.index)
    tsd_csv.to_datetime_index(unit_of_index="s")
    print(tsd_csv.index)
    tsd_csv.to_float_index(offset=0)
    print(tsd_csv.index)
    # Some filter options
    tsd_csv.low_pass_filter(crit_freq=0.1, filter_order=2,
                            variable="outputs.TRoom", new_tag="lowPass2")
    print(tsd_csv)
    tsd_csv.moving_average(window=50, variable="outputs.TRoom",
                           tag="raw", new_tag="MovingAverage")
    print(tsd_csv)
    for tag in tsd_csv.get_tags(variable="outputs.TRoom")[::-1]:
        plt.plot(tsd_csv.loc[:, ("outputs.TRoom", tag)], label=tag)
    plt.legend()

    # How-to re-sample your data:
    # Call the function. Desired frequency is a string (s: seconds), 60: 60 seconds.
    # Play around with this value to see what happens.
    # First convert to DateTimeIndex (required for this function)
    tsd_csv.to_datetime_index(unit_of_index="s")
    # Create a copy to later reference the change.
    tsd_csv_ref = tsd_csv.copy()
    tsd_csv.clean_and_space_equally(desired_freq="60s")
    plt.figure()
    plt.plot(tsd_csv_ref.loc[:, ("outputs.TRoom", "raw")], label="Reference", color="blue")
    plt.plot(tsd_csv.loc[:, ("outputs.TRoom", "raw")], label="Resampled", color="red")
    plt.legend()
    if with_plot:
        plt.show()


if __name__ == '__main__':
    from ebcpy.utils import reproduction
    main()
    reproduction.save_reproduction_archive(title="log-testing", log_message='insert custom message here')


