"""
Goals of this part of the examples:
1. Learn how to use `load_time_series_data` and the `.tsd` accessor
2. Understand why we use this approach
3. Get to know the different processing functions
4. See how this compares to the legacy `TimeSeriesData` class
"""
# Start by importing all relevant packages
import pathlib
import numpy as np
import matplotlib.pyplot as plt
# Imports from ebcpy
from ebcpy import load_time_series_data

# For backwards compatibility example
from ebcpy import TimeSeriesData  # Will show a DeprecationWarning


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

    # ######################### Loading Time Series Data ##########################
    # First we open a simulation result file (.mat)
    df_mat = load_time_series_data(basepath.joinpath('simulatedData.mat'))
    print(df_mat)
    # Now a .csv. .xlsx works as well (with sheet_name parameter).
    df_csv = load_time_series_data(basepath.joinpath('excelData.csv'))
    print(df_csv)
    # Or construct like any pandas DataFrame
    df_random = load_time_series_data({"A": np.random.rand(100), "B": np.random.rand(100)})
    print(df_random)

    # ######################### Why do we use this approach? ##########################
    # Unlike the old TimeSeriesData which inherited from DataFrame,
    # our new approach uses standard pandas DataFrames with a custom accessor.
    # This makes it fully compatible with pandas ecosystem and tools like PyCharm's DataFrame viewer.
    # Moreover, the old MultiColumn approach using variable names and tags was useful when processing
    # variables with multiple stages, but made data handling much harder for everyone else.
    # Obviously, you can still create a multicolumn pd.DataFrame and use the old tag system,
    # it is just not the default anymore.
    print("The loaded object is a standard", type(df_csv).__name__)
    print("Time series functionality is available through the .tsd accessor")

    # ######################### Processing Time Series Data ##########################
    # Index changing:
    print(df_csv.index)
    df_csv.tsd.to_datetime_index(unit_of_index="s")
    print(df_csv.index)
    df_csv.tsd.to_float_index(offset=0)
    print(df_csv.index)

    # Some filter options
    # Apply filters and create new columns with results
    df_csv["outputs.TRoom_lowPass2"] = df_csv.tsd.low_pass_filter(
        crit_freq=0.1, filter_order=2, variable="outputs.TRoom")
    print(df_csv)

    # Moving average
    df_csv["outputs.TRoom_MovingAverage"] = df_csv.tsd.moving_average(
        window=50, variable="outputs.TRoom")
    print(df_csv)

    # Plot the different processed signals
    plt.figure()
    plt.plot(df_csv.index, df_csv["outputs.TRoom"], label="Raw")
    plt.plot(df_csv.index, df_csv["outputs.TRoom_lowPass2"], label="Low-pass (order 2)")
    plt.plot(df_csv.index, df_csv["outputs.TRoom_MovingAverage"], label="Moving Average")
    plt.legend()

    # How-to re-sample your data:
    # Call the function. Desired frequency is a string (s: seconds), 60: 60 seconds.
    # Play around with this value to see what happens.
    # First convert to DateTimeIndex (required for this function)
    df_csv.tsd.to_datetime_index(unit_of_index="s")
    # Create a copy to later reference the change.
    df_csv_ref = df_csv.copy()
    df_csv.tsd.clean_and_space_equally(desired_freq="60s")
    plt.figure()
    plt.plot(df_csv_ref.index, df_csv_ref["outputs.TRoom"], label="Reference", color="blue")
    plt.plot(df_csv.index, df_csv["outputs.TRoom"], label="Resampled", color="red")
    plt.legend()

    # ######################### Legacy TimeSeriesData Example ##########################
    # For reference, here's how the same operations would be done with the legacy class
    # Note: This will display a DeprecationWarning
    print("\n--- Legacy TimeSeriesData Example (Deprecated) ---")
    tsd_legacy = TimeSeriesData(basepath.joinpath('excelData.csv'), use_multicolumn=True)
    tsd_legacy.to_datetime_index(unit_of_index="s")
    tsd_legacy.low_pass_filter(crit_freq=0.1, filter_order=2,
                               variable="outputs.TRoom", new_tag="lowPass2")
    tsd_legacy.moving_average(window=50, variable="outputs.TRoom",
                              tag="raw", new_tag="MovingAverage")
    print("Legacy TimeSeriesData object with tags:", tsd_legacy.get_tags(variable="outputs.TRoom"))
    print(tsd_legacy)

    if with_plot:
        plt.show()


if __name__ == '__main__':
    from ebcpy.utils import reproduction
    main()
    reproduction.save_reproduction_archive(title="log-testing")
