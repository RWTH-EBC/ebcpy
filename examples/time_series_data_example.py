"""
Goals of this part of the examples:
1. Learn how to use TimeSeriesData
2. Understand why we use TimeSeriesData
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


if __name__ == '__main__':
    main()
