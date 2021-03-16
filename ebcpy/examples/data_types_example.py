"""
Example file for the data_types module. The usage of classes inside
the data_types module should be clear when looking at the examples.
If not, please raise an issue.
"""
import os
from ebcpy.data_types import TimeSeriesData


def data_types_example():
    """
    Example to show setup and usage of data_types class TimeSeriesData.
    """
    mat = os.path.join(os.path.dirname(__file__), "data", "measTargetData.mat")
    tsd = TimeSeriesData(mat, tag='sim')
    print(tsd)
    return tsd


if __name__ == "__main__":
    data_types_example()
