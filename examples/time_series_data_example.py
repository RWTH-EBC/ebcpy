"""
Goals of this part of the examples:
1. Learn how to use TimeSeriesData
2. Understand why we use TimeSeriesData
3. Get to know the different processing functions
"""
# Start by importing all relevant packages
import pathlib
import matplotlib.pyplot as plt
# Imports from ebcpy
from ebcpy import TimeSeriesData


def main():
    """
    This example has no arguments
    """

    # ######################### Instantiation of TimeSeriesData ##########################

    # ######################### Why do we use TimeSeriesData? ##########################
    print("TimeSeriesData inherits from", TimeSeriesData.__base__)

    # ######################### Processing TimeSeriesData ##########################


if __name__ == '__main__':
    main()
