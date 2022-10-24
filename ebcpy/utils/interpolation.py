""" utility function(s) for interpolation and table lookup """
import warnings
import pandas as pd
import numpy as np


def interp_df(t_act: float, df: pd.DataFrame, interpolate: bool = False) -> dict:
    """
    Returns the values of the dataframe (row) at a given index.
    If the index is not present in the dataframe, either the next lower index
    is chosen or values are interpolated linearly. If the last or first index value is exceeded the
    value is hold. In both cases a warning is printed.

    :param float t_act:
        Time index of interest
    :param pd.dataFrame df:
        "Table" (pd.Dataframe) in which to lookup
    :param bool interpolate:
        Whether to interpolate (True) or chose the last available index (False)
    :return:
        Dict: Dictionary of column name (key) and value (value) at the selected index
    """

    # initialize dict that represents row in dataframe with interpolated or hold values
    row = {}

    # catch values that are out of bound
    if t_act < df.index[0]:
        row.update(df.iloc[0].to_dict())
        warnings.warn(
            f"Time {t_act} s is below the first entry of the dataframe {df.index[0]} s, "
            f"which is hold. Please check input data!")
    elif t_act >= df.index[-1]:
        row.update(df.iloc[-1].to_dict())
        # a time matching the last index value causes
        # problems with interpolation but should not raise a warning
        if t_act > df.index[-1]:
            warnings.warn(
                f"Time {t_act} s is above the last entry of the dataframe {df.index[-1]} s, "
                f"which is hold. Please check input data!")
    # either hold value of last index or interpolate
    else:
        # look for next lower index
        idx_l = df.index.get_indexer([t_act], method='pad')[0]  # get_loc() depreciated

        # return values at lower index
        if not interpolate:
            row = df.iloc[idx_l].to_dict()

        # return interpolated values
        else:
            idx_r = idx_l + 1
            for column in df.columns:
                row.update({column: np.interp(t_act, [df.index[idx_l], df.index[idx_r]],
                                              df[column].iloc[idx_l:idx_r + 1])})
    return row
