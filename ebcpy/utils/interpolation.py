import pandas as pd

def interp_df(t: int, df: pd.DataFrame,
              interpolate: bool = False):
    """
    The function returns the values of the dataframe (row) at a given index.
    If the index is not present in the dataframe, either the next lower index
    is chosen or values are interpolated. If the last or first index value is exceeded the
    value is hold. In both cases a warning is printed.
    """
    # todo: consider check if step of input time step matches communication step size
    #  (or is given at a higher but aligned frequency).
    #  This might be the case very often and potentially inefficient df interpolation can be omitted in these cases.

    # initialize dict that represents row in dataframe with interpolated or hold values
    row = {}

    # catch values that are out of bound
    if t < df.index[0]:
        row.update(df.iloc[0].to_dict())
        warnings.warn(
            'Time {} s is below the first entry of the dataframe {} s, which is hold. Please check input data!'.format(
                t, df.index[0]))
    elif t >= df.index[-1]:
        row.update(df.iloc[-1].to_dict())
        # a time mathing the last index value causes problems with interpolation but should not raise a warning
        if t > df.index[-1]:
            warnings.warn(
                'Time {} s is above the last entry of the dataframe {} s, which is hold. Please check input data!'.format(
                    t, df.index[-1]))
    # either hold value of last index or interpolate
    else:
        # look for next lower index
        idx_l = df.index.get_indexer([t], method='pad')[0]  # get_loc() depreciated

        # return values at lower index
        if not interpolate:
            row = df.iloc[idx_l].to_dict()

        # return interpolated values
        else:
            idx_r = idx_l + 1
            for column in df.columns:
                row.update({column: np.interp(t, [df.index[idx_l], df.index[idx_r]],
                                              df[column].iloc[idx_l:idx_r + 1])})
    return row
