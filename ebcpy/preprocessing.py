"""Module with static functions used to preprocess or alter
data, maily in the format of datafames or np.arrays."""
from datetime import datetime
from scipy import signal
from sklearn import model_selection
import numpy as np
import pandas as pd


def build_average_on_duplicate_rows(df):
    """
    If the dataframe has duplicate-indexes, the average
    value of all those indexes is calculated and given to
    the first occurrence of this duplicate index. Therefore,
    any dataFrame should be already sorted before calling this
    function.

    Example:

    >>> df = pd.DataFrame({"idx": np.ones(5), "val": np.arange(5)}).set_index("idx")
    >>> df = convert_index_to_datetime_index(df, origin=datetime(2007, 1, 1))
    >>> print(df)
                         val
    idx
    2007-01-01 00:00:01    0
    2007-01-01 00:00:01    1
    2007-01-01 00:00:01    2
    2007-01-01 00:00:01    3
    2007-01-01 00:00:01    4
    >>> print(build_average_on_duplicate_rows(df))
                         val
    idx
    2007-01-01 00:00:01  2.0

    :param pd.DataFame df:
        DataFrame with the data to process
    :return: pd.DataFame
    """
    # Find entries that are exactly the same timestamp
    double_ind = df.index[df.index.duplicated()].unique()
    # Calculate the mean value
    mean_values = []
    for item in double_ind:
        mean_values.append(df.loc[item].values.mean(axis=0))
    # Delete duplicate indices
    df_dropped = df[~df.index.duplicated(keep='first')].copy()

    # Set mean values in rows that were duplicates before
    for idx, values in zip(double_ind, mean_values):
        df_dropped.loc[idx] = values

    return df_dropped


def convert_index_to_datetime_index(df, unit_of_index="s", origin=datetime.now()):
    """
    Converts the index of the given DataFrame to a
    pandas.core.indexes.datetimes.DatetimeIndex.

    :param pd.DataFrame df:
        dataframe with index not being a DateTime.
        Only numeric indexes are supported. Every integer
        is interpreted with the given unit, standard form
        is in seocnds.
    :param str unit_of_index: default 's'
        The unit of the given index. Used to convert to
        total_seconds later on.
    :param datetime.datetime origin:
        The reference datetime object for the first index.
        Default is the current system time.
    :return: df
        DataFrame with correct index for usage in this
        framework.

    Examples:
    >>> import pandas as pd
    >>> df = pd.DataFrame(np.ones([3, 4]), columns=list('ABCD'))
    >>> print(df)
         A    B    C    D
    0  1.0  1.0  1.0  1.0
    1  1.0  1.0  1.0  1.0
    2  1.0  1.0  1.0  1.0
    >>> print(convert_index_to_datetime_index(df, origin=datetime(2007, 1, 1)))
                           A    B    C    D
    2007-01-01 00:00:00  1.0  1.0  1.0  1.0
    2007-01-01 00:00:01  1.0  1.0  1.0  1.0
    2007-01-01 00:00:02  1.0  1.0  1.0  1.0

    """
    # Check for unit of given index. Maybe one uses hour-based data.
    _unit_conversion_to_seconds = {"ms": 1e-3,
                                   "s": 1,
                                   "min": 1/60,
                                   "h": 1/3600,
                                   "d": 1/86400}
    if unit_of_index not in _unit_conversion_to_seconds:
        raise ValueError("Given unit_of_index is not supported.")
    _unit_factor_to_seconds = _unit_conversion_to_seconds.get(unit_of_index)

    #Convert
    old_index = df.index.copy()
    # Check if already converted:
    if isinstance(old_index, pd.DatetimeIndex):
        return df
    # Convert strings to numeric values.
    old_index = pd.to_numeric(old_index)
    # Convert to seconds.
    old_index *= _unit_factor_to_seconds
    # Alter the index
    df.index = pd.to_datetime(old_index, unit="s", origin=origin)

    return df


def convert_datetime_index_to_float_index(df, offset=0):
    """
    Convert a datetime-based index to FloatIndex (in seconds).
    Seconds are used as a standard unit as simulation software
    outputs data in seconds (e.g. Modelica)

    :param pd.DataFrame df:
        DataFrame to be converted to FloatIndex
    :param float offset:
        Offset in seconds
    :return: pd.DataFrame df:
        DataFrame with correct index

    Examples:
    >>> import pandas as pd
    >>> df = pd.DataFrame(np.ones([3, 4]), columns=list('ABCD'))
    >>> print(convert_index_to_datetime_index(df, origin=datetime(2007, 1, 1)))
                           A    B    C    D
    2007-01-01 00:00:00  1.0  1.0  1.0  1.0
    2007-01-01 00:00:01  1.0  1.0  1.0  1.0
    2007-01-01 00:00:02  1.0  1.0  1.0  1.0
    >>> print(convert_datetime_index_to_float_index(df))
           A    B    C    D
    0.0  1.0  1.0  1.0  1.0
    1.0  1.0  1.0  1.0  1.0
    2.0  1.0  1.0  1.0  1.0
    """
    # Check correct input
    if not isinstance(df.index, pd.DatetimeIndex):
        raise IndexError("Given DataFrame has no DatetimeIndex, conversion not possible")

    new_index = pd.to_timedelta(df.index - df.index[0]).total_seconds()
    df.index = np.round(new_index, 4) + offset
    return df


def time_based_weighted_mean(df):
    """
    Creates the weighted mean according to time index that does not need to be equidistant.
    Further info:
    https://stackoverflow.com/questions/26343252/create-a-weighted-mean-for-a-irregular-timeseries-in-pandas

    :param pd.DataFrame df:
        A pandas DataFrame with DatetimeIndex.
    :return np.array:
        A numpy array containing weighted means of all columns

    Examples:
    >>> from datetime import datetime
    >>> import numpy as np
    >>> import pandas as pd
    >>> time_vec = [datetime(2007,1,1,0,0),
    >>>             datetime(2007,1,1,0,0),
    >>>             datetime(2007,1,1,0,5),
    >>>             datetime(2007,1,1,0,7),
    >>>             datetime(2007,1,1,0,10)]
    >>> df = pd.DataFrame({'A': [1,2,4,3,6], 'B': [11,12,14,13,16]}, index=time_vec)
    >>> print(time_based_weighted_mean(df=df))
    [  3.55  13.55]
    """

    if not isinstance(df.index, pd.DatetimeIndex):
        raise IndexError(f"df.index must be DatetimeIndex, but it is {type(df.index)}.")

    time_delta = [(x-y).total_seconds() for x, y in zip(df.index[1:], df.index[:-1])]
    weights = [x+y for x, y in zip([0] + time_delta, time_delta + [0])]
    # Create empty numpy array
    res = np.empty(len(df.columns))
    res[:] = np.nan
    for i, col_name in enumerate(df.columns):
        res[i] = np.average(df[col_name], weights=weights)
    return res


def clean_and_space_equally_time_series(df, desired_freq):
    """
    Function for cleaning of the given dataFrame and interpolating
    based on the the given desired frequency. Linear interpolation
    is used.

    :param pd.DataFrame df:
        Unclean DataFrame. Needs to have a pd.DateTimeIndex
    :param str desired_freq:
        Frequency to determine number of elements in processed dataframe.
        Options are for example:
        - s: second-based
        - 5s: Every 5 seconds
        - 6min: Every 6 minutes
        This also works for h, d, m, y, ms etc.
    :return: pd.DataFrame
        Cleaned and equally spaced data-frame

    Examples:
    **Note:** As this function works best with some random
    data, we will leave it up to you to see the structure of the
    dataframe in every step.

    >>> df = pd.DataFrame(np.random.randint(0,100,size=(100, 4)),
    >>>                   columns=list('ABCD')).set_index("A").sort_index()
    >>> df = convert_index_to_datetime_index(df, origin=datetime(2007, 1, 1))
    >>> clean_and_space_equally_time_series(df, "30s")

    """
    # Convert indexes to datetime_index:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame needs a DateTimeIndex for executing this function."
                        "Call convert_index_to_datetime_index() to convert any index to "
                        "a DateTimeIndex")
    #%% Check DataFrame for NANs
    # Create a pandas Series with number of invalid values for each column of df
    series_with_na = df.isnull().sum()
    for name in series_with_na.index:
        if series_with_na.loc[name] > 0:
            # Print only columns with invalid values
            print("{} has following number of invalid "
                  "values\n {}".format(name, str(series_with_na.loc[name])))
    # Drop all rows where at least one NA exists
    df.dropna(how='any', inplace=True)

    # Check if DataFrame still has non-numeric-values:
    if not all(df.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all())):
        raise ValueError("Given DataFrame contains non-numeric values.")

    # Merge duplicate rows using mean.
    df = build_average_on_duplicate_rows(df)

    #%% Re-sampling to new frequency with linear interpolation
    # Create new equally spaced DatetimeIndex. Last entry is always < df.index[-1]
    time_index = pd.date_range(start=df.index[0], end=df.index[-1], freq=desired_freq)
    # Create an empty data frame
    # If multi-columns is used, first get the old index and make it empty:
    multi_cols = df.columns
    if isinstance(multi_cols, pd.MultiIndex):
        empty_multi_cols = pd.MultiIndex.from_product([[] for _ in range(multi_cols.nlevels)],
                                                      names=multi_cols.names)
        df_time_temp = pd.DataFrame(index=time_index, columns=empty_multi_cols)
    else:
        df_time_temp = pd.DataFrame(index=time_index)

    # Insert temporary time_index into df. fill_value = 0 can only be used,
    # since all NaNs should be eliminated prior
    df = df.radd(df_time_temp, axis='index', fill_value=0)
    del df_time_temp

    # Interpolate linearly according to time index
    df.interpolate(method='time', axis=0, inplace=True)
    # Determine Timedelta between current first index entry
    # in df and the first index entry that would be created
    # when applying df.resample() without loffset
    delta_time = df.index[0] - df.resample(rule=desired_freq).first().first(desired_freq).index[0]
    # Resample to equally spaced index.
    # All fields should already have a value. Thus NaNs and maybe +/- infs
    # should have been filtered beforehand.
    df = df.resample(rule=desired_freq, loffset=delta_time).first()
    del delta_time

    return df


def low_pass_filter(data, crit_freq, filter_order):
    """
    Create a low pass filter with given order and frequency.

    :param numpy.ndarray data:
        For dataframe e.g. df['a_col_name'].values
    :param float crit_freq:
    :param int filter_order:
    :return: numpy.ndarray,

    Examples:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> rand_series = np.random.rand(100)
    >>> plt.plot(rand_series, label="reference")
    >>> plt.plot(low_pass_filter(rand_series, 0.2, 2), label="filtered")
    >>> plt.legend()
    >>> plt.show()

    """
    if len(data.shape) > 1: # Check if given data has multiple dimensions
        if data.shape[1] == 1:
            data = data[:, 0]  # Resize to 1D-Array
        else:
            raise ValueError("Given data has multiple dimensions. "
                             "Only one-dimensional arrays are supported in this function.")
    _filter_order = int(filter_order)
    numerator, denominator = signal.butter(N=_filter_order, Wn=crit_freq,
                                           btype='low', analog=False, output='ba')
    output = signal.filtfilt(numerator, denominator, data)
    return output


def moving_average(data, window):
    """
    Creates a pandas Series as moving average of the input series.

    :param pd.Series values:
        For dataframe e.g. df['a_col_name'].values
    :param int window:
        sample rate of input
    :return: numpy.array
        shape has (###,). First and last points of input Series are extrapolated as constant
        values (hold first and last point).

    Examples:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> series = np.sin(np.linspace(-30, 30, 1000))
    >>> plt.plot(series, label="reference")
    >>> plt.plot(moving_average(series, 10), label="window=10")
    >>> plt.plot(moving_average(series, 50), label="window=50")
    >>> plt.plot(moving_average(series, 100), label="window=100")
    >>> plt.legend()
    >>> plt.show()

    """
    if len(data.shape) > 1: # Check if given data has multiple dimensions
        if data.shape[1] == 1:
            data = data[:, 0]  # Resize to 1D-Array
        else:
            raise ValueError("Given data has multiple dimensions. "
                             "Only one-dimensional arrays are supported in this function.")
    window = int(window)
    weights = np.repeat(1.0, window) / window
    sma = np.convolve(data, weights, 'valid')
    # Create array with first entries and window/2 elements
    fill_start = np.full((int(np.floor(window/2)), 1), sma[0])
    # Same with last value of -data-
    fill_end = np.full((int(np.ceil(window/2)) - 1, 1), sma[-1])
    # Stack the arrays
    sma = np.concatenate((fill_start[:, 0], sma, fill_end[:, 0]), axis=0)
    return sma


def create_on_off_signal(df, col_names, threshold, col_names_new,
                         tags="raw", new_tag="converted_signal"):
    """
    Create on and off signals based on the given threshold for all column names.

    :param pd.DataFame df:
        DataFrame with the data to process
    :param list col_names:
        Column names of variables to convert to signals
    :param float,list threshold:
        Threshold for all column-names (single float) or
        a list with specific thresholds for specific columns.
    :param list col_names_new:
        New name for the signal-column
    :param str,list tags:
        If a 2-Level DataFrame for TimeSeriesData is used, one has to
        specify the tag of the variables. Default value is to use the "raw"
        tag set in the TimeSeriesClass. However one can specify a list
        (Different tag for each variable), or on can pass a string
        (same tags for all given variables)
    :param str new_tag:
        The tag the newly created variable will hold. This can be used to
        indicate where the signal was converted from.
    :return: pd.DataFrame
        Now with the created signals.

    Examples:

    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> df = pd.DataFrame({"P_el": np.sin(np.linspace(-20, 20, 10000))*100})
    >>> df = create_on_off_signal(df, col_names=["P_el"],
    >>>                           threshold=25, col_names_new=["Device On"])
    >>> plt.plot(df)
    >>> plt.show()
    """
    if len(col_names) != len(col_names_new):
        raise IndexError("Given lists differ in length. col_names: {}, "
                         "col_names_new: {}".format(len(col_names), len(col_names_new)))
    if isinstance(threshold, list):
        if len(col_names) != len(threshold):
            raise IndexError("Given lists differ in length. col_names: {}, "
                             "threshold: {}".format(len(col_names), len(threshold)))
    else:
        threshold = [threshold for _ in enumerate(col_names)]
    # Do on_off signal creation for all desired columns
    if isinstance(df.columns, pd.MultiIndex):
        # Convert given tags to a list
        if isinstance(tags, str):
            tags = [tags for _ in enumerate(col_names)]

        for i, _ in enumerate(col_names):
            # Create zero-array
            df.loc[:, (col_names_new[i], new_tag)] = 0.0
            # Change all values to 1.0 according to threshold
            df.loc[df[col_names[i], tags[i]] >= threshold[i], (col_names_new[i], new_tag)] = 1.0
    else:
        for i, _ in enumerate(col_names):
            # Create zero-array
            df.loc[:, col_names_new[i]] = 0.0
            # Change all values to 1.0 according to threshold
            df.loc[df[col_names[i]] >= threshold[i], col_names_new[i]] = 1.0
    return df


def number_lines_totally_na(df):
    """
    Returns the number of rows in the given dataframe
    that are filled with NaN-values.

    :param pd.DataFrame df:
        Given dataframe to process
    :return: int
        Number of NaN-Rows.

    Examples:

    >>> import numpy as np
    >>> import pandas as pd
    >>> dim = np.random.randint(100) + 10
    >>> nan_col = [np.NaN for i in range(dim)]
    >>> col = [i for i in range(dim)]
    >>> df_nan = pd.DataFrame({"col_1":nan_col, "col_2":nan_col})
    >>> df_normal = pd.DataFrame({"col_1":nan_col, "col_2":col})
    >>> print(number_lines_totally_na(df_nan)-dim)
    0
    >>> print(number_lines_totally_na(df_normal))
    0
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Input must be a pandas data frame')
    counter = 0
    for _, row in df.iterrows():
        # Check if the whole row is filled with NaNs.
        if all(row.isnull()):
            counter += 1
    return counter


def z_score(x, limit=3):
    """
    Calculate the z-score using the mea
    and standard deviation of the given data.

    :param np.array x:
        For dataframe e.g. df['a_col_name'].values
    :param float limit: default 3
        Lower limit for required z-score
    :return: np.array iqr:
        modified z score

    Examples:

    >>> import numpy as np
    >>> normal_dis = np.random.normal(0, 1, 1000)
    >>> res = z_score(normal_dis, limit=2)
    >>> values = normal_dis[res]

    """
    mean = np.mean(x)
    standard_deviation = np.std(x)
    z_score_value = (x-mean)/standard_deviation
    return np.where(np.abs(z_score_value) > limit)[0]


def modified_z_score(x, limit=3.5):
    """
    Calculate the modified z-score using the median
    and median average deviation of the given data.

    :param np.array x:
        For dataframe e.g. df['a_col_name'].values
    :param float limit: default 3.5
        Lower limit for required z-score
    :return: np.array iqr:
        modified z score

    Examples:

    >>> import numpy as np
    >>> normal_dis = np.random.normal(0, 1, 1000)
    >>> res = modified_z_score(normal_dis, limit=2)
    >>> values = normal_dis[res]

    """
    median = np.median(x)
    median_average_deviation = np.median(np.abs(x-median))
    z_score_mod = 0.6745*(x-median)/median_average_deviation
    return np.where(np.abs(z_score_mod) > limit)[0]


def interquartile_range(x):
    """
    Calculate interquartile range of given array.
    Returns the indices of values outside of the interquartile range.

    :param np.array x:
        For dataframe e.g. df['a_col_name'].values
    :return: np.array iqr:
        Array matching the interquartile-range

    Examples:

    >>> import numpy as np
    >>> normal_dis = np.random.normal(0, 1, 1000)
    >>> res = interquartile_range(normal_dis)
    >>> values = normal_dis[res]

    """
    quartile_1, quartile_3 = np.percentile(x, [25, 75])
    iqr = quartile_3 - quartile_1
    lower = quartile_1 - (iqr * 1.5)
    upper = quartile_3 + (iqr * 1.5)
    return np.where((x > upper) | (x < lower))[0]


def cross_validation(x, y, test_size=0.3):
    """
    Split data set randomly with test_size
    (if test_size = 0.30 --> 70 % are training data).
    You can use this function for segmentation tasks.
    Time-series-data may not be splitted with this function
    as the results are not coherent (time-wise).

    :param x:
        Indexables with same length / shape[0] as y.
        Allowed inputs are lists, numpy arrays, scipy-sparse
        matrices or pandas dataframes.
    :param list,np.ndarray,pd.DataFrame y:
        Indexables with same length / shape[0] as x.
        Allowed inputs are lists, numpy arrays, scipy-sparse
        matrices or pandas dataframes.
    :param float test_size:
        Value between 0 and 1 specifying what percentage of the data
        will be used for testing.
    :return: list
        Split data into 4 objects. The order is:
        x_train, x_test, y_train, y_test

    Examples:

    >>> import numpy as np
    >>> x = np.random.rand(100)
    >>> y = np.random.rand(100)
    >>> ret = cross_validation(x, y)
    >>> len(ret)
    4
    """
    return model_selection.train_test_split(x, y, test_size=test_size)