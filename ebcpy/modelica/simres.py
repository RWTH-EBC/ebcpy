"""
Module based on the simres module of modelicares. As no new content is going to be
merged upstream, this "fork" of the to_pandas() function is used.
"""
import pandas as pd
import numpy as np
from modelicares.simres import SimRes
from modelicares import util


def to_pandas(sim, names=None, aliases={}, with_unit=True):
    """Return a `pandas DataFrame`_ with values from selected variables.

    The index is time.  The column headings indicate the variable names and
    units.

    The data frame has methods for further manipulation and exporting (e.g.,
    :meth:`~pandas.DataFrame.to_clipboard`,
    :meth:`~pandas.DataFrame.to_csv`, :meth:`~pandas.DataFrame.to_excel`,
    :meth:`~pandas.DataFrame.to_hdf`, and
    :meth:`~pandas.DataFrame.to_html`).

    **Arguments:**

    - *names*: String or list of strings of the variable names

         If *names* is *None* (default), then all variables are included.

    - *aliases*: Dictionary of aliases for the variable names

         The keys are the "official" variable names from the Modelica_ model
         and the values are the names as they should be included in the
         column headings.  Any variables not in this list will not be
         aliased.  Any unmatched aliases will not be used.

    - *with_unit*: Boolean to determine format of keys

        If set to True, the unit will be added to the key. As not all modelica-
        result files export the unit information, using with_unit=True can lead
        to errors.

    **Examples:**

    >>> sim = SimRes('examples/ChuaCircuit.mat')
    >>> voltages = sim.names('^[^.]*.v$', re=True)
    >>> to_pandas(sim, voltages) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
                C1.v / V  C2.v / V   G.v / V   L.v / V  Nr.v / V  Ro.v / V
    Time / s
    0.000000    4.000000  0.000000 -4.000000  0.000000  4.000000  0.000000
    5.000000    3.882738  0.109426 -3.773312  0.109235  3.882738  0.000191
    ...
    [514 rows x 6 columns]

    We can relabel columns using the *aliases* argument:

    >>> sim = SimRes('examples/ThreeTanks.mat')
    >>> aliases = {'tank1.level': "Tank 1 level",
    ...            'tank2.level': "Tank 2 level",
    ...            'tank3.level': "Tank 3 level"}
    >>> to_pandas(sim, list(aliases), aliases) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
               Tank 1 level / m  Tank 2 level / m  Tank 3 level / m
    Time / s
    0.000000           8.000000          3.000000          3.000000
    0.400000           7.974962          2.990460          3.034578
    0.800000           7.950003          2.981036          3.068961
    1.200000           7.925121          2.971729          3.103150
    1.600000           7.900317          2.962539          3.137144
    ...
    [502 rows x 3 columns]

    >>> sim = SimRes('examples/ChuaCircuit.mat')
    >>> voltages = sim.names('^[^.]*.v$', re=True)
    >>> to_pandas(sim, voltages, with_unit=False) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
              C1.v      C2.v      G.v       L.v       Nr.v      Ro.v
    Time
    0         4.000000  0.000000 -4.000000  0.000000  4.000000  0.000000
    5         3.882738  0.109426 -3.773312  0.109235  3.882738  0.000191
    ...
    [514 rows x 6 columns]

    """
    # Note: The first doctest above requires pandas >= 0.14.0.  Otherwise,
    # more decimal places are shown in the Time column.

    # Create the list of variable names.
    if names:
        names = set(util.flatten_list(names))
        names.add('Time')
    else:
        names = sim.names()

    # Create a dictionary of names and values.
    times = sim['Time'].values()
    data = {}
    for name in names:

        # Get the values.
        if np.array_equal(sim[name].times(), times):
            values = sim[name].values()  # Save computation.
        # Check if all values are constant to save resampling time
        elif np.count_nonzero(sim[name].values() - np.max(sim[name].values())) == 0:
            # Passing a scalar converts automatically to an array.
            values = np.max(sim[name].values())
        else:
            values = sim[name].values(t=times)  # Resample.

        unit = sim[name].unit

        # Apply an alias if available.
        try:
            name = aliases[name]
        except KeyError:
            pass

        if unit and with_unit:
            data.update({name + ' / ' + unit: values})
        else:
            data.update({name: values})

    # Create the pandas data frame.
    if with_unit:
        return pd.DataFrame(data).set_index('Time / s')
    else:
        return pd.DataFrame(data).set_index('Time')


def get_trajectories(sim):
    """Function to filter time-variant parameters.

    All variables which are trajectories are extracted from the simulation result.
    Either the length of the variable is greater than two, or the values are not
    equal. In both cases, the variable is considered to be a trajectory.

    **Examples:**
    >>> from modelicares import SimRes
    >>> sim = SimRes('examples/ChuaCircuit.mat')
    >>> trajectory_names = get_trajectories(sim)
    >>> len(trajectory_names)
    39
    """
    trajectory_names = []
    for name in sim.names():
        values = sim[name].values()
        # If the value array is greater then two, it is always a trajectory
        if len(values) > 2:
            trajectory_names.append(name)
        # Special Case: Only two time-steps are simulated.
        # In that case, if the last value does not equal the first value, it is also a trajectory
        elif len(values) == 2 and values[0] != values[-1]:
            trajectory_names.append(name)
    return trajectory_names
