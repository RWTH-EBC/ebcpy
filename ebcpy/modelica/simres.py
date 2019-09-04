"""
Module based on the simres module of modelicares. As no new content is going to be
merged upstream, this "fork" of the to_pandas() function is used.
"""
import pandas as pd
import numpy as np
from modelicares import util


def to_pandas(sim, names=None, aliases=None, with_unit=True):
    """
    Return a `pandas.DataFrame` with values from selected variables.

    The index is time.  The column headings indicate the variable names and
    units.

    :param modelicares.SimRes sim:
        Simulation result object loaded with modelicares.SimRes.
    :param str,list names
         If None (default), then all variables are included.
    :param dict aliases:
        Dictionary of aliases for the variable names

         The keys are the "official" variable names from the Modelica model
         and the values are the names as they should be included in the
         column headings. Any variables not in this list will not be
         aliased. Any unmatched aliases will not be used.
    :param bool with_unit:
        Boolean to determine format of keys. Default value is True.

        If set to True, the unit will be added to the key. As not all modelica-
        result files export the unit information, using with_unit=True can lead
        to errors.

    **Examples:**
    For further examples, please see `to_pandas <http://kdavies4.github.io/ModelicaRes/modelicares.simres.html>`_

    >>> from modelicares import SimRes
    >>> dir_path = os.path.dirname(os.path.dirname(__file__))
    >>> sim = SimRes(dir_path + '\\examples\\data\\ChuaCircuit.mat')
    >>> voltages = sim.names('^[^.]*.v$', re=True)
    >>> to_pandas(sim, voltages) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
                C1.v / V  C2.v / V   G.v / V   L.v / V  Nr.v / V  Ro.v / V
    Time / s
    0.000000    4.000000  0.000000 -4.000000  0.000000  4.000000  0.000000
    5.000000    3.882738  0.109426 -3.773312  0.109235  3.882738  0.000191
    ...
    [514 rows x 6 columns]

    >>> from modelicares import SimRes
    >>> dir_path = os.path.dirname(os.path.dirname(__file__))
    >>> sim = SimRes(dir_path + '\\examples\\data\\ChuaCircuit.mat')
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

    # Avoid mutable argument
    if aliases is None:
        aliases = {}

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
        time_key = 'Time / s'
    else:
        time_key = 'Time'
    return pd.DataFrame(data).set_index(time_key)


def get_trajectories(sim):
    """
    Function to filter time-variant parameters.

    All variables which are trajectories are extracted from the simulation result.
    Either the length of the variable is greater than two, or the values are not
    equal. In both cases, the variable is considered to be a trajectory.

    :param modelicares.SimRes sim:
        Simulation result object loaded with modelicares.SimRes.

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
