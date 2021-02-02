"""
Module based on the simres module of modelicares. As no new content is going to be
merged upstream, this "fork" of the to_pandas() function is used.

Update 18.01.2021:
As modelicares is no longer compatible with matplotlib > 3.3.2, we integrated all
necessary functions from modelicares to still be able and use SimRes.to_pandas().
# TODO: If we put this open-source, how do we cite this?!
"""
import os
from fnmatch import fnmatchcase
from itertools import count
from collections import namedtuple
from scipy.io import loadmat
from scipy.io.matlab.mio_utils import chars_to_strings
from six import PY2
import pandas as pd
import numpy as np
import re as regexp


# Namedtuple to store the time and value information of each variable
Samples = namedtuple('Samples', ['times', 'values', 'negated'])


def flatten_list(l, ltypes=(list, tuple)):
    """
    Flatten a nested list.

    **Arguments:**

    - *l*: List (may be nested to an arbitrary depth)

          If the type of *l* is not in ltypes, then it is placed in a list.

    - *ltypes*: Tuple (not list) of accepted indexable types

    **Example:**

    >>> flatten_list([1, [2, 3, [4]]])
    [1, 2, 3, 4]
    """
    # Based on
    # http://rightfootin.blogspot.com/2006/09/more-on-python-flatten.html,
    # 10/28/2011
    ltype = type(l)
    if ltype not in ltypes: # So that strings aren't split into characters
        return [l]
    l = list(l)
    i = 0
    while i < len(l):
        while isinstance(l[i], ltypes):
            if l[i]:
                l[i:i + 1] = l[i]
            else:
                l.pop(i)
                i -= 1
                break
        i += 1
    return ltype(l)


def match(strings, pattern=None, re=False):
    r"""Reduce a list of strings to those that match a pattern.

    By default, all of the strings are returned.

    **Arguments:**

    - *strings*: List of strings

    - *pattern*: Case-sensitive string used for matching

      - If *re* is *False* (next argument), then the pattern follows the
        Unix shell style:

        ============   ============================
        Character(s)   Role
        ============   ============================
        \*             Matches everything
        ?              Matches any single character
        [seq]          Matches any character in seq
        [!seq]         Matches any char not in seq
        ============   ============================

        Wildcard characters ('\*') are not automatically added at the
        beginning or the end of the pattern.  For example, '\*x\*' matches all
        strings that contain "x", but 'x\*' matches only the strings that begin
        with "x".

      - If *re* is *True*, regular expressions are used a la `Python's re
        module <http://docs.python.org/2/library/re.html>`_.  See also
        http://docs.python.org/2/howto/regex.html#regex-howto.

        Since :mod:`re.search` is used to produce the matches, it is as if
        wildcards ('.*') are automatically added at the beginning and the
        end.  For example, 'x' matches all strings that contain "x".  Use '^x$'
        to match only the strings that begin with "x" and 'x$' to match only the
        strings that end with "x".

        Note that '.' is a subclass separator in Modelica_ but a wildcard in
        regular expressions.  Escape the subclass separator as '\\.'.

    - *re*: *True* to use regular expressions (*False* to use shell style)

    **Example:**

    >>> match(['apple', 'orange', 'banana'], '*e')
    ['apple', 'orange']


    .. _Modelica: http://www.modelica.org/
    """
    if pattern is None or (pattern in ['.*', '.+', '.', '.?', ''] if re
                           else pattern == '*'):
        return list(strings) # Shortcut
    else:
        if re:
            matcher = regexp.compile(pattern).search
        else:
            matcher = lambda name: fnmatchcase(name, pattern)
        return list(filter(matcher, strings))


def loadsim(fname, constants_only=False):
    r"""Load Dymola\ :sup:`®` or OpenModelica simulation results.

    **Arguments:**

    - *fname*: Name of the results file, including the path

         The file extension ('.mat') is optional.

    - *constants_only*: *True* to load only the variables from the first data
      matrix

         The first data matrix usually contains all of the constants,
         parameters, and variables that don't vary.  If only that information is
         needed, it may save resources to set *constants_only* to *True*.

    **Returns:** An instance of :class:`~modelicares.simres._VarDict`, a
    specialized dictionary of variables (instances of :class:`Variable`)

    **Example:**

    >>> variables = loadsim('examples/ChuaCircuit.mat')
    >>> variables['L.v'].unit
    'V'
    """
    # This does the task of mfiles/traj/tload.m from the Dymola installation.

    def parse(description):
        """Parse the variable description string into description, unit, and
        displayUnit.
        """
        description = description.rstrip(']')
        displayUnit = ''
        try:
            description, unit = description.rsplit('[', 1)
        except ValueError:
            unit = ''
        else:
            try:
                unit, displayUnit = unit.rsplit('|', 1)
            except ValueError:
                pass # (displayUnit = '')
        description = description.rstrip()
        if PY2:
            description = description.decode('utf-8')

        return description, unit, displayUnit

    # Load the file.
    mat, Aclass = read(fname, constants_only)

    # Check the type of results.
    if Aclass[0] == 'AlinearSystem':
        raise AssertionError(fname + ' is a linearization result.  Use LinRes '
                             'instead.')
    else:

        assert Aclass[0] == 'Atrajectory', (fname + ' is not a simulation or '
                                            'linearization result.')

    # Determine if the data is transposed.
    try:
        transposed = Aclass[3] == 'binTrans'
    except IndexError:
        transposed = False
    else:
        assert transposed or Aclass[3] == 'binNormal', (
            'The orientation of the Dymola/OpenModelica results is not '
            'recognized.  The third line of the "Aclass" variable is "%s", but '
            'it should be "binNormal" or "binTrans".' % Aclass[3])

    # Get the format version.
    version = Aclass[1]

    # Process the name, description, parts of dataInfo, and data_i variables.
    # This section has been optimized for speed.  All time and value data
    # remains linked to the memory location where it is loaded by scipy.  The
    # negated variable is carried through so that copies are not necessary.  If
    # changes are made to this code, be sure to compare the performance (e.g.,
    # using %timeit in IPython).
    if version == '1.0':
        data = mat['data'].T if transposed else mat['data']
        times = data[:, 0]
        names = get_strings(mat['names'].T if transposed else mat['names'])
        variables = _VarDict({name: Variable(Samples(times, data[:, i], False),
                                             '', '', '')
                              for i, name in enumerate(names)})
    else:
        assert version == '1.1', ('The version of the Dymola/OpenModelica '
                                  'result file ({v}) is not '
                                  'supported.'.format(v=version))
        names = get_strings(mat['name'].T if transposed else mat['name'])
        descriptions = get_strings(mat['description'].T if transposed else
                                   mat['description'])
        dataInfo = mat['dataInfo'] if transposed else mat['dataInfo'].T
        data_sets = dataInfo[0, :]
        sign_cols = dataInfo[1, :]
        variables = _VarDict()
        for i in count(1):
            try:
                data = (mat['data_%i' % i].T if transposed else
                        mat['data_%i' % i])
            except KeyError:
                break # There are no more "data_i" variables.
            else:
                if data.shape[1] > 1: # In case the data set is empty.
                    times = data[:, 0]
                    variables.update({name:
                                      Variable(Samples(times,
                                                       data[:,
                                                            abs(sign_col) - 1],
                                                       sign_col < 0),
                                               *parse(description))
                                      for (name, description, data_set,
                                           sign_col)
                                      in zip(names, descriptions, data_sets,
                                             sign_cols)
                                      if data_set == i})

        # Time is from the last data set.
        variables['Time'] = Variable(Samples(times, times, False),
                                     'Time', 's', '')

    return variables


def read(fname, constants_only=False):
    r"""Read variables from a MATLAB\ :sup:`®` file with Dymola\ :sup:`®` or
    OpenModelica results.

    **Arguments:**

    - *fname*: Name of the results file, including the path

         This may be from a simulation or linearization.

    - *constants_only*: *True* to load only the variables from the first data
      matrix, if the result is from a simulation

    **Returns:**

    1. A dictionary of variables

    2. A list of strings from the lines of the 'Aclass' matrix
    """

    # Load the file.
    try:
        if constants_only:
            mat = loadmat(fname, chars_as_strings=False, appendmat=False,
                          variable_names=['Aclass', 'class', 'name', 'names',
                                          'description', 'dataInfo', 'data',
                                          'data_1', 'ABCD', 'nx', 'xuyName'])
        else:
            mat = loadmat(fname, chars_as_strings=False, appendmat=False)
    except IOError:
        raise IOError('"{f}" could not be opened.'
                      '  Check that it exists.'.format(f=fname))

    # Check if the file contains the Aclass variable.
    try:
        Aclass = mat['Aclass']
    except KeyError:
        raise TypeError('"{f}" does not appear to be a Dymola or OpenModelica '
                        'result file.  The "Aclass" variable is '
                        'missing.'.format(f=fname))

    return mat, get_strings(Aclass)


def get_strings(str_arr):
    """Return a list of strings from a character array.

    Strip the whitespace from the right and recode it as utf-8.
    """
    return [line.rstrip(' \0').encode('latin-1').decode('utf-8')
            for line in chars_to_strings(str_arr)]


class _VarDict(dict):
    """Special dictionary for simulation variables (instances of
    :class:`Variable`)
    """

    def __getattr__(self, attr):
        """Look up a property for each of the variables (e.g., n_constants).
        """
        return util.CallList([getattr(variable, attr)
                              for variable in self.values()])

    def __getitem__(self, key):
        """Include suggestions in the error message if a variable is missing.
        """
        try:
            return dict.__getitem__(self, key)
        except KeyError:
            msg = '%s is not a valid variable name.' % key
            close_matches = get_close_matches(key, self.keys())
            if close_matches:
                msg += "\n       ".join(["\n\nDid you mean one of these?"]
                                        + close_matches)
            raise LookupError(msg)


class Variable(namedtuple('VariableNamedTuple', ['samples', 'description', 'unit', 'displayUnit'])):
    """Special namedtuple_ to represent a variable in a simulation, with
    methods to retrieve and perform calculations on its values

    This class is usually not instantiated directly by the user, but instances
    are returned when indexing a variable name from a simulation result
    (:class:`SimRes` instance).
    """

    def times(self):
        return self.samples.times

    def values(self):
        return -self.samples.values if self.samples.negated else self.samples.values


class SimRes:
    """Class to load, analyze, and plot results from a Modelica_ simulation

    **Initialization arguments:**

    - *fname*: Name of the file (including the directory if necessary)

    - *constants_only*: *True* to load only the variables from the first
      data table

         The first data table typically contains all of the constants,
         parameters, and variables that don't vary.  If only that
         information is needed, it may save resources to set
         *constants_only* to *True*.

    **Other methods:**

    - :meth:`names` - Return a list of variable names, optionally filtered by
      pattern matching.

    - :meth:`to_pandas` - Return a `pandas DataFrame`_ with selected variables.

    **Properties:**

    - *dirname* - Directory from which the variables were loaded

    - *fbase* - Base filename from which the results were loaded, without the
      directory or file extension.

    - *fname* - Filename from which the variables were loaded, with absolute
      path

    - *n_constants* - Number of variables that do not change over time

    **Example:**

    >>> sim = SimRes('examples/ChuaCircuit.mat')
    >>> print(sim) # doctest: +ELLIPSIS
    Modelica simulation results from .../examples/ChuaCircuit.mat


    .. _Python: http://www.python.org/
    .. _pandas DataFrame: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html?highlight=dataframe#pandas.DataFrame
    """

    # pylint: disable=R0921

    def __init__(self, fname='dsres.mat', constants_only=False):
        """Upon initialization, load Modelica_ simulation results from a file.

        See the top-level class documentation.
        """

        # Load the file and store the variables.
        self._variables = loadsim(fname, constants_only)

        # Remember the tool and filename.
        self.fname = os.path.abspath(fname)

    def __repr__(self):
        """Return a formal description of an instance of this class.
        """
        return "{Class}('{fname}')".format(Class=self.__class__.__name__,
                                           fname=self.fname)
        # Note:  The class name is inquired so that this method will still be
        # correct if the class is extended.

    @property
    def dirname(self):
        """Directory from which the variables were loaded
        """
        return os.path.dirname(self.fname)

    @property
    def fbase(self):
        """Base filename from which the variables were loaded, without the
        directory or file extension
        """
        return os.path.splitext(os.path.basename(self.fname))[0]

    def to_pandas(self, names=None, aliases=None, with_unit=True):
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
        For further examples, please see
        `to_pandas <http://kdavies4.github._io/ModelicaRes/modelicares.simres.html>`_

        >>> from ebcpy.modelica.simres import SimRes
        >>> dir_path = os.path.dirname(os.path.dirname(__file__))
        >>> sim = SimRes(dir_path + '//examples//data//ChuaCircuit.mat')
        >>> voltages = sim.names('^[^.]*.v$', re=True)
        >>> sim.to_pandas(voltages) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
                    C1.v / V  C2.v / V   G.v / V   L.v / V  Nr.v / V  Ro.v / V
        Time / s
        0.000000    4.000000  0.000000 -4.000000  0.000000  4.000000  0.000000
        5.000000    3.882738  0.109426 -3.773312  0.109235  3.882738  0.000191
        ...
        [514 rows x 6 columns]

        >>> from ebcpy.modelica.simres import SimRes
        >>> dir_path = os.path.dirname(os.path.dirname(__file__))
        >>> sim = SimRes(dir_path + '//examples//data//ChuaCircuit.mat')
        >>> voltages = sim.names('^[^.]*.v$', re=True)
        >>> sim.to_pandas(voltages, with_unit=False) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
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
            names = set(flatten_list(names))
            names.add('Time')
        else:
            names = self.names()

        # Create a dictionary of names and values.
        times = self._variables['Time'].values()
        data = {}
        for name in names:

            # Get the values.
            if np.array_equal(self._variables[name].times(), times):
                values = self._variables[name].values()  # Save computation.
            # Check if all values are constant to save resampling time
            elif np.count_nonzero(self._variables[name].values() - np.max(self._variables[name].values())) == 0:
                # Passing a scalar converts automatically to an array.
                values = np.max(self._variables[name].values())
            else:
                values = self._variables[name].values(t=times)  # Resample.

            unit = self._variables[name].unit

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

    def get_trajectories(self):
        """
        Function to filter time-variant parameters.

        All variables which are trajectories are extracted from the simulation result.
        Either the length of the variable is greater than two, or the values are not
        equal. In both cases, the variable is considered to be a trajectory.

        **Examples:**
        >>> from ebcpy.modelica.simres import SimRes
        >>> sim = SimRes('examples/ChuaCircuit.mat')
        >>> trajectory_names = sim.get_trajectories()
        >>> len(trajectory_names)
        39
        """
        trajectory_names = []
        for name in self.names():
            values = self._variables[name].values()
            # If the value array is greater then two, it is always a trajectory
            if len(values) > 2:
                trajectory_names.append(name)
            # Special Case: Only two time-steps are simulated.
            # In that case, if the last value does not equal the first value, it is also a trajectory
            elif len(values) == 2 and values[0] != values[-1]:
                trajectory_names.append(name)
        return trajectory_names

    def names(self, pattern=None, re=False, constants_only=False):
        r"""Return a list of variable names that match a pattern.

        By default, all names are returned.

        **Arguments:**

        - *pattern*: Case-sensitive string used for matching

          - If *re* is *False* (next argument), then the pattern follows the
            Unix shell style:

            ============   ============================
            Character(s)   Role
            ============   ============================
            \*             Matches everything
            ?              Matches any single character
            [seq]          Matches any character in seq
            [!seq]         Matches any char not in seq
            ============   ============================

            Wildcard characters ('\*') are not automatically added at the
            beginning or the end of the pattern.  For example, '\*x\*' matches
            all variables that that contain "x", but 'x\*' matches only the
            variables that begin with "x".

          - If *re* is *True*, the regular expressions are used a la `Python's
            res module <http://docs.python.org/2/library/re.html>`_.  See also
            http://docs.python.org/2/howto/regex.html#regex-howto.

            Since :mod:`re.search` is used to produce the matches, it is as if
            wildcards ('.*') are automatically added at the beginning and the
            end.  For example, 'x' matches all variables that contain "x".  Use
            '^x$' to match only the variables that begin with "x" and 'x$' to
            match only the variables that end with "x".

            Note that '.' is a subclass separator in Modelica_ but a wildcard in
            regular expressions.  Escape subclass separators as '\\.'.

        - *re*: *True* to use regular expressions (*False* to use shell style)

        - *constants_only*: *True* to include only the variables that do not
          change over time

        **Example:**

        .. code-block:: python

           >>> sim = SimRes('examples/ChuaCircuit.mat')

           >>> # Names for voltages across all of the components:
           >>> sim.names('^[^.]*.v$', re=True) # doctest: +SKIP
           ['C1.v', 'C2.v', 'G.v', 'L.v', 'Nr.v', 'Ro.v']

        .. testcleanup::

           >>> sorted(sim.names('^[^.]*.v$', re=True))
           ['C1.v', 'C2.v', 'G.v', 'L.v', 'Nr.v', 'Ro.v']
        """
        # Get a list of all the variables or just the constants.
        if constants_only:
            names = (name for (name, variable) in self._variables.items()
                     if variable.is_constant)
        else:
            names = self._variables.keys()

        # Filter the list and return it.
        return match(names, pattern, re)
