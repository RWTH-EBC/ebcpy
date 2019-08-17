"""Module for calculating statistical
measures based on given methods."""

import numpy as np
import sklearn.metrics as skmetrics


class StatisticsAnalyzer:
    """Class for calculation of the statistical measure based on the
    given method. Either instantiate the class and run
    StatisticsAnalyzer.calc(meas, sim), or go for direct calculation with
    StatisticsAnalyzer.calc_METHOD(meas, sim).

    :param str method:
        One of the following:
            - MAE(Mean absolute error)
            - R2(coefficient of determination)
            - MSE (Mean squared error)
            - RMSE(root mean square error)
            - CVRMSE(variance of RMSE)
            - NRMSE(Normalized RMSE)
    """

    def __init__(self, method):
        """Instantiate class parameters"""
        _supported_methods = {"mae": self.calc_mae,
                              "r2": self.calc_r2,
                              "mse": self.calc_mse,
                              "rmse": self.calc_rmse,
                              "cvrmse": self.calc_cvrmse,
                              "nrmse": self.calc_nrmse}

        # Remove case-sensitive input
        _method_internal = method.lower()

        if _method_internal not in _supported_methods:
            raise ValueError("The given method {} is not supported.\n Choose one out "
                             "of: {}".format(_method_internal, ", ".join(_supported_methods.keys())))
        self.calc = _supported_methods[_method_internal]

    @staticmethod
    def calc(meas, sim):
        """Placeholder class before instantiating the class correctly."""
        raise NotImplementedError('Instantiate the class to call this function.\n For direct '
                                  'analysis, call calc_METHOD with the method you want to use.')

    @staticmethod
    def calc_mae(meas, sim):
        """
        Calculates the MAE (mean absolute error)
        for the given numpy array of measured and simulated data.

        :param np.array meas:
            Array with measurement data
        :param np.array sim:
            Array with simulation data
        :return: float MAE:
            MAE os the given data.
        """
        return skmetrics.mean_absolute_error(meas, sim)

    @staticmethod
    def calc_r2(meas, sim):
        """
        Calculates the MAE (mean absolute error)
        for the given numpy array of measured and simulated data.

        :param np.array meas:
            Array with measurement data
        :param np.array sim:
            Array with simulation data
        :return: float MAE:
            R2 of the given data.
        """
        return 1 - skmetrics.r2_score(meas, sim)

    @staticmethod
    def calc_mse(meas, sim):
        """
        Calculates the MSE (mean square error)
        for the given numpy array of measured and simulated data.

        :param np.array meas:
            Array with measurement data
        :param np.array sim:
            Array with simulation data
        :return: float MSE:
            MSE of the given data.
        """
        return skmetrics.mean_squared_error(meas, sim)

    @staticmethod
    def calc_rmse(meas, sim):
        """
        Calculates the RMSE (root mean square error)
        for the given numpy array of measured and simulated data.

        :param np.array meas:
            Array with measurement data
        :param np.array sim:
            Array with simulation data
        :return: float RMSE:
            RMSE of the given data.
        """
        return np.sqrt(skmetrics.mean_squared_error(meas, sim))

    @staticmethod
    def calc_nrmse(meas, sim):
        """
        Calculates the NRMSE (normalized root mean square error)
        for the given numpy array of measured and simulated data.

        :param np.array meas:
            Array with measurement data
        :param np.array sim:
            Array with simulation data
        :return: float NRMSE:
            NRMSE of the given data.
        """
        # Check if NRMSE can be calculated
        if (np.max(meas) - np.min(meas)) != 0:
            return np.sqrt(skmetrics.mean_squared_error(meas, sim)) \
                   / (np.max(meas) - np.min(meas))
        else:
            raise ValueError("The given measurement data's maximum is equal to "
                             "it's minimum. This makes the calculation of the "
                             "NRMSE impossible. Choose another method.")

    @staticmethod
    def calc_cvrmse(meas, sim):
        """
        Calculates the CVRMSE (variance of root mean square error)
        THIS IS A TEST
        for the given numpy array of measured and simulated data.

        :param np.array meas:
            Array with measurement data
        :param np.array sim:
            Array with simulation data
        :return: float CVRMSE:
            CVRMSE of the given data.
        """
        # Check if CVRMSE can be calculated
        
        if np.mean(meas) != 0:
            return np.sqrt(skmetrics.mean_squared_error(meas, sim)) / np.mean(meas)
        else:
            raise ValueError("The given measurement data has a mean of 0. "
                             "This makes the calculation of the CVRMSE impossible. "
                             "Choose another method.")
