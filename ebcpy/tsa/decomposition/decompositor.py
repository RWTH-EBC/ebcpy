"""Base-module for the clustering-package.
"""

from abc import abstractmethod
import os
from ebcpy.tsa.utils import visualizer


class Decompositer:
    """
    Base-Class for a decompositer. All decompositers should
    be able to process some MeasTarget-Data and MeasInput-Data
    into classes of format: Name, Start-Time, Stop-Time.

    :param str,os.path.normpath cd:
        Working directory for storing logs and plots
    """

    def __init__(self, cd, **kwargs):
        """Instantiate instance parameters"""
        #%%Set class parameters
        if not os.path.isdir(cd):
            os.mkdir(cd)
        self.cd = cd
        self.logger = visualizer.SegmentationVisualizer(cd, self.__class__.__name__)

        #%% Update kwargs:
        self.__dict__.update(kwargs)

    @abstractmethod
    def decompositer(self, df, **kwargs):
        """
        Base function for executing the clustering based on the given data.

        :param pd.DataFrame df:
            Given dataframe may be extracted from the TimeSeriesData class. Should
            contain all relevant keys.
        :return: list
            List containing dicts with names, start and stop-time
        """
        raise NotImplementedError('{}.decomposite function is not '
                                  'defined'.format(self.__class__.__name__))