import os
from ebcpy import data_types
from ebcpy.tsa.decomposition.decompositor import Decompositer
import pandas as pd
import statsmodels.api as sm



class timeDecompositer(Decompositer):
    """
    According to Thomas Schreiber's paper mini batch k-means produces good
    results when applied on building energy data.Bode, Gerrit, et al.
    "A time series clustering approach for Building
    Automation and Control Systems." Applied energy 238 (2019): 1337-1345.
    The Kmeans clustering class is based on unsupervised learning
    methods to split time-series-data into classes.

    :param str,os.path.normpath cd:
        Working Directory
    :param aixcal.data_types.TimeSeriesData time_series_data:
        Given object contains all trajectories necessary to train
        the decision tree.
    :param list variable_list:
        List containing keys of dataframe of the trajectories
    :param str model_type: 'additive' :
        The model type of the decomposition
    :param int freq default: 1
        The frequence of the decomposition
    """
    _X = pd.DataFrame()
    # kwarg for exporting the created image to a png or not.
    save_image = False

    def __init__(self, cd, time_series_data=None, variable_list=None,
                  model_type='additive', freq=1 , **kwargs):
        """Instantiate instance attributes"""
        super().__init__(cd, **kwargs)
        if not isinstance(time_series_data, data_types.TimeSeriesData):
            raise TypeError("Given time_series_data is of type {} but should"
                            "be of type TimeSeriesData".format(type(time_series_data).__name__))
        if not isinstance(variable_list, (list, str)):
            raise TypeError("Given variable_list is of type {} but should"
                            "be of type list or str".format(type(variable_list).__name__))
        self.variable_list = variable_list
        self.freq = freq
        self.df = time_series_data
        self.df = self.df.droplevel('Tags', axis=1)
        # Data frame with interesting values
        self._X = self.df[variable_list].copy()
        self.model_type = model_type
        self.__dict__.update(kwargs)

    def decomposition(self):
        for varibale in self.variable_list:
            result = sm.tsa.seasonal_decompose(self._X[varibale], model=self.model_type,freq= self.freq)
            if self.save_image:
                self.logger.plot_sasonal_decompose(result, variable_name=varibale)
            self.output_filename = os.path.join(self.cd, "data", "decompositon{:.4}.xlsx".format(varibale))
            pd.DataFrame(result.trend).dropna().to_excel(self.output_filename)
        return result
