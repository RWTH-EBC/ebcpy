"""Module for clustering using unsupervised learning-techniques."""

from ebcpy.tsa.segmentation.clusterer import Clusterer
import pandas as pd
from ticc.RunProblem import RunTicc
import numpy as np
import os


class TiccClusterer(Clusterer):
    """
    The TICC clustering class is based on unsupervised learning
    methods to split time-series-data into classes.

    :param str,os.path.normpath cd:
        Working Directory
    :param int cluster_number:
        the number of the cluster number
    :param str input_filename:
        The path of the input data
    :param str output_filename:
        The path of the output data
    :param aixcal.data_types.TimeSeriesData time_series_data:
        Given object contains all trajectories necessary to train
        the decision tree.
    :param list variable_list:
        List containing keys of dataframe of the trajectories
    :param ticc.RunProblem  RunTicc :
        the runner of Ticc provide from ticc
    """

    _X = pd.DataFrame()
    # kwarg for exporting the created image to a png or not.
    save_image = False

    def __init__(self, cd, output_filename,cluster_number, time_series_data, variable_list=None,
                  delimiter=',',**kwargs):
        """Instantiate instance attributes"""
        super().__init__(cd, **kwargs)
        self._class_used_for_fitting = True
        self.variable_list = variable_list
        # Data frame with interesting values
        self.df = time_series_data
        self.df = self.df.droplevel('Tags', axis=1)
        self.delimiter = delimiter
        self._X = self.df[variable_list].copy()
        self.convert_inputdata()
        self.output_filename = output_filename
        self.cluster_number = cluster_number
        self.__dict__.update(kwargs)

    def cluster(self, cluster_number=range(2, 11), process_pool_size=10,
            window_size=1, lambda_param=[1e-2], beta=[0.01, 0.1, 0.5, 10, 50, 100, 500],
            maxIters=1000, threshold=2e-5, covariance_filename=None,
            input_format='matrix', BIC_Iters=15, input_dimensions=50, **kwargs):
        """
        Clustering of given data in dataframe with ticc.
        Optional Parameters: BIC
        For each of these parameters, one can choose to specify:
            - a single number: this value will be used as the parameter
            - a list of numbers: the solver will use grid search on the BIC to choose the parameter
            - not specified: the solver will grid search on a default range (listed) to choose the parameter
        :param cluster_number: The number of clusters to classify. Default: BIC on [2...10]
        :param lambda_param: sparsity penalty. Default: BIC on 11e-2]
        :param beta: the switching penalty. If not specified, BIC on [50, 100, 200, 400]

        Other Optional Parameters:
        :param input_dimensions:
            if specified, will truncated SVD the matrix to the given number of features
            if the input is a graph, or PCA it if it's a matrix
        :param BIC_iters:
            if specified, will only run BIC tuning for the given number of iterations
        :param process_pool_size:
            the number of processes to spin off for optimization. Default 1
        :param window_size:
            The size of the window for each cluster. Default 1
        :param maxIters:
            the maximum number of iterations to allow TICC to run. Default 1000
        :param threshold:
            the convergence threshold. Default 2e-5
        :param covariance_filename:
            if not None, write the covariance into this file
        :param file_type is the type of data file. the data file must
           be a comma separated CSV. the options are:
           -- "matrix": a numpy matrix where each column is a feature and each
              row is a time step
           -- "graph": an adjacency list with each row having the form:
              <start label>, <end label>, value
        :param delimiter is the data file delimiter

        :param pd.DataFrame df:
            Given dataframe may be extracted from the TimeSeriesData class. Should
            contain all relevant keys.
        :return: list
            List containing dicts with names, start and stop-time
        """
        # Predict classes for test data set
        predictions = RunTicc(input_filename=self.input_filename, output_filename=self.output_filename,
                              cluster_number=self.cluster_number, delimiter=self.delimiter,
                              process_pool_size=process_pool_size,
                              window_size=window_size, lambda_param=lambda_param, beta=beta,
                              maxIters=maxIters, threshold=threshold, covariance_filename=covariance_filename,
                              input_format=input_format, BIC_Iters=BIC_Iters, input_dimensions=input_dimensions)
        column = 0
        self._X['Class'] = 0
        self._X['Time'] = 0
        for classnumber in predictions[0][0]:
            self._X['Class'][column] = classnumber
            self._X['Time'][column] = column
            column = column + 1

        if self.save_image:
            self.logger.plot_TICC(self._X, self._X['Class'])
        return predictions

    def convert_inputdata(self):
        self.input_filename = os.path.join(self.cd , "data","test.txt")
        np.savetxt(X=self._X, fname= self.input_filename, delimiter=',', fmt='%.10f')