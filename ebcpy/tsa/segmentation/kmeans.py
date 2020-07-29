"""Module for clustering using unsupervised learning-techniques."""

import os
import pickle
import warnings
from ebcpy import data_types
from ebcpy.tsa.segmentation.clusterer import Clusterer
from sklearn import __version__ as sk_version
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import Normalizer


class KmeansClusterer(Clusterer):
    """
    According to Thomas Schreiber's paper mini batch k-means produces good
    results when applied on building energy data.Bode, Gerrit, et al.
    "A time series clustering approach for Building
    Automation and Control Systems." Applied energy 238 (2019): 1337-1345.
    The Kmeans clustering class is based on unsupervised learning
    methods to split time-series-data into classes.

    :param str,os.path.normpath cd:
        Working Directory
    :param int n_clutsters:
        The number of class mini Batch Kmeans shold fit
    :param aixcal.data_types.TimeSeriesData time_series_data:
        Given object contains all trajectories necessary to train
        the decision tree.
    :param list variable_list:
        List containing keys of dataframe of the trajectories
    :param sklearn.cluster.MiniBatchKMeans mini_kmeans:
        Mini Batch K-Means that is already fitted.
    :param int random_state defaut: 0 :
        Determines random number generation for centroid initialization
        and random reassignment.
    :param int batch_size default: 100
        size of the mini batches
    """

    # Dummy object for the later calculated MiniBatchKMeans.
    _mini_kmeans = MiniBatchKMeans()
    _trained_successfully = False
    _class_used_for_fitting = True
    _X = pd.DataFrame()
    # kwarg for exporting the created image to a png or not.
    save_image = False

    def __init__(self, cd, n_clusters, norm='max', time_series_data=None, variable_list=None,
                 mini_kmeans=None, random_state=0, batch_size=6, **kwargs):
        """Instantiate instance attributes"""
        super().__init__(cd, **kwargs)
        if mini_kmeans is None:
            self._class_used_for_fitting = True

            if not isinstance(time_series_data, data_types.TimeSeriesData):
                raise TypeError("Given time_series_data is of type {} but should"
                                "be of type TimeSeriesData".format(type(time_series_data).__name__))
            if not isinstance(variable_list, (list, str)):
                raise TypeError("Given variable_list is of type {} but should"
                                "be of type list or str".format(type(variable_list).__name__))
            self.variable_list = variable_list
            self.df = time_series_data
            self.df = self.df.droplevel('Tags', axis=1)
            # Data frame with interesting values
            self._X = self.df[variable_list].copy()
            self.n_clusters = n_clusters
            self.random_state = random_state
            self.batch_size = batch_size


        else:
            self._class_used_for_fitting = False
            self._trained_successfully = True
            if not isinstance(mini_kmeans, MiniBatchKMeans):
                raise TypeError("Given mini_kmeans is of type {} but should"
                                "be of type MiniBatchKMeans".format(
                    type(mini_kmeans).__name__))
            self._mini_kmeans = mini_kmeans
        self.norm= norm
        self.__dict__.update(kwargs)

    def create_mini_batch_kmeans(self):
        """Creates a mini batch kmeans based on the training data
        defined in this class. If wanted, the mini batch kmeans can
        be exported as a image.

        :return sklearn.cluster.MiniBatchKMeans _mini_kmeans:
            May be used for storing of further processing.

        """
        if not self._class_used_for_fitting:
            raise AttributeError("When instantiating this class, you passed an existing"
                                 "mini_kmeans. Therefore, you can't create or validate a new one. "
                                 "Re-Instatiate the class with the necessary arguments.")

        # Create mini kmean instance. Fit the known classes to the known training-data
        self._mini_kmeans = MiniBatchKMeans(n_clusters=self.n_clusters,
                                            random_state=self.random_state,
                                            batch_size=self.batch_size)

        df_norm = self.normalize(self._X.values)
        self._mini_kmeans.fit(df_norm)
        fit_predict = self._mini_kmeans.fit_predict(df_norm)
        self._X['Class'] = fit_predict
        self._X['Time'] = self.df[self.df.columns[0]]

        # Export image
        if self.save_image:
            self.logger.plot_mini_batch_kmeans(self._X, fit_predict)
            self.logger.plot_mini_batch_kmeans_plotly(self._X)

        # Set info if the mini kmeans was successfully created.
        self._trained_successfully = True
        return self._mini_kmeans

    def export_mini_batch_kmeans_to_pickle(self, mini_kmeans=None, savepath=None, info=None):
        """
        Exports the given mini batch kmeans in form of a pickle and
        stores it on your machine. To avoid losses of data in future
        versions, the version number is stored alongside the kmeans-object.

        :param sklearn.cluster.MiniBatchKMeans,optional mini_kmeans:
            If no mini_kmeans is given, the mini_kmeans of this class will be saved.
        :param str,optional savepath:
            If not savepath is given, the pickle is stored in the
            current working directory.
        :param str,optional info:
            Provide some info string on which columns should be passed when
            using this mini_kmeans.
        """
        if mini_kmeans is None:
            _mini_kmeans = self._mini_kmeans
        else:
            _mini_kmeans = mini_kmeans
        if not isinstance(_mini_kmeans, MiniBatchKMeans):
            raise TypeError("Given mini_kmeans is of type {} but should be"
                            "of type MiniBatchKMeans".format(type(_mini_kmeans).__name__))

        if savepath is None:
            _savepath = os.path.join(self.cd, "mini_kmeans_export.pickle")
        else:
            _savepath = savepath

        with open(_savepath, "wb") as pickle_file:
            pickle.dump({"mini_kmeans": _mini_kmeans,
                         "version": sk_version,
                         "info": info}, pickle_file)

        return _savepath

    @staticmethod
    def load_mini_batch_kmeans_from_pickle(filepath):
        """
        Loads the given pickle file and checks for the correct
        version of sklearn

        :param str,os.path.normpath filepath:
            Path of the pickle file
        :return: sklearn.cluster.MiniBatchKMeans mini_kmeans
            Loaded Mini Batch K-Means
        """

        with open(filepath, "rb") as pickle_file:
            dumped_dict = pickle.load(pickle_file)
        if dumped_dict["version"] != sk_version:
            warnings.warn("Saved mini_kmeans is under version {} but you are using {}. "
                          "Different behaviour of the mini_kmeans may "
                          "occur.".format(dumped_dict["version"], sk_version))
        return dumped_dict["mini_kmeans"], dumped_dict["info"]

    def cluster(self, df, **kwargs):
        """
        Clustering of given data in dataframe with
        a mini-kmeans-clusterer of sklearn. If no mini_kmeans
        object is given, the current dtree of this class will
        be used.

        :param pd.DataFrame df:
            Given dataframe may be extracted from the TimeSeriesData class. Should
            contain all relevant keys.
        :keyword sklearn.cluster.MiniBatchKMeans mini_kmeans:
            If not provided, the current class mini_kmeans will be used.
            You can create a mini_kmeans and export it using this class's methods.
        :return: list
            List containing dicts with names, start and stop-time
        """
        # If no mini kmeans is provided, the class create mini batch kmeans will be used.
        if "mini_kmeans" not in kwargs:
            # If the class mini_batch kmeans is not trained yet, the training will be executed here.
            if not self._trained_successfully:
                self.create_mini_batch_kmeans()
            mini_kmeans = self._mini_kmeans
        else:
            mini_kmeans = kwargs.get("mini_kmeans")
            if not isinstance(mini_kmeans, MiniBatchKMeans):
                raise TypeError("Given mini_kmeans is of type {} but should be of type "
                                "sklearn.cluster.MiniBatchKMeans".format(
                    type(mini_kmeans).__name__))

        if not isinstance(df, pd.DataFrame):
            raise TypeError("Given df is of type {} but should "
                            "be pd.DataFrame".format(type(df).__name__))

        classes = []

        # Predict classes for test data set
        df_norm = self.normalize(df.values)
        predictions = mini_kmeans.predict(df_norm)
        predictions = predictions.astype(str)
        # Convert predictions to classes.
        pred_df = pd.DataFrame({"time": df.index, "pred": predictions}).set_index("time")
        pred_df = pred_df.loc[pred_df["pred"].shift(-1) != pred_df["pred"]]
        _last_stop_time = df.index[0]
        for idx, row in pred_df.iterrows():
            stop_time = idx
            temp_class = {"name": row["pred"],
                          "start_time": _last_stop_time,
                          "stop_time": stop_time}
            classes.append(temp_class)
            _last_stop_time = stop_time

        return classes

    def normalize(self, array):
        """
        Normalizer is the process of scaling individual samples to have unit norm and important for clustering.
        It make the calculation quickly.
        :param array: the data to normalize
        :param str,optional norm: the norm to user to normalize each non zero sample
        :return: the normalized data
        """
        transformer = Normalizer(norm=self.norm).fit(array.T)
        array_norm = transformer.transform(array.T)
        return array_norm.T