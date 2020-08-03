"""Module for clustering using unsupervised learning-techniques."""

import os
import pickle
import warnings
from ebcpy import data_types
from ebcpy.tsa.segmentation.clusterer import Clusterer
from sklearn import __version__ as sk_version
import pandas as pd
from sklearn.preprocessing import Normalizer
from pyclustering.cluster.kmedoids import kmedoids
from ebcpy.tsa.utils import reweight

class KmedoidsClusterer(Clusterer):
    """
    The Kmedoids clustering class is based on unsupervised learning
    methods to split time-series-data into classes.

    :param str,os.path.normpath cd:
        Working Directory
    :param list initial_medoids:
        The indexes of the point in input data
    :param string metric:
        what distance metric to use
    :param aixcal.data_types.TimeSeriesData time_series_data:
        Given object contains all trajectories necessary to train
        the decision tree.
    :param list variable_list:
        List containing keys of dataframe of the trajectories
    :param pyclustering.cluster.kmedoids  kmedoids :
        kmedoids that is already fitted.
    """

    # Dummy object for the later calculated kmedoids.
    _kmedoids = kmedoids
    _trained_successfully = False
    _class_used_for_fitting = True
    _X = pd.DataFrame()
    # kwarg for exporting the created image to a png or not.
    save_image = False

    def __init__(self, cd, initial_index_medoids, weight_faktor=None, metric='euclidean', norm ='max', time_series_data=None, variable_list=None,
                 k_medoids=None ,**kwargs):
        """Instantiate instance attributes"""
        super().__init__(cd, **kwargs)
        if k_medoids is None:
            self._class_used_for_fitting = True

            if not isinstance(time_series_data, data_types.TimeSeriesData):
                raise TypeError("Given time_series_data is of type {} but should"
                                "be of type TimeSeriesData".format(type(time_series_data).__name__))
            if not isinstance(variable_list, (list, str)):
                raise TypeError("Given variable_list is of type {} but should"
                                "be of type list or str".format(type(variable_list).__name__))
            self.variable_list = variable_list
            self.df = time_series_data
            # Data frame with interesting values
            self._X = self.df[variable_list].copy()
            self.metric = metric
            self.weight_faktor = weight_faktor
            self.initial_index_medoids = initial_index_medoids

        else:
            self._class_used_for_fitting = False
            self._trained_successfully = True
            if not isinstance(k_medoids, kmedoids):
                raise TypeError("Given k_medoids is of type {} but should"
                                "be of type KMedoids".format(
                    type(k_medoids).__name__))
            self._kmedoids = k_medoids
        self.norm = norm
        self.__dict__.update(kwargs)

    def create_kmedoids(self):
        """Creates a kmedoids based on the training data
        defined in this class. If wanted, the kmedoids can
        be exported as a image.

        :return pyclustering.cluster.kmedoids  kmedoids:
            May be used for storing of further processing.

        """
        if not self._class_used_for_fitting:
            raise AttributeError("When instantiating this class, you passed an existing"
                                 "kmedoids. Therefore, you can't create or validate a new one. "
                                 "Re-Instatiate the class with the necessary arguments.")

        # Create kmedoids instance. Fit the known classes to the known training-data
        df_input = self.normalize(self._X.values)
        if self.weight_faktor:
            df_input = reweight.reweight(df_input, self.weight_faktor)
        self._kmedoids = kmedoids(data=df_input, initial_index_medoids=self.initial_index_medoids
                                         )
        # Run kmedoids to train the process
        self._kmedoids.process()

        # Get the class after training
        cluster_class =self._kmedoids.get_clusters()

        #reformat the dataframe, add the class column to the dataframe
        classnumber = 0
        self._X['Class'] = 0
        for column in cluster_class:
            self._X['Class'][column] = classnumber
            classnumber = classnumber + 1

        self._X['Time'] = self.df[self.df.columns[0]]

        # Export image
        if self.save_image:
            self.logger.plot_kmedoids(self._X, self._X['Class'])
            self.logger.plot_kmedoids_plotly=(self._X)

        # Set info if the kmedoids was successfully created.
        self._trained_successfully = True
        return self._kmedoids

    def export_kmedoids_to_pickle(self, k_medoids=None, savepath=None, info=None):
        """
        Exports the given kmedoids in form of a pickle and
        stores it on your machine. To avoid losses of data in future
        versions, the version number is stored alongside the kmedoids-object.

        :param sklearn.cluster.kmedoids,optional kmedoids:
            If no kmedoids is given, the kmedoids of this class will be saved.
        :param str,optional savepath:
            If not savepath is given, the pickle is stored in the
            current working directory.
        :param str,optional info:
            Provide some info string on which columns should be passed when
            using this kmedoids.
        """
        if k_medoids is None:
            _kmedoids = self._kmedoids
        else:
            _kmedoids = k_medoids
        if not isinstance(_kmedoids, kmedoids):
            raise TypeError("Given kmedoids is of type {} but should be"
                            "of type KMedoids)".format(type(_kmedoids).__name__))

        if savepath is None:
            _savepath = os.path.join(self.cd, "kmedoids_export.pickle")
        else:
            _savepath = savepath


        return _kmedoids

    @staticmethod
    def load_kmedoids_from_pickle(filepath):
        """
        Loads the given pickle file and checks for the correct
        version of sklearn

        :param str,os.path.normpath filepath:
            Path of the pickle file
        :return: pyclustering.cluster.kmedoids  kmedoids
            Loaded kmedoids
        """

        with open(filepath, "rb") as pickle_file:
            dumped_dict = pickle.load(pickle_file)
        if dumped_dict["version"] != sk_version:
            warnings.warn("Saved kmedoids is under version {} but you are using {}. "
                          "Different behaviour of the kmedoids may "
                          "occur.".format(dumped_dict["version"], sk_version))
        return dumped_dict["kmedoids"], dumped_dict["info"]


    def load_kmedoids(filepath):
        """
        Loads the given pickle file and checks for the correct
        version of sklearn

        :param str,os.path.normpath filepath:
            Path of the pickle file
        :return: pyclustering.cluster.kmedoids  kmedoids
            Loaded kmedoids
        """

        with open(filepath, "rb") as pickle_file:
            dumped_dict = pickle.load(pickle_file)
        if dumped_dict["version"] != sk_version:
            warnings.warn("Saved kmedoids is under version {} but you are using {}. "
                          "Different behaviour of the kmedoids may "
                          "occur.".format(dumped_dict["version"], sk_version))
        return dumped_dict["kmedoids"], dumped_dict["info"]

    def cluster(self, df, weight_faktor=None, **kwargs):
        """
        Clustering of given data in dataframe with
        a k_medoids-clusterer of sklearn. If no k_medoids
        object is given, the current dtree of this class will
        be used.

        :param pd.DataFrame df:
            Given dataframe may be extracted from the TimeSeriesData class. Should
            contain all relevant keys.
        :keyword pyclustering.cluster.kmedoids.kmedoids:
            If not provided, the current class k_medoids will be used.
            You can create a k_medoids and export it using this class's methods.
        :return: list
            List containing dicts with names, start and stop-time
        """
        # If no k_medoids is provided, the class creat k_medoids will be used.
        if "k_medoids" not in kwargs:
            # If the class kmediods is not trained yet, the training will be executed here.
            if not self._trained_successfully:
                self.create_kmedoids()
            k_medoids = self._kmedoids
        else:
            k_medoids = kwargs.get("k_medoids")
            if not isinstance(k_medoids, kmedoids):
                raise TypeError("Given k_medoids is of type {} but should be of type "
                                "sklearn.cluster.Kmedoids".format(
                    type(k_medoids).__name__))

        if not isinstance(df, pd.DataFrame):
            raise TypeError("Given df is of type {} but should "
                            "be pd.DataFrame".format(type(df).__name__))

        classes = []
        # Predict classes for test data set
        df_input = self.normalize(df.values)
        if self.weight_faktor:
            df_input = reweight.reweight(df_input, weight_faktor)
        predictions = k_medoids.predict(df_input)

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

        return classes, pred_df

    def normalize(self, array):
        """
        Normalizer is the process of scaling individual samples to have unit norm and important for clustering.
        It make the calculation quickly.
        :param array:
            the data to normalize
        :param str,optional norm:
            the norm to user to normalize each non zero sample
        :return array:
            the normalized data
        """
        transformer = Normalizer(norm=self.norm).fit(array.T)
        array_norm = transformer.transform(array.T)
        return array_norm.T

    def get_kmedoids(self):
        return self._kmedoids

