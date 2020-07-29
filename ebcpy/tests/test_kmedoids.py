"""Test-module for all classes inside
enstats.tsa.clustering.kmedoids"""
import unittest
import os
import shutil
import numpy as np
from ebcpy import data_types
from pyclustering.cluster import kmedoids
from ebcpy.tsa import KmedoidsClusterer



class TestClustering(unittest.TestCase):
    """
    Test the clustering class of TiSA
    """

    def setUp(self):
        """Called before every test.
        Used to setup relevant paths and APIs etc."""
        self.framework_dir = os.path.dirname(os.path.dirname(__file__))
        self.example_dir = os.path.normpath(self.framework_dir + "//examples//data")
        self.example_xlsx_input = os.path.normpath(self.example_dir +
                                                   "//classifier_input.xlsx")
        # Load timeSeriesData
        self.tsd = data_types.TimeSeriesData(self.example_xlsx_input,sheet_name="classifier_input")
        self.example_clas_dir = os.path.normpath(self.example_dir + "//test_classifier")
        # List with column names that should be part of the classifier analysis
        self.variable_names = ["T_sink / K", "T_source / K", "m_flow_sink / kg/s"]

    def test_decision_tree_classifier(self):
        """Test class DecisionTreeClassifier and all it's main methods"""
        # Use a random size between 1 and 100 to ensure all values will work
        _test_size = np.random.randint(1, 100)
        _test_clusters_n = 3
        _test_metric = "euclidean"
        _test_initial_index_medoids = [1, 100, 200]
        # Instantiate class
        KmedoidsCluster = KmedoidsClusterer(self.example_dir,
                                                                   initial_index_medoids =_test_initial_index_medoids,
                                                                   time_series_data=self.tsd,
                                                                   variable_list=self.variable_names
                                                                   )
        KmedoidsCluster.save_image = True
        kmedoids_cluster = KmedoidsCluster.create_kmedoids()

        self.assertIsInstance(kmedoids_cluster, kmedoids.kmedoids)
        # This will only work if plotly-orca is installed. Therefor we skip the test
        #self.assertTrue(os.path.isfile(self.example_dir + "//kmedoidsPlot.png"))

        KmedoidsCluster.export_kmedoids_to_pickle()
        classes = KmedoidsCluster.cluster(self.tsd.df[self.variable_names])
        for temp_class in classes:
            self.assertIsInstance(temp_class, dict)


    def tearDown(self):
        """Delete all files created while testing"""
        try:
            shutil.rmtree(self.example_clas_dir)
        except (FileNotFoundError, PermissionError):
            pass


if __name__ == "__main__":
    unittest.main()
