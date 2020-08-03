"""Test-module for all classes inside
ebcpy.tsa.segmentation.kmeans"""
import unittest
import os
import shutil
from ebcpy.tsa import KmeansClusterer
from ebcpy import data_types
from sklearn.cluster import MiniBatchKMeans
import numpy as np


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
        self.class_names = 'class'  # Column name where classes are listed

    def test_KmeansClusterer(self):
        """Test class DecisionTreeClassifier and all it's main methods"""
        # Use a random size between 1 and 100 to ensure all values will work
        _test_size = np.random.randint(1, 100)
        _test_clusters_n = 3
        # Instantiate class
        KmeansCluster = KmeansClusterer(self.example_dir,
                                        n_clusters=_test_clusters_n,
                                        time_series_data=self.tsd,
                                        variable_list=self.variable_names,
                                        batch_size=_test_size)
        KmeansCluster.save_image = True
        mini_batch_kmeans = KmeansCluster.create_mini_batch_kmeans()

        self.assertIsInstance(mini_batch_kmeans, MiniBatchKMeans)
        # This will only work if plotly-orca is installed. Therefor we skip the test
        #self.assertTrue(os.path.isfile(self.example_dir + "//KmeansPairPlot.png"))

        KmeansCluster.export_mini_batch_kmeans_to_pickle()
        variable_names_multi_index = list(map(tuple, [[var_name, 'raw'] for var_name in self.variable_names]))
        classes = KmeansCluster.cluster(self.tsd[variable_names_multi_index])
        for temp_class in classes:
            self.assertIsInstance(temp_class, dict)

        # Test-save of kmeans:
        filepath = KmeansCluster.export_mini_batch_kmeans_to_pickle(mini_kmeans=mini_batch_kmeans)
        self.assertTrue(os.path.isfile(filepath))
        # Test-load of kmeans:
        kmeans_loaded, _ = KmeansCluster.load_mini_batch_kmeans_from_pickle(filepath)
        # Create new classifier-object with loaded tree:
        KmeansCluster_loaded = KmeansClusterer(self.example_clas_dir,
                                               n_clusters=_test_clusters_n,
                                               mini_kmeans=kmeans_loaded)
        # Classify again with loaded tree and check if the output is still the same.
        classes_loaded = KmeansCluster_loaded.cluster(self.tsd[variable_names_multi_index])

        for (i, temp_class) in enumerate(classes):
            self.assertEqual(temp_class["name"], classes_loaded[i]["name"])
            self.assertEqual(temp_class["stop_time"], classes_loaded[i]["stop_time"])
            self.assertEqual(temp_class["start_time"], classes_loaded[i]["start_time"])

    def tearDown(self):
        """Delete all files created while testing"""
        try:
            shutil.rmtree(self.example_clas_dir)
        except (FileNotFoundError, PermissionError):
            pass


if __name__ == "__main__":
    unittest.main()
