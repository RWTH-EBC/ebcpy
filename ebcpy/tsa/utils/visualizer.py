"""
Module with classes and function to help visualize
different processes inside the framework. Both plots
and print-function/log-function will be implemented here.
The Visualizer Class inherits the Logger class, as logging
will always be used as a default.
"""

import os
import pydot
import seaborn
from ebcpy.utils.visualizer import Logger
from IPython.display import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_colors
import sklearn.tree as sktree
from io import StringIO
import plotly.express as px


class SegmentationVisualizer(Logger):
    """Visualizer class used for all classification processes.
    More advanced class to not only log ongoing function
    evaluations but also show the process of the functions
    by plotting interesting causalities and saving these plots."""

    plt.ioff()  # Turn of interactive mode.

    def export_decision_tree_image(self, dtree, variable_list):
        """
        Saves the given dtree object by exporting it
        via graphviz to a png image

        :param DecisionTree dtree:
        :param list variable_list:
            List with names of decision-variables
        """
        # Save the created tree as a png.
        try:
            # Visualization decision tree
            dot_data = StringIO()
            sktree.export_graphviz(dtree,
                                   out_file=dot_data,
                                   feature_names=variable_list, filled=True, rounded=True)
            graph = pydot.graph_from_dot_data(dot_data.getvalue())
            Image(graph[0].create_png())  # Creating the image needs some time
            plt.show(graph[0])
            graph[0].write_png(os.path.join(self.cd, 'tree_plot.png'))
        except OSError:
            self.log("ERROR: Can not export the decision tree, "
                     "please install graphviz on your machine.")

    def plot_decision_tree(self, df, class_list):
        """Visualization pair plot (df is data frame with whole X values (train and test)
        This function takes a long time to be executed.

        :param pd.DataFrame df:
        :param list class_list:
            List with names for classes.
        """

        seaborn.pairplot(df, hue=class_list)
        plt.savefig(os.path.join(self.cd, 'pairplot.png'), bbox_inches='tight', dpi=400)

    def plot_mini_batch_kmeans(self, df, labels):
        """Visualization pair plot (df is data frame with whole X values (train and test)
        This function takes a long time to be executed.

        :param pd.DataFrame df:
            DataFrame with whole X values (train and test)
        :param list labels:
            List with names for classes.
        """

        #colors = ['blue', 'green', 'deepskyblue', 'red']
        time = df['Time'][0:len(df)]
        df = df.drop(columns = ['Time', 'Class'])
        for i in range(df.shape[1]):
            plt.figure(figsize=(6, 4)).add_axes([0.1,
                                                 0.1,
                                                 0.8,
                                                 0.8]).scatter(time,
                                                               df[df.columns[i]][0:len(df)],
                                                               c=labels,
                                                            #cmap=mpl_colors.ListedColormap(colors),
                                                               s=8)
            plt.savefig(os.path.join(self.cd, 'KmeansPairPlot{:.4}.png'.format(df.columns[i])), bbox_inches='tight', dpi=400)

    def plot_TICC(self, df, labels):
        """Visualization pair plot (df is data frame with whole X values (train and test)
        This function takes a long time to be executed.

        :param pd.DataFrame df:
            DataFrame with whole X values (train and test)
        :param list labels:
            List with names for classes.
        """

        # colors = ['blue', 'green', 'deepskyblue', 'red']
        for i in range(df.shape[1]):
            plt.figure(figsize=(6, 4)).add_axes([0.1,
                                                 0.1,
                                                 0.8,
                                                 0.8]).scatter(df['Time'][0:len(df)],
                                                               df[df.columns[i]][0:len(df)],
                                                               c=labels,
                                                               # cmap=mpl_colors.ListedColormap(colors),
                                                               s=8)
            plt.savefig(os.path.join(self.cd, 'TICCPairPlot{:.4}.png'.format(df.columns[i])), bbox_inches='tight',
                        dpi=400)

    def plot_MASA(self, df, labels):
        """Visualization pair plot (df is data frame with whole X values (train and test)
        This function takes a long time to be executed.

        :param pd.DataFrame df:
            DataFrame with whole X values (train and test)
        :param list labels:
            List with names for classes.
        """

        # colors = ['blue', 'green', 'deepskyblue', 'red']
        for i in range(df.shape[1]):
            plt.figure(figsize=(6, 4)).add_axes([0.1,
                                                 0.1,
                                                 0.8,
                                                 0.8]).scatter(df['Time'][0:len(df)],
                                                               df[df.columns[i]][0:len(df)],
                                                               c=labels,
                                                               # cmap=mpl_colors.ListedColormap(colors),
                                                               s=8)
            plt.savefig(os.path.join(self.cd, 'MASAPairPlot{:.4}.png'.format(df.columns[i])), bbox_inches='tight',
                        dpi=400)


    def plot_kmedoids(self, df, labels):
        """Visualization pair plot (df is data frame with whole X values (train and test)
        This function takes a long time to be executed.

        :param pd.DataFrame df:
            DataFrame with whole X values (train and test)
        :param list labels:
            List with names for classes.
        """

        #colors = ['blue', 'green', 'deepskyblue', 'red']
        for i in range(df.shape[1]):
            plt.figure(figsize=(6, 4)).add_axes([0.1,
                                                 0.1,
                                                 0.8,
                                                 0.8]).scatter(df['Time'][0:len(df)],
                                                               df[df.columns[i]][0:len(df)],
                                                               c=labels,
                                                            #cmap=mpl_colors.ListedColormap(colors),
                                                               s=8)
            plt.savefig(os.path.join(self.cd, 'KmedoidsPairPlot{:.4}.png'.format(df.columns[i][0])), bbox_inches='tight', dpi=400)

    def plot_mini_batch_kmeans_plotly(self, df):
        """Visualization pair plot (df is data frame with whole X values (train and test)
        This function takes a long time to be executed.

        :param pd.DataFrame df:
            DataFrame with whole X values (train and test)
        :param list labels:
            List with names for classes.
        """
        try:
            for columns in df.drop(columns = ['Time', 'Class']).columns:
                fig = px.scatter(df, x="Time", y=columns, color="Class")
                fig.write_image(os.path.join(self.cd, 'KmeansPlot{:.4}.svg'.format(columns)))
                fig.write_image(os.path.join(self.cd, 'KmeansPlot{:.4}.png'.format(columns)))
        except ValueError:
            self.log("ERROR: Can not export the pair plot, "
                     "please install plotly-orca on your machine via:\n"
                     "'conda install -c plotly plotly-orca'")

    def plot_kmedoids_plotly(self, df):
        """Visualization pair plot (df is data frame with whole X values (train and test)
        This function takes a long time to be executed.

        :param pd.DataFrame df:
            DataFrame with whole X values (train and test)
        :param list labels:
            List with names for classes.
        """
        try:
            for colummn in df.columns:
                fig = px.scatter(df, x="Time", y=colummn, color="Class")
                fig.write_image(os.path.join(self.cd, 'kmedoidsPlot{:.4}.svg'.format(colummn)))
                fig.write_image(os.path.join(self.cd, 'kmedoidsPlot{:.4}.png'.format(colummn)))
        except ValueError:
            self.log("ERROR: Can not export the pair plot, "
                     "please install plotly-orca on your machine via:\n"
                     "'conda install -c plotly plotly-orca'")

    def plot_TICC_plotly(self, df):
        """Visualization pair plot (df is data frame with whole X values (train and test)
        This function takes a long time to be executed.

        :param pd.DataFrame df:
            DataFrame with whole X values (train and test)
        :param list labels:
            List with names for classes.
        """
        try:
            for colummn in df.columns:
                fig = px.scatter(df, x="Time", y=colummn, color="Class")
                fig.write_image(os.path.join(self.cd, 'ticcPlot{:.4}.svg'.format(colummn)))
                fig.write_image(os.path.join(self.cd, 'ticcPlot{:.4}.png'.format(colummn)))
        except ValueError:
            self.log("ERROR: Can not export the pair plot, "
                     "please install plotly-orca on your machine via:\n"
                     "'conda install -c plotly plotly-orca'")

    def plot_MASA_plotly(self, df):
        """Visualization pair plot (df is data frame with whole X values (train and test)
        This function takes a long time to be executed.

        :param pd.DataFrame df:
            DataFrame with whole X values (train and test)
        :param list labels:
            List with names for classes.
        """
        try:
            for colummn in df.columns:
                fig = px.scatter(df, x="Time", y=colummn, color="Class")
                fig.write_image(os.path.join(self.cd, 'masaPlot{:.4}.svg'.format(colummn)))
                fig.write_image(os.path.join(self.cd, 'masaPlot{:.4}.png'.format(colummn)))
        except ValueError:
            self.log("ERROR: Can not export the pair plot, "
                     "please install plotly-orca on your machine via:\n"
                     "'conda install -c plotly plotly-orca'")


    def plot_sasonal_decompose(self, decompose_result, variable_name):
        resplot = decompose_result.plot()
        resplot.savefig(os.path.join(self.cd, 'Decomposition{:.4}.png'.format(variable_name)), bbox_inches='tight', dpi=400)

