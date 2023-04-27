"""Classes for reducing correlated features"""
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import numpy as np
import seaborn as sns


class AgglomerativeClusteringFeatureReduction:

    """
    Removes similar columns (given a similarity function) by calculating the 
    simility matrix then iteratively running Agglomerative Clustering on the 
    similarity matrix and dropping columns that are correlated. Only one column 
    per cluster is kept.

    Attributes:
        columns_to_keep (list): The final list of columns with the correlated 
            columns removed.
    """

    def __init__(self, threshold: float, strategy: str,
                 similarity_function: object, columns_performance=None):
        """
        Args:
            threshold (float): The median of the cluster's simility metric is 
                compared against this threshold - if the median is greater than 
                this threshold, the columns within the cluster are deemed 
                correlated, with only the top performing column being kept.
            strategy (str): Can be either 'top_down' or 'bottom_up'. 'top_down' 
                begins clustering from the top, with two clusters per iteration 
                being calculated. 'bottom_up' begins clustering from the 
                bottom, with half of the total number of columns per iteration 
                being used to define the number of clusters.
            similarity_function (object): The similarity function to use for 
                calculating the similarity between columns. It must return a 
                dataframe containing the simility matrix. See the 
                simility_functions module for out-of-the-box functions.
            columns_performance (pd.Series, optional): Series containing the 
                performance metric of each column (e.g. Fbeta score). This is 
                used to determine the top performing column per cluster. If not
                provided, a random column from the cluster will be kept. 
                Defaults to None.
        """

        self.threshold = threshold
        if strategy not in ['top_down', 'bottom_up']:
            raise Exception('strategy must be either top_down or bottom_up')
        self.strategy = strategy
        self.similarity_function = similarity_function
        self.columns_performance = columns_performance
        self.columns_to_keep = []

    def fit(self, X: pd.DataFrame, print_clustermap=False) -> None:
        """
        Calculates the similar columns in the dataset X.

        Args:
            X (pd.DataFrame): Dataframe to be reduced.
            print_clustermap (bool, optional): If True, the clustermap at each 
                iteration will be printed. Defaults to False.
        """
        zero_var_cols = X.columns[X.values.var(axis=0) == 0.0].tolist()
        if zero_var_cols:
            raise Exception(
                f'Columns {", ".join(zero_var_cols)} have zero variance, which will result in NaN values for the similarity matrix')
        similarity_df = self.similarity_function(X)
        num_remaining_columns = similarity_df.shape[1]
        # While more than 1 column remains in the similarity_df, continue to
        # cluster and drop correlated columns
        while num_remaining_columns > 1:
            n_clusters = self._set_n_clusters(similarity_df)
            clusters = self._agglomerative_clustering(
                similarity_df, n_clusters=n_clusters)
            if print_clustermap:
                self._plot_clustermap(similarity_df)
            if self.strategy == 'top_down':
                columns_to_drop = self._top_down(
                    clusters=clusters, n_clusters=n_clusters, similarity_df=similarity_df)
            elif self.strategy == 'bottom_up':
                columns_to_drop = self._bottom_up(
                    clusters=clusters, n_clusters=n_clusters, similarity_df=similarity_df)
            if columns_to_drop:
                similarity_df.drop(columns_to_drop, axis=1, inplace=True)
                similarity_df.drop(columns_to_drop, axis=0, inplace=True)
                num_remaining_columns = similarity_df.shape[1]
            else:
                self.columns_to_keep = self.columns_to_keep + similarity_df.columns.tolist()
                break

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Removes similar columns from the dataset X.

        Args:
            X (pd.DataFrame): Dataframe to be reduced.

        Returns:
            pd.DataFrame: Dataframe with the similar columns removed.
        """

        return X[self.columns_to_keep]

    def fit_transform(self, X: pd.DataFrame,
                      print_clustermap=False) -> pd.DataFrame:
        """
        Calculates the similar columns in the dataset X, then removes them.        

        Args:
            X (pd.DataFrame): Dataframe of binary columns.
            print_clustermap (bool, optional): If True, the clustermap at each iteration will be printed. Defaults to False.

        Returns:
            pd.DataFrame: Dataframe of dissimilar binary columns.
        """

        self.fit(X, print_clustermap=print_clustermap)
        return self.transform(X)

    def _bottom_up(self, clusters: pd.Series, n_clusters: int,
                   similarity_df: pd.DataFrame) -> list:
        """
        Begins clustering from the bottom, with half of the total number of 
        columns per iteration being used to define the number of clusters
        """

        columns_to_drop = []
        for n_cluster in range(0, n_clusters):
            cluster = clusters[clusters == n_cluster].index
            similarity_cluster = similarity_df.loc[cluster, cluster]
            # If the cluster contains one column only, continue to next
            # iteration
            if similarity_cluster.shape == (1, 1):
                continue
            cluster_median = self._calculate_cluster_median(
                similarity_cluster=similarity_cluster)
            # If cluster_median > threshold, keep top performing column by
            # Fscore only
            if cluster_median > self.threshold:
                columns = similarity_cluster.columns.tolist()
                top_performer = self._get_top_performer(
                    columns=columns, columns_performance=self.columns_performance)
                # If bottom_up, drop all columns in cluster except top
                # performer
                columns.remove(top_performer)
                # If bottom_up and only one cluster remains, keep top
                # performing column.
                if similarity_df.shape == (2, 2):
                    self.columns_to_keep.append(top_performer)
                columns_to_drop = columns_to_drop + columns
        return columns_to_drop

    def _top_down(self, clusters: pd.Series, n_clusters: int,
                  similarity_df: pd.DataFrame) -> list:
        """
        Begins clustering from the top, with two clusters per iteration being 
        calculated
        """

        columns_to_drop = []
        for n_cluster in range(0, n_clusters):
            cluster = clusters[clusters == n_cluster].index
            similarity_cluster = similarity_df.loc[cluster, cluster]
            if similarity_cluster.shape == (1, 1):
                self.columns_to_keep.append(similarity_cluster.columns[0])
                continue
            cluster_median = self._calculate_cluster_median(
                similarity_cluster=similarity_cluster)
            # If cluster_median > threshold, keep top performing column by
            # Fscore only
            if cluster_median > self.threshold:
                columns = similarity_cluster.columns.tolist()
                top_performer = self._get_top_performer(
                    columns=columns, columns_performance=self.columns_performance)
                # For top_down, keep top performing column then drop all
                # columns in cluster for next iteration
                self.columns_to_keep.append(top_performer)
                columns_to_drop = columns_to_drop + columns
        return columns_to_drop

    def _set_n_clusters(self, similarity_df: pd.DataFrame) -> int:
        """Sets the number of clusters to use"""

        if self.strategy == 'top_down':
            n_clusters = 2
        elif self.strategy == 'bottom_up':
            n_clusters = int(similarity_df.shape[0]/2)
        return n_clusters

    @staticmethod
    def _calculate_cluster_median(similarity_cluster: pd.DataFrame) -> float:
        """Calculates the median of a cluster"""

        mask = np.triu(np.ones(similarity_cluster.shape), k=1).astype(bool)
        cluster_median = np.nanmedian(similarity_cluster.where(mask).values)
        return cluster_median

    @staticmethod
    def _get_top_performer(columns: list,
                           columns_performance: pd.Series) -> str:
        """
        Returns the top performing column in a cluster by it's performance 
        (if provided). If not provided, it will just return a column from the 
        cluster
        """

        if columns_performance is not None:
            performance = columns_performance.loc[columns].sort_values(
                ascending=False)
            top_performer = performance.index[0]
        else:
            top_performer = columns[0]
        return top_performer

    @staticmethod
    def _agglomerative_clustering(similarity_df: pd.DataFrame,
                                  n_clusters: int) -> pd.Series:
        """
        Performs Agglomerative Clustering on a dataframe of similarities and 
        returns the cluster each column falls into
        """

        agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
        ac_preds = agg_clustering.fit_predict(similarity_df)
        clusters = pd.Series(ac_preds, similarity_df.columns)
        return clusters

    @staticmethod
    def _plot_clustermap(similarity_df: pd.DataFrame) -> sns.clustermap:
        """Plots the clustermap of a given similarity dataframe"""

        sns.clustermap(similarity_df)
