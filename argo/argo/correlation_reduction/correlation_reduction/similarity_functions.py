"""Classes to calculate similarity metrics"""
from sklearn.metrics.pairwise import pairwise_distances
import pandas as pd
import numpy as np


class CosineSimilarity:
    """Computes the cosine similarity between columns in X"""

    def __init__(self, **kwargs):
        """
        **kwargs : Any keyword arguments to be used in the sklearn 
            cosine_similarity function.        
        """
        self.kwargs = kwargs

    def fit(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the cosine similarity between columns in X.

        Args:
            X (pd.DataFrame): Dataframe containing binary columns.

        Returns:
            pd.DataFrame: Dataframe containing the pairwise cosine 
                similarities.
        """
        cos_sim_matrix = 1 - \
            pairwise_distances(X=X.values.T, metric='cosine', **self.kwargs)
        return pd.DataFrame(cos_sim_matrix, index=X.columns, columns=X.columns)


class JaccardSimilarity:
    """ Computes the Jaccard similarity between columns in X"""

    def __init__(self, **kwargs):
        """
        **kwargs : Any keyword arguments to be used in the sklearn 
            pairwise_distances function.        
        """
        self.kwargs = kwargs

    def fit(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the Jaccard similarity between columns in X.

        Args:
            X (pd.DataFrame): Dataframe containing binary columns.

        Returns:
            pd.DataFrame: Dataframe containing the pairwise Jaccard 
                similarities.
        """
        jaccard_matrix = 1 - \
            pairwise_distances(X=X.values.T.astype(
                bool), metric="jaccard", **self.kwargs)
        return pd.DataFrame(jaccard_matrix, index=X.columns, columns=X.columns)
