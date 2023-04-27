"""Class for scoring rules"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import math


class PerformanceScorer:
    """
    Generates rule scores from a performance function.
    """

    def __init__(self, performance_func: object):
        """
        Args:
            performance_func (object): The method/function to calculate the 
                metric used to score the rules. Should have parameters 
                `y_true`, `y_pred` and `sample_weight`.
        """

        self.performance_func = performance_func

    def fit(self, X_rules: pd.DataFrame, y: pd.Series,
            sample_weight=None) -> pd.DataFrame:
        """
        Generates rule scores from a weighting function.

        Args:
            X_rules (pd.DataFrame): The binary columns associated with the 
                rules.
            y (pd.Series): The binary target column.
            sample_weight (np.array, optional): Row-wise sample_weights to 
                apply. Defaults to None.

        Returns:
            pd.DataFrame: The rule scores applied to the dataset (similar to 
                binary columns, with the rule score replacing the 1 value).
        """

        scores = np.apply_along_axis(lambda x: self.performance_func(
            y_true=y, y_pred=x, sample_weight=sample_weight), axis=0,
            arr=X_rules.values)
        rule_scores = pd.Series(scores, X_rules.columns)

        return rule_scores


class LogRegScorer:
    """
    Generates rule scores from the exponentiated coefficients of a trained 
    Logistic Regression model.
    """

    def __init__(self, *args, **kwargs):
        """        
        Args:
            *args: Positional arguments associated with Sklearn's 
                `LogisisticRegression()` class constructor.            
            **kwargs: Keyword arguments associated with Sklearn's 
                `LogisisticRegression()` class constructor.
        """

        self.args = args
        self.kwargs = kwargs

    def fit(self, X_rules: pd.DataFrame, y: pd.Series,
            sample_weight=None) -> pd.DataFrame:
        """
        Generates rule scores from the coefficients of a trained Logistic 
        Regression model.

        Args:
            X_rules (pd.DataFrame): The binary columns associated with the 
                rules.
            y (pd.Series): The binary target column.
            sample_weight (np.array, optional): Row-wise sample_weights to 
                apply. Defaults to None.

        Returns:
            pd.DataFrame: The rule scores applied to the dataset (similar to
                binary columns, with the rule score replacing the 1 value).
        """

        lr = LogisticRegression(*self.args, **self.kwargs, random_state=0)
        lr.fit(X=X_rules, y=y, sample_weight=sample_weight)
        scores = np.array(list(map(math.exp, lr.coef_[0])))
        rule_scores = pd.Series(scores, X_rules.columns)

        return rule_scores


class RandomForestScorer:
    """
    Generates rule scores from the feature importance of a trained Random 
    Forest model.
    """

    def __init__(self, *args, **kwargs):
        """
        Args:
            *args: Positional arguments associated with Sklearn's 
                `RandomForestClassifier()` class constructor.            
            **kwargs: Keyword arguments associated with Sklearn's 
                `RandomForestClassifier()` class constructor.
        """
        self.args = args
        self.kwargs = kwargs

    def fit(self, X_rules: pd.DataFrame, y: pd.Series,
            sample_weight=None) -> pd.DataFrame:
        """
        Generates rule scores from the feature importance of a trained Random
        Forest model.

        Args:
            X_rules (pd.DataFrame): The binary columns associated with the 
                rules.
            y (pd.Series): The binary target column.
            sample_weight (np.array, optional): Row-wise sample_weights to 
                apply. Defaults to None.

        Returns:
            pd.DataFrame: The rule scores applied to the dataset (similar to 
                binary columns, with the rule score replacing the 1 value).
        """

        rf = RandomForestClassifier(*self.args, **self.kwargs, random_state=0)
        rf.fit(X=X_rules, y=y, sample_weight=sample_weight)
        scores = rf.feature_importances_
        rule_scores = pd.Series(scores, X_rules.columns)

        return rule_scores
