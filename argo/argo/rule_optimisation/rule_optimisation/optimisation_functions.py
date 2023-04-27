"""Classes for calculating optimisation metrics"""
import numpy as np
import pandas as pd
from typing import Union
import argo_utils.argo_utils as argo_utils


class Precision:
    """Calculates the Precision using Numpy"""

    def fit(self,
            y_true: Union[np.array, pd.Series],
            y_pred: Union[np.array, pd.Series],
            sample_weight=None) -> float:
        """
        Calculates the Precision using Numpy.

        Args:
            y_true (Union[np.array, pd.Series]): The target column.
            y_pred (Union[np.array, pd.Series]): The binary predictor column.
            sample_weight ([type], optional): Row-wise weights to apply. 
                Defaults to None.

        Returns:
            float: The Precision score.
        """
        precision = argo_utils.return_binary_pred_perf_of_set_numpy(y_true=y_true,
                                                                    y_preds=y_pred,
                                                                    y_preds_columns=[
                                                                        'y_pred'],
                                                                    sample_weight=sample_weight).iloc[0]['Precision']
        return precision


class Recall:
    """Calculates the Recall using Numpy"""

    def fit(self,
            y_true: Union[np.array, pd.Series],
            y_pred: Union[np.array, pd.Series],
            sample_weight=None) -> float:
        """
        Calculates the Recall using Numpy.

        Args:
            y_true (Union[np.array, pd.Series]): The target column.
            y_pred (Union[np.array, pd.Series]): The binary predictor column.
            sample_weight ([type], optional): Row-wise weights to apply. 
                Defaults to None.

        Returns:
            float: The Recall score.
        """

        recall = argo_utils.return_binary_pred_perf_of_set_numpy(y_true=y_true,
                                                                 y_preds=y_pred,
                                                                 y_preds_columns=[
                                                                     'y_pred'],
                                                                 sample_weight=sample_weight).iloc[0]['Recall']
        return recall


class FScore:
    """Calculates the Fbeta score using Numpy"""

    def __init__(self, beta: float):
        """
        Args:
            beta (float): The beta value used to calculate the Fbeta score.
        """
        self.beta = beta

    def fit(self,
            y_true: Union[np.array, pd.Series],
            y_pred: Union[np.array, pd.Series],
            sample_weight=None) -> float:
        """
        Calculates the Fbeta score using Numpy.

        Args:
            y_true (Union[np.array, pd.Series]): The target column.
            y_pred (Union[np.array, pd.Series]): The binary predictor column.
            sample_weight ([type], optional): Row-wise weights to apply. 
                Defaults to None.

        Returns:
            float: The Fbeta score.
        """
        precision, recall = argo_utils.return_binary_pred_perf_of_set_numpy(y_true=y_true,
                                                                            y_preds=y_pred,
                                                                            y_preds_columns=[
                                                                                'y_pred'],
                                                                            sample_weight=sample_weight).iloc[0][['Precision', 'Recall']]
        if precision == 0 or recall == 0:
            fscore = 0
        else:
            fscore = (1 + self.beta ** 2) * ((precision * recall) /
                                             ((precision * self.beta ** 2) + recall))
        return fscore


class Revenue:
    """Calculates the revenue"""

    def __init__(self, y_type: str, chargeback_multiplier: int):
        """
        Args:
            y_type (str): Dictates whether the binary target column flags fraud 
                (y_type = 'Fraud') or non-fraud (y_type = 'NonFraud').
            chargeback_multiplier (int): Multiplier to apply to chargeback 
                transactions.            
        """

        if y_type not in ['Fraud', 'NonFraud']:
            raise ValueError('y_type must be either "Fraud" or "NonFraud"')
        self.y_type = y_type
        self.chargeback_multiplier = chargeback_multiplier

    def fit(self,
            y_true: Union[np.array, pd.Series],
            y_pred: Union[np.array, pd.Series],
            sample_weight: Union[np.array, pd.Series]) -> float:
        """
        Calculates the revenue.

        Args:
            y_true (Union[np.array, pd.Series]): The target column.
            y_pred (Union[np.array, pd.Series]): The binary predictor column.
            sample_weight (Union[np.array, pd.Series]): Row-wise transaction 
                amounts to apply.

        Returns:
            float: Revenue.
        """
        if isinstance(y_true, pd.Series):
            y_true = np.asarray(y_true)
        if isinstance(y_pred, pd.Series):
            y_pred = np.asarray(y_pred)
        if isinstance(sample_weight, pd.Series) and sample_weight is not None:
            sample_weight = np.asarray(sample_weight)
        tps = np.sum(y_true * y_pred * sample_weight)
        tns = np.sum(np.where(y_true == 0, 1, 0) *
                     np.where(y_pred == 0, 1, 0) * sample_weight)
        fps = np.sum(np.where(y_true == 0, 1, 0) * y_pred * sample_weight)
        fns = np.sum(y_true * np.where(y_pred == 0, 1, 0) * sample_weight)
        if self.y_type == 'Fraud':
            revenue = self.chargeback_multiplier * (tps - fns) + tns - fps
        elif self.y_type == 'NonFraud':
            revenue = tps - fns + self.chargeback_multiplier * (tns - fps)
        return revenue


class AlertsPerDay:
    """
    Calculates the negative squared difference between the number of alerts per
    day in the binary predictor vs the expected.
    """

    def __init__(self, n_alerts_expected_per_day: int,
                 no_of_days_in_file: int):
        """
        Args:
            n_alerts_expected_per_day (int): expected number of alerts for the 
                given rule.
            no_of_days_in_file (int): number of days of data provided in the 
                file.
        """

        self.n_alerts_expected_per_day = n_alerts_expected_per_day
        self.no_of_days_in_file = no_of_days_in_file

    def fit(self, y_pred: Union[np.array, pd.Series]) -> float:
        """
        Calculates the negative squared difference between the number of alerts 
        per day in the binary predictor vs the expected.

        Args:
            y_pred (Union[np.array, pd.Series]): The binary predictor column.
            n_alerts_expected_per_day : The expected number of alerts per day 
                for the given rule.
            no_of_days_in_file : The number of days of data provided in 
                `y_pred`.

        Returns:
            float: The negative squared difference between the number of alerts 
                per day in `y_pred` vs `n_alerts_expected_per_day`.
        """

        if isinstance(y_pred, pd.Series):
            y_pred = np.asarray(y_pred)
        n_alerts_per_day = np.sum(y_pred)/self.no_of_days_in_file
        f_min = (n_alerts_per_day-self.n_alerts_expected_per_day) ** 2
        return -f_min


class PercVolume:
    """
    Calculates the negative squared difference between the percentage of the 
    overall volume that the binary predictor flags vs the expected.    
    """

    def __init__(self, perc_vol_expected: float):
        """
        Args:
            perc_vol_expected (float): expected percentage of the overall
                volume that the binary predictor should flag.            
        """

        self.perc_vol_expected = perc_vol_expected

    def fit(self, y_pred: Union[np.array, pd.Series]) -> float:
        """
        Calculates the negative squared difference between the percentage of 
        the overall volume that the binary predictor flags vs the expected. 

        Args:
            y_pred (Union[np.array, pd.Series]): The binary predictor column.
            n_alerts_expected_per_day : The expected number of alerts per day 
                for the given rule.
            no_of_days_in_file : The number of days of data provided in 
                `y_pred`.

        Returns:
            float: the negative squared difference between the percentage of 
                the overall volume that `y_pred` flags vs `perc_vol_expected`.              
        """

        if isinstance(y_pred, pd.Series):
            y_pred = np.asarray(y_pred)
        perc_flagged = y_pred.mean()
        f_min = (perc_flagged-self.perc_vol_expected) ** 2
        return -f_min
