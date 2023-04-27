"""Class for scaling rule scores"""
from sklearn.preprocessing import minmax_scale
import pandas as pd


class ConstantScaler:
    """
    Scales rule scores using the formula (depending on the sign of the rule 
    scores):

        For negative scores: `x_scaled = (limit / x_min) * x`        

        For positive scores: `x_scaled = (limit / x_max) * x`

    where the `limit` parameter is specified in the class constructor. Note that 
    the scores are also converted to int.
    """

    def __init__(self, limit: int):
        """
        Args:
            limit (int): The limit to apply when scaling the scores.            
        """
        self.limit = limit

    def fit(self, rule_scores: pd.Series) -> pd.Series:
        """
        Scales rule scores using the formula (depending on the sign of the 
        rule scores):

            For negative scores: `x_scaled = (limit / x_min) * x`

            For positive scores: `x_scaled = (limit / x_max) * x`

        where the `limit` parameter is specified in the class constructor. Note
        that the scores are also converted to int.

        Args:
            rule_scores (pd.Series): Rule scores to scale.

        Returns:
            pd.Series: The scaled rule scores.
        """

        if all(rule_scores <= 0):
            multiplier = self.limit / rule_scores.min()
            rule_scores_scaled = rule_scores * multiplier
        elif all(rule_scores >= 0):
            multiplier = self.limit / rule_scores.max()
            rule_scores_scaled = rule_scores * multiplier
        else:
            raise ValueError(
                'rule_scores must contain only negative scores or only positive scores, not a mixture')
        return round(rule_scores_scaled).astype(int)


class MinMaxScaler:
    """
    Scales rule scores using the formula:

        `x_scaled = (x - x_min) / (x_max - x_min)`

    Note that the scores are also converted to int.
    """

    def __init__(self, min_value: int, max_value: int):
        """
        Args:
            min_value (int): The minimum value of the scaled rule score range.
            max_value (int): The maximum value of the scaled rule score range.
        """
        self.min_value = min_value
        self.max_value = max_value

    def fit(self, rule_scores: pd.Series) -> pd.Series:
        """
        Scales rule scores using the formula:

            `x_scaled = (x - x_min) / (x_max - x_min)`

        Note that the scores are also converted to int.

        Args:
            rule_scores (pd.Series): Rule scores to scale.

        Returns:
            pd.Series: The scaled rule scores.
        """
        if not (all(rule_scores <= 0) or all(rule_scores >= 0)):
            raise ValueError(
                'rule_scores must contain only negative scores or only positive scores, not a mixture')
        if self.min_value < 0 and all(rule_scores >= 0):
            rule_scores = -rule_scores
        rule_scores_scaled_arr = minmax_scale(rule_scores, feature_range=(
            self.min_value, self.max_value))
        rule_scores_scaled = pd.Series(
            rule_scores_scaled_arr, rule_scores.keys())
        return round(rule_scores_scaled).astype(int)
