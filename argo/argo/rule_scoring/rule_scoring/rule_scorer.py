"""Class for scoring rules and scaling the scores"""
import pandas as pd


class RuleScorer:
    """
    Generates rule scores using the rule binary columns and the target column.

    Attributes:
        rule_scores (dict): Contains the generated score (values) for each 
            rule (keys).
    """

    def __init__(self, scoring_class: object, scaling_class: object):
        """
        Args:
            scoring_class (object): The instantiated scoring class - this 
                defines the method for generating the scores. Scoring classes 
                are available in the `rule_scoring_methods` module.
            scaling_class (object): The instantiated scaling class - this 
                defines the method for scaling the raw scores from the scoring 
                class. Scaling classes are available in the 
                `rule_score_scalers` module.
        """

        self.scoring_class = scoring_class
        self.scaling_class = scaling_class

    def fit(self, X_rules: pd.DataFrame, y: pd.Series,
            sample_weight=None) -> None:
        """
        Generates rule scores using the rule binary columns and the binary 
        target column.

        Args:
            X_rules (pd.DataFrame): The rule binary columns.
            y (pd.Series): The binary target column.
            sample_weight (pd.Series, optional): Row-wise weights to apply in 
                the `scoring_class`. Defaults to None.
        """

        self.rule_scores = self.scoring_class.fit(
            X_rules=X_rules, y=y, sample_weight=sample_weight)
        self.rule_scores = self.scaling_class.fit(rule_scores=self.rule_scores)

    def transform(self, X_rules: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the rule binary columns to show the generated scores applied
        to the dataset (i.e. replaces the 1 in `X_rules` with the generated 
        score).

        Args:
            X_rules (pd.DataFrame): The rule binary columns.

        Returns:
            pd.DataFrame: Shows the generated scores applied to the dataset 
                (i.e. replaces the 1 in `X_rules` with the generated score).
        """

        X_scores = self.rule_scores * X_rules
        return X_scores

    def fit_transform(self, X_rules: pd.DataFrame, y: pd.Series,
                      sample_weight=None) -> pd.DataFrame:
        """
        Generates rule scores using the rule binary columns and the binary 
        target column, then transforms the rule binary columns to show the 
        generated scores applied to the dataset (i.e. replaces the 1 in 
        `X_rules` with the generated score).

        Args:
            X_rules (pd.DataFrame): The rule binary columns.
            y (pd.Series): The binary target column.
            sample_weight (pd.Series, optional): Row-wise weights to apply in 
                the `scoring_class`. Defaults to None.

        Returns:
            pd.DataFrame: Shows the generated scores applied to the dataset 
                (i.e. replaces the 1 in `X_rules` with the generated score).
        """

        self.fit(X_rules=X_rules, y=y, sample_weight=sample_weight)
        X_scores = self.transform(X_rules=X_rules)
        return X_scores
