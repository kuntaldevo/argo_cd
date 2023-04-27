"""Class for applying rules using the `sim_ll` column"""
import pandas as pd
import numpy as np
import argo_utils.argo_utils as argo_utils
import warnings


class SimRuleApplier:

    """
    Applies a set of system rules to a dataset (which are applied by
    flattening the `sim_ll` column).

    Attributes:
        rule_descriptions (pd.Dataframe): Contains the logic of the rules and 
            heir performance metrics as applied to the dataset.
        rules_not_in_sim_ll (list): List of rule names that were provided in 
            the `rules` class constructor parameter but could not be found in 
            the `sim_ll` column.
    """

    def __init__(self, opt_func=None, sim_ll_column='sim_ll', rules=None):
        """
        Args:
            opt_func (object, optional): A function/method which calculates 
                the desired optimisation metric (e.g. Fbeta score). Defaults 
                to None.
            sim_ll_column (str, optional): The name of the column containing 
                the `sim_ll` field. Defaults to 'sim_ll'.
            rules (list, optional): If only a subset of system rules are 
                required, specify their names in a list here. If None, all 
                rules found in `sim_ll` are returned. Defaults to None.
        """
        self.opt_func = opt_func
        self.sim_ll_column = sim_ll_column
        self.rules = rules

    def apply(self, X: pd.DataFrame, y=None,
              sample_weight=None) -> pd.DataFrame:
        """
        Applies a set of system rules to the dataset `X` (using the `sim_ll`
        column). If `y` is provided, the performance metrics for each rule will
        also be calculated.

        Args:
            X (pd.Dataframe): The dataset containing the `sim_ll` column.
            y (pd.Series, optional): The target column. Defaults to None.                    
            sample_weight (pd.Series, optional): Record-wise weights to apply. 
                Defaults to None.

        Returns:
            pd.DataFrame: Contains the binary columns for each rule, which 
                dictate whether the rule has triggered (i.e. value is 1) for a 
                particular record.
        """

        if self.sim_ll_column not in X.columns:
            raise Exception(
                f'The sim_ll_column given (`{self.sim_ll_column}`) is not in `X`.')
        sim_ll_flattened = argo_utils.flatten_stringified_json_column(
            X[self.sim_ll_column])
        # Convert to binary columns
        X_rules = (~sim_ll_flattened.isna()).replace({True: 1, False: 0})
        if self.rules is not None:
            X_rules = self._filter_rules(X_rules=X_rules)
        # If unlabelled data and opt_func provided, or labelled data,
        # calculate rule_descriptions
        if (y is None and self.opt_func is not None) or (y is not None):
            rule_descriptions = argo_utils.return_rule_descriptions_from_X_rules(X_rules=X_rules,
                                                                                 X_rules_cols=X_rules.columns,
                                                                                 y_true=y,
                                                                                 sample_weight=sample_weight,
                                                                                 opt_func=self.opt_func)
            self.rule_descriptions, X_rules = argo_utils.sort_rule_dfs_by_opt_metric(
                rule_descriptions, X_rules)
        return X_rules

    def _filter_rules(self, X_rules: pd.DataFrame) -> pd.DataFrame:
        """
        Filters the binary columns based on the rule names provided in 
        `rules`
        """

        self.rules_not_in_sim_ll = [
            rule for rule in self.rules if rule not in X_rules.columns]
        if self.rules_not_in_sim_ll:
            warnings.warn(
                f'Rules `{"`, `".join(self.rules_not_in_sim_ll)}` not found in `{self.sim_ll_column}` - unable to apply these rules.')
        X_rules = X_rules[[
            rule for rule in self.rules if rule not in self.rules_not_in_sim_ll]]
        return X_rules
