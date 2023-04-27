"""Class for applying rules in the standard ARGO string format"""
import pandas as pd
import numpy as np
import argo_utils.argo_utils as argo_utils


class ArgoRuleApplier:
    """
    Applies rules (stored in the standard ARGO string format) to a dataset.

    Attributes:
        rule_descriptions (pd.DataFrame): Contains the logic of the rules and 
            their performance metrics as applied to the dataset.        
    """

    def __init__(self, rule_strings: dict, opt_func=None):
        """
        Args:
            rule_strings (dict): Set of rules defined using the standard ARGO 
                string format (values) and their names (keys).
            opt_func (object, optional): A function/method which calculates a 
                custom metric (e.g. Fbeta score) for each rule. Defaults to 
                None.
        """

        self.opt_func = opt_func
        self.rule_strings = rule_strings
        self.unapplied_rule_names = []

    def apply(self, X: pd.DataFrame, y=None,
              sample_weight=None) -> pd.DataFrame:
        """
        Applies the set of rules to a dataset, `X`. If `y` is provided, the 
        performance metrics for each rule will also be calculated.

        Args:
            X (pd.DataFrame: The feature set on which the 
                rules should be applied.            
            y (pd.DataFrame, optional): The target column. 
                Defaults to None.        
            sample_weight (pd.Series, optional): Record-wise weights 
                to apply. Defaults to None.

        Returns:
            pd.DataFrame: Contains the binary columns for each rule, which 
                dictate whether the rule has triggered (i.e. value is 1) for a 
                particular record.
        """

        if not isinstance(X, pd.DataFrame):
            raise TypeError('X must be a Pandas DataFrame')
        X_rules = self._get_X_rules(X)
        rule_strings_list = list(self.rule_strings.values())
        # If unlabelled data and opt_func provided, or labelled data,
        # calculate rule_descriptions
        if (y is None and self.opt_func is not None) or (y is not None):
            rule_descriptions = argo_utils.return_rule_descriptions_from_X_rules(X_rules=X_rules,
                                                                                 X_rules_cols=X_rules.columns,
                                                                                 y_true=y,
                                                                                 sample_weight=sample_weight,
                                                                                 opt_func=self.opt_func)
            rule_descriptions['Logic'] = rule_strings_list
            rule_descriptions['nConditions'] = list(map(
                argo_utils.count_rule_conditions, rule_strings_list))
            self.rule_descriptions, X_rules = argo_utils.sort_rule_dfs_by_opt_metric(
                rule_descriptions, X_rules)
        return X_rules

    def _get_X_rules(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Returns the binary columns of the list of rules applied to the 
        dataset `X`.
        """

        X_rules_list = []
        for rule_name, rule_string in self.rule_strings.items():
            try:
                X_rule = eval(rule_string)
            except KeyError as e:
                raise KeyError(
                    f'Feature {e} in rule `{rule_name}` not found in `X`')
            if isinstance(X_rule, pd.Series):
                X_rule = X_rule.fillna(False).astype(int)
            elif isinstance(X_rule, np.ndarray):
                X_rule = X_rule.astype(int)
            X_rules_list.append(X_rule)
        if isinstance(X_rules_list[0], np.ndarray):
            X_rules = pd.DataFrame(np.asarray(X_rules_list)).T
        else:
            X_rules = pd.concat(X_rules_list, axis=1, sort=False)
        X_rules.columns = list(self.rule_strings.keys())
        X_rules.index = X.index
        return X_rules
