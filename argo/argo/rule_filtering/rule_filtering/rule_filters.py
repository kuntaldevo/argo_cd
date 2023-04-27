"""Classes for filtering rules"""
import numpy as np
import pandas as pd
from correlation_reduction.correlation_reduction_methods import AgglomerativeClusteringFeatureReduction
import argo_utils.argo_utils as argo_utils
import seaborn as sns
import matplotlib.pyplot as plt
import warnings


class FilterRules:
    """
    Filter rules based on performance metrics.

    Attributes:
        rules_to_keep (list): List of rules which remain after the filters have
            been applied.
    """

    def __init__(self, filters: dict, rule_descriptions=None, opt_func=None):
        """
        Args:
            filters (dict): Gives the filtering metric(keys) and the filtering
                conditions(values). The filtering conditions are another
                dictionary containing the keys 'Operator' (which specifies the
                filter operator) and 'Value' (which specifies the value to
                filter by).
            rule_descriptions (pd.DataFrame, optional): The standard
                performance metrics dataframe associated with the rules (if
                available). If not given, it will be calculated from `X_rules`.
                Defaults to None.
            opt_func (object, optional): The custom method/function to be
                applied to the rules(e.g. Fbeta score) if rule_descriptions is
                not given. Use the filtering metric key 'OptMetric' in the
                filters parameter if you need to filter by this metric.
                Defaults to None.
        """

        self.filters = filters
        self.rule_descriptions = rule_descriptions
        self.opt_func = opt_func

    def fit(self, X_rules: pd.DataFrame, y=None, sample_weight=None) -> None:
        """
        Calculates the rules remaining after the filters have been applied.

        Args:
            X_rules (pd.DataFrame): The binary columns of the rules applied to
                a dataset.
            y (pd.Series, optional): The binary target column. Not required if
                `rule_descriptions` is given. Defaults to None.
            sample_weight (pd.Series, optional): Row-wise weights to apply.
                Defaults to None.
        """

        if self.rule_descriptions is None:
            if self.opt_func is None and 'OptMetric' in self.filters.keys():
                raise ValueError(
                    'Must provide `opt_func` when `rule_descriptions` is None and "OptMetric" is included in filters.')
            self.rule_descriptions = argo_utils.return_rule_descriptions_from_X_rules(X_rules=X_rules,
                                                                                      X_rules_cols=X_rules.columns,
                                                                                      y_true=y,
                                                                                      sample_weight=sample_weight,
                                                                                      opt_func=self.opt_func)
        self.rules_to_keep = self._iterate_rule_descriptions(
            rule_descriptions=self.rule_descriptions, filters=self.filters)

    def transform(self, X_rules: pd.DataFrame) -> pd.DataFrame:
        """
        Removes the filtered rules.

        Args:
            X_rules (pd.DataFrame): The binary columns of the rules applied
                to a dataset.

        Returns:
            pd.DataFrame: The filtered rule binary columns.
        """

        X_rules = X_rules[self.rules_to_keep]
        self.rule_descriptions = self.rule_descriptions.loc[self.rules_to_keep]
        return X_rules

    def fit_transform(self, X_rules: pd.DataFrame, y=None,
                      sample_weight=None) -> pd.DataFrame:
        """
        Calculates the rules remaining after the filters have been applied,
        then removes the filtered rules.

        Args:
            X_rules (pd.DataFrame): The binary columns of the rules applied to
                a dataset.
            y (pd.Series, optional): The binary target column. Not required if
                `rule_descriptions` is given. Defaults to None.
            sample_weight (pd.Series, optional): Row-wise weights to apply.
                Defaults to None.

        Returns:
            pd.DataFrame: The filtered rule binary columns.
        """

        self.fit(X_rules=X_rules, y=y, sample_weight=sample_weight)
        X_rules = self.transform(X_rules=X_rules)
        return X_rules

    @staticmethod
    def _iterate_rule_descriptions(rule_descriptions: pd.DataFrame,
                                   filters: dict) -> list:
        """
        Iterates through rule_descriptions and applies filters, returning
        the rules which meet the filter requirements
        """

        for metric, operator_value in filters.items():
            if metric not in rule_descriptions.columns:
                raise ValueError(
                    f'{metric} is not in the rule_descriptions dataframe')
            operator = operator_value['Operator']
            value = operator_value['Value']
            mask = eval(f'rule_descriptions["{metric}"] {operator} {value}')
            rule_descriptions = rule_descriptions[mask]
        rules_to_keep = rule_descriptions.index.tolist()
        if not rules_to_keep:
            warnings.warn('No rules remaining after filtering')
        return rules_to_keep


class GreedyFilter:
    """
    Sorts rules by a given metric, calculates the combined performance of the
    top n rules, then filters to the rules which give the best performance.

    Attributes:
        rules_to_keep(list): List of rules which give the best combined
            performance.
    """

    def __init__(self, opt_func, rule_descriptions=None,
                 sorting_col='Precision', show_plots=True):
        """
        Args:
            opt_func (object): The method/function used to calculate the
                performance of the top n rules(e.g. Fbeta score).
            rule_descriptions (pd.DataFrame, optional): The standard
                performance metrics dataframe associated with the rules (if
                available). If not given, it will be calculated from `X_rules`.
                Defaults to None.
            sorting_col (str, optional): Specifies the column within
                `rule_descriptions` to sort the rules by. Defaults to
                'Precision'.
            show_plots (bool, optional): If True, the combined performance of
                the top n rules is plotted. Defaults to True.
        """

        self.rule_descriptions = rule_descriptions
        self.opt_func = opt_func
        self.sorting_col = sorting_col
        self.show_plots = show_plots

    def fit(self, X_rules: pd.DataFrame, y=pd.Series,
            sample_weight=None) -> None:
        """
        Sorts rules by a given metric, calculates the combined performance of
        the top n rules, then calculates the rules which give the best combined
        performance.

        Args:
            X_rules (pd.DataFrame): The binary columns of the rules applied to
                a dataset.
            y (pd.Series): The binary target column.
            sample_weight (pd.Series, optional): Row-wise weights to apply.
                Defaults to None.
        """

        if self.rule_descriptions is None:
            self.rule_descriptions = argo_utils.return_binary_pred_perf_of_set_numpy(
                y_true=y, y_preds=X_rules, y_preds_columns=X_rules.columns,
                sample_weight=sample_weight, opt_func=self.opt_func)
        self.rule_descriptions.sort_values(
            self.sorting_col, ascending=False, inplace=True)
        self.top_n_rule_descriptions = self._return_performance_top_n(
            self.rule_descriptions, X_rules, y, sample_weight, self.opt_func)
        self.rules_to_keep = self._return_top_rules_by_opt_func(
            self.top_n_rule_descriptions, self.rule_descriptions)
        if self.show_plots:
            self._plot_performance(self.top_n_rule_descriptions[[
                                   'Precision', 'Recall']], 'Precision/Recall performance of the top n rules (by precision)')
            self._plot_performance(
                self.top_n_rule_descriptions['OptMetric'].to_frame(), 'OptMetric performance of the top n rules')

    def transform(self, X_rules: pd.DataFrame) -> pd.DataFrame:
        """
        Reduces the rule set by keeping the rules which give the best combined
        performance.

        Args:
            X_rules (pd.DataFrame): The binary columns of the rules applied to
                a dataset.

        Returns:
            pd.DataFrame: The binary columns of the rules which give the best
                combined performance.
        """

        X_rules = X_rules[self.rules_to_keep]
        self.rule_descriptions = self.rule_descriptions.loc[self.rules_to_keep]
        return X_rules

    def fit_transform(self, X_rules: pd.DataFrame, y: pd.Series,
                      sample_weight=None) -> pd.DataFrame:
        """
        Sorts rules by a given metric, calculates the combined performance of
        the top n rules, then keeps only the rules which give the best combined
        performance.

        Args:
            X_rules (pd.DataFrame): The binary columns of the rules applied to
                a dataset.
            y (pd.Series): The binary target column.
            sample_weight (pd.Series, optional): Row-wise weights to apply.
                Defaults to None.

        Returns:
            pd.DataFrame: The binary columns of the rules which give the best
                combined performance.
        """

        self.fit(X_rules=X_rules, y=y, sample_weight=sample_weight)
        X_rules = self.transform(X_rules=X_rules)
        return X_rules

    @staticmethod
    def _return_performance_top_n(rule_descriptions: pd.DataFrame,
                                  X_rules: pd.DataFrame, y: pd.Series,
                                  sample_weight: pd.Series,
                                  opt_func: object) -> pd.DataFrame:
        """
        Sorts rules by a given metric, calculates the combined performance
        of the top n rules
        """

        top_n_rule_descriptions_list = []
        X_rules = X_rules.reindex(rule_descriptions.index, axis=1)
        for n in range(1, rule_descriptions.shape[0] + 1):
            top_n_X_rules = X_rules.iloc[:, :n]
            top_n_combined = np.bitwise_or.reduce(top_n_X_rules.values, axis=1)
            top_n_rule_descriptions_list.append(argo_utils.return_binary_pred_perf_of_set_numpy(
                y_true=y, y_preds=top_n_combined, y_preds_columns=[n],
                sample_weight=sample_weight, opt_func=opt_func))
        top_n_rule_descriptions = pd.concat(
            top_n_rule_descriptions_list, axis=0)
        top_n_rule_descriptions.index.rename('Top n rules', inplace=True)
        return top_n_rule_descriptions

    @staticmethod
    def _return_top_rules_by_opt_func(top_n_rule_descriptions: pd.DataFrame,
                                      rule_descriptions: pd.DataFrame) -> list:
        """Returns rules which give the top combined performance"""

        idx_max_perf_func = top_n_rule_descriptions['OptMetric'].idxmax()
        rules_to_keep = rule_descriptions.index[:idx_max_perf_func].tolist()
        return rules_to_keep

    @staticmethod
    def _plot_performance(data: pd.DataFrame, title: str) -> sns.lineplot:
        """Creates seaborn lineplot"""

        plt.figure(figsize=(20, 10))
        sns.lineplot(data=data)
        plt.title(title)
        plt.show()


class FilterCorrelatedRules:
    """
    Filters correlated rules based on a correlation reduction class (see the
    `correlation_reduction` sub-package).
    """

    def __init__(self, correlation_reduction_class: object,
                 rule_descriptions=None):
        """
        Args:
            correlation_reduction_class (object): Instatiated class from the
                `correlation_reduction` sub-package.
            rule_descriptions (pd.DataFrame, optional): The standard
                performance metrics dataframe associated with the rules(if
                available). Defaults to None.
        """

        self.correlation_reduction_class = correlation_reduction_class
        self.rule_descriptions = rule_descriptions

    def fit(self, X_rules: pd.DataFrame, **kwargs) -> None:
        """
        Calculates the uncorrelated rules(using the correlation reduction
        class).

        Args:
            X_rules (pd.DataFrame): The binary columns of the rules applied to
                a dataset.
            **kwargs (dict): Any keyword arguments to pass to the correlation
                reduction class's `.fit()` method
        """
        self.correlation_reduction_class.fit(X=X_rules, **kwargs)
        self.rules_to_keep = self.correlation_reduction_class.columns_to_keep

    def transform(self, X_rules: pd.DataFrame) -> pd.DataFrame:
        """
        Keeps only the uncorrelated rules in `X_rules` and `rule_descriptions`.

        Args:
            X_rules (pd.DataFrame): The binary columns of the rules applied
                to a dataset.

        Returns:
            pd.DataFrame: The binary columns of the uncorrelated rules.
        """
        X_rules = X_rules[self.correlation_reduction_class.columns_to_keep]
        self.rule_descriptions = self.rule_descriptions.loc[
            self.correlation_reduction_class.columns_to_keep]
        return X_rules

    def fit_transform(self, X_rules: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculates the uncorrelated rules(using the correlation reduction
        class) then keeps only these uncorrelated rules in `X_rules` and
        `rule_descriptions`.

        Args:
            X_rules (pd.DataFrame): The binary columns of the rules applied to
                a dataset.
            **kwargs (dict): Any keyword arguments to pass to the correlation
                reduction class's `.fit()` method.

        Returns:
            pd.DataFrame: The binary columns of the uncorrelated rules.
        """
        self.fit(X_rules=X_rules, **kwargs)
        return self.transform(X_rules=X_rules)
