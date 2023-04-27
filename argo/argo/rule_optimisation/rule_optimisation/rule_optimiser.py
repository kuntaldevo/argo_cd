"""Class for optimising rules"""
from hyperopt import hp, tpe, fmin
from hyperopt.pyll import scope
import numpy as np
import pandas as pd
from rule_application.argo_rule_applier import ArgoRuleApplier
from rules.rules import Rules
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from copy import deepcopy
import argo_utils.argo_utils as argo_utils


class RuleOptimiser:
    """
    Optimises a set of rules (given in the standard ARGO lambda expression 
    format).

    Attributes:
        opt_rule_strings (dict): The optimised rules stored in the standard 
            ARGO string format (values) and their names (keys).    
        rule_names_missing_features (list): Names of rules which use features 
            that are not present in the dataset (and therefore can't be 
            optimised or applied).
        rule_names_no_opt_conditions (list): Names of rules which have no 
            optimisable conditions (e.g. rules that only contain string-based 
            conditions).
        rule_names_zero_var_features (list): Names of rules which exclusively 
            contain zero variance features (based on `X`), so cannot be 
            optimised.
        rules (object): Class containing the optimised rules stored in the 
            standard ARGO string format. Methods from this class can be used to 
            convert the rules into other representations. See the rules module 
            for more information.
        opt_rule_performances (dict): The optimisation metric (values) 
            calculated for each optimised rule (keys).
        orig_rule_performances (dict): The optimisation metric (values) 
            calculated for each original rule (keys).
    """

    def __init__(self, rule_lambdas: dict, lambda_kwargs: dict,
                 opt_func: object, n_iter: int, show_progressbar=True):
        """
        Args:
            rule_lambdas (dict): Set of rules defined using the standard ARGO 
                lambda expression format (values) and their names (keys).
            lambda_kwargs (dict): For each rule (keys), a dictionary containing 
                the features used in the rule (keys) and the current values 
                (values).
            opt_func (object): The optimisation function used to calculate the 
                metric which is optimised for (e.g. F1 score).
            n_iter (int): The number of iterations that the optimiser should 
                perform.
            show_progressbar (bool, optional): If True, the optimisation 
                progress for each rule is shown. Defaults to True.
        """
        self.rule_lambdas = rule_lambdas.copy()
        self.lambda_kwargs = lambda_kwargs.copy()
        self.opt_func = opt_func
        self.n_iter = n_iter
        self.show_progressbar = show_progressbar
        self.opt_rule_strings = {}

    def fit(self, X: pd.DataFrame, y=None, sample_weight=None) -> dict:
        """
        Optimises a set of rules (given in the standard ARGO lambda expression 
        format).

        Args:
            X (pd.DataFrame): The feature set.
            y (pd.Series): The binary target column. Not required if optimising 
                rules on unlabelled data. Defaults to None.
            sample_weight (pd.Series, optional): Record-wise weights to apply. 
                Defaults to None.

        Returns:
            dict: Dictionary of optimised rules stored in the standard ARGO 
                string format (values) and their names (keys).
        """

        self.rules = Rules(rule_lambdas=self.rule_lambdas,
                           lambda_kwargs=self.lambda_kwargs)
        _ = self.rules.as_rule_strings(as_numpy=False)
        self.rule_names_missing_features, rule_features_in_X = self._return_rules_missing_features(
            rules=self.rules, X=X)
        if self.rule_names_missing_features:
            self.rules.filter_rules(exclude=self.rule_names_missing_features)
        X = X[rule_features_in_X]
        all_rule_features, self.rule_names_no_opt_conditions = self._return_all_optimisable_rule_features(
            lambda_kwargs=self.rules.lambda_kwargs, X=X)
        int_cols = self._return_int_cols(X=X)
        all_space_funcs = self._return_all_space_funcs(
            all_rule_features=all_rule_features, X=X, int_cols=int_cols)
        self.rule_names_zero_var_features = self._return_rules_with_zero_var_features(
            lambda_kwargs=self.rules.lambda_kwargs, all_space_funcs=all_space_funcs,
            rule_names_no_opt_conditions=self.rule_names_no_opt_conditions)
        optimisable_rules, non_optimisable_rules = self._return_optimisable_rules(
            rules=self.rules, rule_names_no_opt_conditions=self.rule_names_no_opt_conditions,
            rule_names_zero_var_features=self.rule_names_zero_var_features)
        if not optimisable_rules.rule_lambdas:
            raise Exception('There are no optimisable rules in the set')
        self.orig_rule_performances = self._return_rule_performances(
            rule_strings=optimisable_rules.rule_strings, X=X, y=y,
            sample_weight=sample_weight, opt_func=self.opt_func)
        opt_rule_strings = self._optimise_rules(rule_lambdas=optimisable_rules.rule_lambdas,
                                                lambda_kwargs=optimisable_rules.lambda_kwargs,
                                                X=X, y=y, sample_weight=sample_weight,
                                                int_cols=int_cols,
                                                all_space_funcs=all_space_funcs)
        self.opt_rule_performances = self._return_rule_performances(
            rule_strings=opt_rule_strings, X=X, y=y, sample_weight=sample_weight,
            opt_func=self.opt_func)
        self.opt_rule_strings, self.opt_rule_performances = self._return_orig_rule_if_better_perf(
            orig_rule_performances=self.orig_rule_performances,
            opt_rule_performances=self.opt_rule_performances,
            orig_rule_strings=optimisable_rules.rule_strings,
            opt_rule_strings=opt_rule_strings)
        if non_optimisable_rules.rule_strings:
            self.opt_rule_strings.update(non_optimisable_rules.rule_strings)
        self.rules = Rules(rule_strings=self.opt_rule_strings)
        return self.opt_rule_strings

    def apply(self, X: pd.DataFrame, y=None,
              sample_weight=None) -> pd.DataFrame:
        """
        Applies the optimised rules to a given dataset.

        Args:
            X (pd.DataFrame): The feature set.
            y (pd.Series, optional): The binary target column. Not required if 
                the rules were optimised on unlabelled data. Defaults to None.
            sample_weight (pd.Series, optional): Record-wise weights to apply. 
                Defaults to None.

        Returns:
            pd.DataFrame: The binary columns of the rules applied to the 
                dataset X.
        """
        ara = ArgoRuleApplier(rule_strings=self.opt_rule_strings,
                              opt_func=self.opt_func)
        X_rules = ara.apply(X=X, y=y, sample_weight=sample_weight)
        self.rule_descriptions = ara.rule_descriptions
        return X_rules

    def plot_performance_uplift(self, orig_rule_performances: dict,
                                opt_rule_performances: dict,
                                figsize=(20, 10)) -> sns.scatterplot:
        """
        Generates a scatterplot showing the performance of each rule before
        and after optimisation.

        Args:
            orig_rule_performances (dict): The performance metric of each rule
                prior to optimisation.
            opt_rule_performances (dict): The performance metric of each rule
                after optimisation.
            figsize (tuple, optional): The width and height of the scatterplot.
                Defaults to (20, 10).

        Returns:
            sns.scatterplot: Scatterplot showing the performance of each rule
                before and after optimisation.
        """
        performance_comp, _ = self._calculate_performance_comparison(orig_rule_performances=orig_rule_performances,
                                                                     opt_rule_performances=opt_rule_performances)
        plt.figure(figsize=figsize)
        sns.scatterplot(x=list(performance_comp.index),
                        y=performance_comp['OriginalRule'], color='blue', label='Original rule')
        sns.scatterplot(x=list(performance_comp.index),
                        y=performance_comp['OptimisedRule'], color='red', label='Optimised rule')
        plt.title(
            'Performance comparison of original rules vs optimised rules')
        plt.xticks(rotation=90)
        plt.ylabel('Performance (of the provided optimisation metric)')
        plt.show()

    def plot_performance_uplift_distribution(self,
                                             orig_rule_performances: dict,
                                             opt_rule_performances: dict,
                                             figsize=(8, 10)) -> sns.boxplot:
        """
        Generates a boxplot showing the distribution of performance uplifts
        (original rules vs optimised rules).

        Args:
            orig_rule_performances (dict): The performance metric of each rule
                prior to optimisation.
            opt_rule_performances (dict): The performance metric of each rule
                after optimisation.
            figsize (tuple, optional): The width and height of the boxplot.
                Defaults to (20, 10).

        Returns:
            sns.boxplot: Boxplot showing the distribution of performance 
            uplifts (original rules vs optimised rules).
        """

        _, performance_difference = self._calculate_performance_comparison(orig_rule_performances=orig_rule_performances,
                                                                           opt_rule_performances=opt_rule_performances)
        plt.figure(figsize=figsize)
        sns.boxplot(y=performance_difference)
        plt.title(
            'Distribution of performance uplift, original rules vs optimised rules')
        plt.xticks(rotation=90)
        plt.ylabel(
            'Performance uplift (of the provided optimisation metric)')
        plt.show()

    def _optimise_rules(self, rule_lambdas: dict, lambda_kwargs: dict,
                        X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series,
                        int_cols: list, all_space_funcs: dict) -> dict:
        """Optimises each rule in the set"""

        opt_rule_strings = {}
        for rule_name, rule_lambda in rule_lambdas.items():
            rule_lambda_kwargs = lambda_kwargs[rule_name]
            rule_features = list(rule_lambda_kwargs.keys())
            rule_space_funcs = self._return_rule_space_funcs(
                all_space_funcs=all_space_funcs, rule_features=rule_features)
            opt_thresholds = self._optimise_rule_thresholds(
                rule_lambda=rule_lambda, rule_space_funcs=rule_space_funcs, X_=X,
                y=y, sample_weight=sample_weight, opt_func=self.opt_func, n_iter=self.n_iter,
                show_progressbar=self.show_progressbar)
            opt_thresholds = self._convert_opt_int_values(
                opt_thresholds=opt_thresholds, int_cols=int_cols)
            opt_rule_strings[rule_name] = rule_lambda(**opt_thresholds)
        return opt_rule_strings

    @staticmethod
    def _return_int_cols(X: pd.DataFrame) -> list:
        """Returns the list of integer columns"""

        int_cols = X.select_dtypes(include=np.int).columns.tolist()
        float_cols = X.select_dtypes(include=np.float).columns.tolist()
        for float_col in float_cols:
            if abs(X[float_col] - X[float_col].round()).sum() == 0:
                int_cols.append(float_col)
        return int_cols

    @staticmethod
    def _return_all_optimisable_rule_features(lambda_kwargs: dict,
                                              X: pd.DataFrame) -> tuple:
        """
        Returns a list of all of the features used in each optimisable rule 
        within the set.
        """
        X_isna_means = X.isna().mean()
        cols_all_null = X_isna_means[X_isna_means == 1].index.tolist()
        all_rule_features = set()
        rule_names_no_opt_conditions = []
        for rule_name, lambda_kwarg in lambda_kwargs.items():
            if lambda_kwarg == {}:
                rule_names_no_opt_conditions.append(rule_name)
                continue
            rule_features = list(lambda_kwarg.keys())
            for feature in rule_features:
                if feature.split('%')[0] in cols_all_null:
                    rule_names_no_opt_conditions.append(rule_name)
                    break
                else:
                    all_rule_features.add(feature)
        if rule_names_no_opt_conditions:
            warnings.warn(
                f'Rules `{"`, `".join(rule_names_no_opt_conditions)}` have no optimisable conditions - unable to optimise these rules')
        all_rule_features = list(all_rule_features)
        return all_rule_features, rule_names_no_opt_conditions

    @staticmethod
    def _return_all_space_funcs(all_rule_features: list, X: pd.DataFrame,
                                int_cols: list) -> dict:
        """
        Returns a dictionary of the space function (used in the optimiser) for 
        each feature in the dataset
        """

        space_funcs = {}
        for feature in all_rule_features:
            # If features contains %, means that there's more than one
            # occurance of the feature in the rule. To get the column, we need
            # to get the string precending the % symbol.
            col = feature.split('%')[0]
            col_min, col_max = X[col].min(), X[col].max()
            # If column is zero variance (excl. nulls), then set the space
            # function to the minimum value
            if col_min == col_max:
                space_funcs[feature] = col_min
                continue
            if col in int_cols:
                space_func = scope.int(
                    hp.uniform(feature, col_min, col_max))
            else:
                space_func = hp.uniform(feature, col_min, col_max)
            space_funcs[feature] = space_func
        return space_funcs

    def _return_rules_with_zero_var_features(self, lambda_kwargs,
                                             all_space_funcs: dict,
                                             rule_names_no_opt_conditions: list) -> list:
        """
        Returns list of rule names that have all zero variance features, 
        so cannot be optimised
        """

        rule_names_zero_var_features = []
        for rule_name, rule_lambda_kwargs in lambda_kwargs.items():
            if rule_name in rule_names_no_opt_conditions:
                continue
            rule_features = list(rule_lambda_kwargs.keys())
            rule_space_funcs = self._return_rule_space_funcs(
                all_space_funcs=all_space_funcs, rule_features=rule_features)
            if all([isinstance(space_func, (int, float)) for space_func in rule_space_funcs.values()]):
                rule_names_zero_var_features.append(rule_name)
                continue
        if rule_names_zero_var_features:
            warnings.warn(
                f'Rules `{"`, `".join(rule_names_zero_var_features)}` have all zero variance features based on the dataset `X` - unable to optimise these rules')
        return rule_names_zero_var_features

    @staticmethod
    def _return_optimisable_rules(rules: Rules,
                                  rule_names_no_opt_conditions: list,
                                  rule_names_zero_var_features: list) -> tuple:
        """
        Copies the Rules class and filters out rules which cannot be 
        optimised from the original Rules class. Then filters to only those
        un-optimisable rules in the copied Rules class, and returns both
        """

        rule_names_to_exclude = rule_names_no_opt_conditions + rule_names_zero_var_features
        non_optimisable_rules = deepcopy(rules)
        rules.filter_rules(exclude=rule_names_to_exclude)
        non_optimisable_rules.filter_rules(
            include=rule_names_to_exclude)
        return rules, non_optimisable_rules

    @staticmethod
    def _return_rules_missing_features(rules: Rules, X: pd.DataFrame) -> list:
        """
        Returns the names of rules that contain features missing from `X`.
        """

        cols = X.columns
        rule_features = rules.get_rule_features()
        rule_names_missing_features = []
        rule_features_set = set()
        for rule_name, feature_set in rule_features.items():
            missing_features = [
                feature for feature in feature_set if feature not in cols]
            [rule_features_set.add(feature)
             for feature in feature_set if feature in cols]
            if missing_features:
                rule_names_missing_features.append(rule_name)
        if rule_names_missing_features:
            warnings.warn(
                f'Rules `{"`, `".join(rule_names_missing_features)}` use features that are missing from `X` - unable to optimise or apply these rules')
        return rule_names_missing_features, rule_features_set

    @staticmethod
    def _return_rule_performances(rule_strings: dict, X: pd.DataFrame,
                                  y: pd.Series, sample_weight: pd.Series,
                                  opt_func: object) -> dict:
        """
        Returns a dictionary of the calculated optimisation metric for 
        each rule.
        """
        ara = ArgoRuleApplier(rule_strings=rule_strings)
        X_rules = ara.apply(X=X)
        opt_metric_results = argo_utils.return_opt_func_perf(opt_func=opt_func,
                                                             y_preds=X_rules,
                                                             y_true=y,
                                                             sample_weight=sample_weight)
        rule_performances = dict(zip(X_rules.columns, opt_metric_results))
        rule_performances = dict(sorted(rule_performances.items(),
                                        key=lambda kv: -kv[1]))
        return rule_performances

    @staticmethod
    def _return_rule_space_funcs(all_space_funcs: dict,
                                 rule_features: list) -> dict:
        """
        Returns a dictionary of the space function for each feature in 
        the rule.
        """

        rule_space_funcs = dict((rule_feature, all_space_funcs[rule_feature])
                                for rule_feature in rule_features)
        return rule_space_funcs

    @staticmethod
    def _optimise_rule_thresholds(rule_lambda: object, rule_space_funcs: dict,
                                  X_: pd.DataFrame, y: pd.Series,
                                  sample_weight: pd.Series, opt_func: object,
                                  n_iter: int, show_progressbar: bool) -> dict:
        """Calculates the optimal rule thresholds"""

        def _objective(rule_space_funcs: dict) -> float:
            """
            Evaluates the optimisation metric for each
            iteration in the optimisation process.
            """
            # Bring X_ into local scope (for eval() function)
            X = X_
            rule_string = rule_lambda(**rule_space_funcs)
            y_pred = eval(rule_string)
            # If evaluated rule is pd.Series, replace pd.NA with False (since
            # pd.NA used in any condition returns pd.NA, not False as with
            # numpy)
            if isinstance(y_pred, pd.Series):
                y_pred = y_pred.fillna(False).astype(int)
            elif isinstance(y_pred, np.ndarray):
                y_pred = y_pred.astype(int)
            if y is not None:
                opt_metric = opt_func(y_true=y, y_pred=y_pred,
                                      sample_weight=sample_weight)
            else:
                opt_metric = opt_func(y_pred=y_pred)
            return -opt_metric

        opt_thresholds = fmin(
            fn=_objective,
            space=rule_space_funcs,
            algo=tpe.suggest,
            max_evals=n_iter,
            show_progressbar=show_progressbar,
            rstate=np.random.RandomState(0))

        # If rule_space_funcs contained constant values (due to min/max of
        # feature being equal in the dataset), then add those values back into
        # the optimised_thresholds dictionary
        if len(opt_thresholds) < len(rule_space_funcs):
            for feature, space_func in rule_space_funcs.items():
                if feature not in opt_thresholds.keys():
                    opt_thresholds[feature] = space_func
        return opt_thresholds

    @staticmethod
    def _return_orig_rule_if_better_perf(orig_rule_performances: dict,
                                         opt_rule_performances: dict,
                                         orig_rule_strings: dict,
                                         opt_rule_strings: dict) -> dict:
        """
        Overwrites the optimised rule string with the original if the original 
        is better performing. Also update the performance dictionary with the 
        original if this is the case.
        """

        for rule_name in opt_rule_strings.keys():
            if orig_rule_performances[rule_name] >= opt_rule_performances[rule_name]:
                opt_rule_strings[rule_name] = orig_rule_strings[rule_name]
                opt_rule_performances[rule_name] = orig_rule_performances[rule_name]
        return opt_rule_strings, opt_rule_performances

    @staticmethod
    def _convert_opt_int_values(opt_thresholds: dict, int_cols: list) -> dict:
        """
        Converts threshold values based on integer columns into integer 
        format.
        """

        for feature, value in opt_thresholds.items():
            col = feature.split('%')[0]
            if col in int_cols:
                opt_thresholds[feature] = int(value)
        return opt_thresholds

    @staticmethod
    def _calculate_performance_comparison(orig_rule_performances: dict,
                                          opt_rule_performances: dict) -> tuple:
        """
        Generates two dataframe - one showing the performance of the original 
        rules and the optimised rules, the other showing the difference in 
        performance per rule.
        """

        performance_comp = pd.concat([pd.Series(
            orig_rule_performances), pd.Series(opt_rule_performances)], axis=1)
        performance_comp.columns = ['OriginalRule', 'OptimisedRule']
        performance_difference = performance_comp['OptimisedRule'] - \
            performance_comp['OriginalRule']
        return performance_comp, performance_difference
