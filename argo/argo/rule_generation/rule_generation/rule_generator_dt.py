"""Class for generating rules using decision trees"""
import multiprocessing
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import argo_utils.argo_utils as argo_utils
from rule_application.argo_rule_applier import ArgoRuleApplier
from rules.rules import Rules
from datetime import date


class RuleGeneratorDT:
    """
    Generate rules by extracting the highest performing branches from a 
    tree ensemble model.

    Attributes:
        rule_descriptions (pd.Dataframe): A dataframe showing the logic of the 
            rules and their performance metrics on the fitted dataset.
        rule_descriptions_applied (pd.Dataframe): A dataframe showing the logic
            of the rules and their performance metrics on the applied dataset.
        rules (object): Class containing the rule stored in the standard ARGO 
            string format. Methods from this class can be used to convert the 
            rules into other representations. See the `rules` sub-package for 
            more information.
    """

    def __init__(self, opt_func: object, n_total_conditions: int,
                 tree_ensemble: object, precision_threshold=0,
                 num_cores=int(multiprocessing.cpu_count() / 2)):
        """
        Args:
            opt_func (object): A function/method which calculates the desired 
                optimisation metric (e.g. Fbeta score).
            n_total_conditions (int): The maximum number of conditions per 
                generated rule.
            tree_ensemble (object): Sklearn tree ensemble classifier object 
                used to generated rules.
            precision_threshold (float, optional): Precision threshold for the
                tree/branch to be used to create rules. If the overall 
                precision of the tree/branch is less than or equal to this
                value, it will not be used in rule generation. Defaults to 0.
            num_cores (int, optional): The number of cores to use when 
                iterating through the ensemble to generate rules. Defaults to 
                `cpu_count` / 2.
        """

        self.tree_ensemble = tree_ensemble
        self.tree_ensemble.max_depth = n_total_conditions
        self.tree_ensemble.random_state = 0
        self.opt_func = opt_func
        self.precision_threshold = precision_threshold
        self.num_cores = num_cores
        self.rule_descriptions = argo_utils.generate_empty_data_structures()[
            0]
        self.rule_descriptions_applied = argo_utils.generate_empty_data_structures()[
            0]
        self._rule_name_counter = 0
        today = date.today()
        self.today = today.strftime("%Y%m%d")

    def fit(self, X: pd.DataFrame, y: pd.Series,
            sample_weight=None) -> pd.DataFrame:
        """
        Generates rules by extracting the highest performing branches in a tree
        ensemble model.

        Args:
            X (pd.DataFrame): The feature set used for training the model.
            y (pd.Series): The target column.            
            sample_weight (pd.Series, optional): Record-wise weights to apply. 
                Defaults to None.

        Returns:
            pd.DataFrame: The binary columns of the rules on the fitted 
                dataset.                   
        """

        columns_int, columns_cat, _ = argo_utils.return_columns_types(X)
        trained_tree_ensemble = self._train_ensemble(
            X=X, y=y, tree_ensemble=self.tree_ensemble, sample_weight=sample_weight)
        self.rule_descriptions, X_rules = self._extract_rules_from_ensemble(X=X, y=y,
                                                                            num_cores=self.num_cores,
                                                                            tree_ensemble=trained_tree_ensemble,
                                                                            columns_int=columns_int,
                                                                            columns_cat=columns_cat,
                                                                            sample_weight=sample_weight)
        rule_strings = self.rule_descriptions['Logic'].to_dict()
        self.rules = Rules(rule_strings=rule_strings)
        return X_rules

    def apply(self, X: pd.DataFrame, y=None,
              sample_weight=None) -> pd.DataFrame:
        """
        Applies the generated rules to another dataset, X.

        Args:
            X (pd.DataFrame): The features set on which to apply the rules.
            y (pd.Series): The target column. Include if you wish to calculate 
                rule performance metrics on the dataset. Defaults to None.
            sample_weight (pd.Series, optional): Record-wise weights to apply.
                Defaults to None.

        Returns:
            pd.Dataframe: The binary columns of the rules applied to the
                dataset `X`.
        """

        ra = ArgoRuleApplier(
            rule_strings=self.rules.rule_strings, opt_func=self.opt_func)
        X_rules_applied = ra.apply(
            X=X, y=y, sample_weight=sample_weight)
        self.rule_descriptions_applied = ra.rule_descriptions
        return X_rules_applied

    def _drop_low_prec_trees_return_rules(self, X: pd.DataFrame, y: pd.Series,
                                          decision_tree: object,
                                          sample_weight: pd.Series,
                                          columns_int: list,
                                          columns_cat: list) -> tuple:
        """
        Method for extracting rules from a decision tree(if the precision of 
        the tree is above the threshold).
        """

        y_pred = decision_tree.predict(X)
        tree_prec = argo_utils.return_binary_pred_perf_of_set_numpy(
            y_true=y, y_preds=y_pred, y_preds_columns=['DT Prediction'],
            sample_weight=sample_weight, opt_func=None).iloc[0]['Precision']
        if tree_prec <= self.precision_threshold:
            return set()
        else:
            return self._extract_rules_from_tree(X=X, decision_tree=decision_tree,
                                                 precision_threshold=self.precision_threshold,
                                                 columns_int=columns_int,
                                                 columns_cat=columns_cat)

    def _extract_rules_from_ensemble(self, X: pd.DataFrame, y: pd.Series,
                                     tree_ensemble: object, num_cores: int,
                                     sample_weight: pd.Series,
                                     columns_int: list,
                                     columns_cat: list) -> tuple:
        """
        Method for returning all of the rules from the ensemble tree-based 
        model.
        """

        with Parallel(n_jobs=num_cores) as parallel:
            list_of_rule_string_sets = parallel(delayed(self._drop_low_prec_trees_return_rules)(X, y, decision_tree,
                                                                                                sample_weight, columns_int,
                                                                                                columns_cat) for decision_tree in tree_ensemble.estimators_)
        rule_strings_set = set().union(*list_of_rule_string_sets)
        rule_strings = dict((self._generate_rule_name(), rule_string)
                            for rule_string in rule_strings_set)
        applier = ArgoRuleApplier(rule_strings, opt_func=self.opt_func)
        X_rules = applier.apply(X, y, sample_weight)
        rule_descriptions = applier.rule_descriptions
        return rule_descriptions, X_rules

    def _extract_rules_from_tree(self, X: pd.DataFrame, decision_tree: object,
                                 precision_threshold: float, columns_int: list,
                                 columns_cat: list) -> tuple:
        """
        Method for returning the rules of all the leaves of the decision 
        tree passed.
        """

        left = decision_tree.tree_.children_left
        right = decision_tree.tree_.children_right
        threshold = decision_tree.tree_.threshold
        leaf_node_splits = decision_tree.tree_.value
        features = decision_tree.tree_.feature
        columns = X.columns
        leaf_nodes = np.argwhere(left == -1)[:, 0]

        def recurse_rule(left: np.ndarray, right: np.ndarray, child: int,
                         rule=None):
            """
            IDs each leaf node, then iterates through up to the parent, noting 
            the conditions at each node.
            """
            if rule is None:
                rule = []
            if child in left:
                parent = np.where(left == child)[0].item()
                split = '<='
            else:
                parent = np.where(right == child)[0].item()
                split = '>'
            rule.append((columns[features[parent]], split,
                         round(threshold[parent], 5)))
            if parent == 0:
                rule.reverse()
                return rule
            else:
                return recurse_rule(left, right, parent, rule)

        if not leaf_nodes.any():
            return None, None
        rule_strings_set = set()
        for child in leaf_nodes:
            child_split = leaf_node_splits[child][0]
            child_precision = child_split[1]/np.sum(child_split)
            if child_precision <= precision_threshold:
                continue
            branch_conditions = recurse_rule(left, right, child)
            branch_conditions = argo_utils.clean_dup_features_from_conditions(
                branch_conditions)
            rule_logic = argo_utils.convert_conditions_to_argo_string(
                list_of_conditions=branch_conditions, columns_int=columns_int,
                columns_cat=columns_cat)
            rule_strings_set.add(rule_logic)
        return rule_strings_set

    def _generate_rule_name(self) -> str:
        """Generates rule name"""

        rule_name = f'RGDT_Rule_{self.today}_{self._rule_name_counter}'
        self._rule_name_counter += 1
        return rule_name

    @staticmethod
    def _train_ensemble(X: pd.DataFrame, y: pd.Series, tree_ensemble: object,
                        sample_weight: pd.Series) -> object:
        """Method for running ML model"""

        tree_ensemble.fit(X=X, y=y, sample_weight=sample_weight)
        return tree_ensemble
