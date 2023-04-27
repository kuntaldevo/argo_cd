"""Class for generating rules by optimisation"""
import pandas as pd
import numpy as np
import math
from itertools import combinations
import argo_utils.argo_utils as argo_utils
from rule_application.argo_rule_applier import ArgoRuleApplier
from correlation_reduction.correlation_reduction_methods import AgglomerativeClusteringFeatureReduction
from correlation_reduction.similarity_functions import CosineSimilarity
from rule_optimisation.optimisation_functions import FScore
from rules.rules import Rules
from datetime import date
f1 = FScore(1)


class RuleGeneratorOpt:

    """
    Generate rules by optimising the thresholds of single features and 
    combining these one condition rules with AND conditions to create more 
    complex rules.

    Attributes:
        rule_descriptions (pd.Dataframe): A dataframe showing the logic of the 
            rules and their performance metrics on the fitted dataset.
        rule_descriptions_applied (pd.Dataframe): A dataframe showing the logic
            of the rules and their performance metrics on the applied dataset.
        rules (object): Class containing the rule stored in the standard ARGO 
            string format. Methods from this class can be used to convert to 
            other representations. See the `rules` sub-package for more 
            information.
    """

    def __init__(self, opt_func: object, n_total_conditions: int,
                 num_rules_keep: int, n_points=10, ratio_window=2,
                 one_cond_rule_opt_func=f1.fit, remove_corr_rules=True):
        """
        Args:
            opt_func (object): A function/method which calculates the desired 
                optimisation metric (e.g. Fbeta score). Note that the module 
                will assume higher values correspond to better performing 
                rules.
            n_total_conditions (int): The maximum number of conditions per 
                generated rule.
            num_rules_keep (int): The top number of rules (by Fbeta score) to 
                keep at the end of each stage of rule combination. Reducing 
                this number will improve the runtime, but may result
                in useful rules being removed.
            n_points (int, optional): Number of points to split a numeric 
                feature's range into when generating the numeric one 
                condition rules. A larger number will result in better 
                optimised one condition rules, but will take longer to 
                calculate. Defaults to 10.
            ratio_window (int, optional): Factor which determines the 
                optimisation range for numeric features (e.g. if a numeric 
                field has range of 1 to 11 and ratio_window = 3, the 
                optimisation range for the <= operator will be from 1 to 
                (11-1)/3 = 3.33; the optimisation range for the >= operator 
                will be from 11-((11-1)/3)=7.67 to 11). A larger number 
                (greater than 1) will result in a smaller range being used for 
                optimisation of one condition rules; set to 1 if you want to 
                optimise the one condition rules across the full range of the 
                numeric feature. Defaults to 2.
            one_cond_rule_opt_func (object, optional): The optimisation 
                function used for one condition rules. Note that the module 
                will assume higher values correspond to better performing 
                rules. Defaults to the method used for calculating the F1 
                score.
            remove_corr_rules (bool, optional): Dictates whether correlated 
                rules should be removed at the end of each pairwise 
                combination. Defaults to True.
        """

        self.opt_func = opt_func
        self.n_total_conditions = n_total_conditions
        self.num_rules_keep = num_rules_keep
        self.n_points = n_points
        self.ratio_window = ratio_window
        self.one_cond_rule_opt_func = one_cond_rule_opt_func
        self.remove_corr_rules = remove_corr_rules
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
        Generate rules by optimising the thresholds of single features and 
        combining these one condition rules with AND conditions to create more 
        complex rules.

        Args:
            X (pd.DataFrame): The feature set used for training the model.
            y (pd.Series): The target column.            
            sample_weight (pd.Series, optional): Record-wise weights to apply. 
                Defaults to None.

        Returns:
            pd.DataFrame: The binary columns of the rules on the fitted 
                dataset.
        """

        rule_descriptions, X_rules = argo_utils.generate_empty_data_structures()
        columns_int, columns_cat, columns_float = argo_utils.return_columns_types(
            X)
        columns_num = [
            col for col in columns_int if col not in columns_cat] + columns_float
        if columns_num:
            rule_descriptions_num, X_rules_num = self._generate_numeric_one_condition_rules(
                X, y, columns_num, columns_int, sample_weight)
            rule_descriptions, X_rules = argo_utils.combine_rule_dfs(
                rule_descriptions_num, X_rules_num, rule_descriptions, X_rules)
        if columns_cat:
            rule_descriptions_cat, X_rules_cat = self._generate_categorical_one_condition_rules(
                X, y, columns_cat, sample_weight)
            rule_descriptions, X_rules = argo_utils.combine_rule_dfs(
                rule_descriptions_cat, X_rules_cat, rule_descriptions, X_rules)
        self.rule_descriptions, X_rules = self._generate_n_order_pairwise_rules(
            rule_descriptions, X_rules, y, self.remove_corr_rules, sample_weight)
        rule_strings = self.rule_descriptions['Logic'].to_dict()
        self.rules = Rules(rule_strings=rule_strings)
        return X_rules

    def apply(self, X: pd.DataFrame, y=None,
              sample_weight=None) -> pd.DataFrame:
        """
        Applies the generated rules to another dataset, `X`.

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

        ara = ArgoRuleApplier(rule_strings=self.rules.rule_strings,
                              opt_func=self.opt_func)
        X_rules_applied = ara.apply(
            X=X, y=y, sample_weight=sample_weight)
        self.rule_descriptions_applied = ara.rule_descriptions
        return X_rules_applied

    def _generate_numeric_one_condition_rules(self, X: pd.DataFrame,
                                              y: pd.Series,
                                              columns_num: list,
                                              columns_int: list,
                                              sample_weight: pd.Series) -> tuple:
        """
        Optimises the threshold of each numeric column based on the Fbeta 
        score
        """

        rule_strings = {}
        rule_descriptions, X_rules = argo_utils.generate_empty_data_structures()
        cols_and_operators = list(
            zip(columns_num * 2, [">="] * len(columns_num) + ["<="] * len(columns_num)))
        for column, operator in cols_and_operators:
            X_col = X[column].values
            # if X_col.std() == 0:
            if np.std(X_col) == 0:
                continue
            x_min, x_max = self._set_iteration_range(
                X_col=X_col, column=column, operator=operator, n_points=self.n_points,
                ratio_window=self.ratio_window, columns_int=columns_int)
            x_iter = self._set_iteration_array(
                column, columns_int, x_min, x_max, self.n_points, self._round_to_n_sf)
            # Optimise threshold using self.one_cond_rule_beta
            opt_metric_iter = self._calculate_opt_metric_across_range(
                x_iter=x_iter, operator=operator, X_col=X_col, y=y,
                opt_func=self.one_cond_rule_opt_func, sample_weight=sample_weight)
            x_max_opt_metric = self._return_x_of_max_opt_metric(
                opt_metric_iter, operator, x_iter)
            rule_logic = f"(X['{column}']{operator}{x_max_opt_metric})"
            rule_name = self._generate_rule_name()
            rule_strings[rule_name] = rule_logic
        ara = ArgoRuleApplier(rule_strings=rule_strings,
                              opt_func=self.opt_func)
        X_rules = ara.apply(X=X, y=y, sample_weight=sample_weight)
        rule_descriptions = ara.rule_descriptions
        # Remove rules with zero variance and precision == 0
        rule_descriptions, X_rules = self._drop_zero_var_and_precision_rules(
            rule_descriptions=rule_descriptions, X_rules=X_rules)
        return rule_descriptions, X_rules

    def _generate_categorical_one_condition_rules(self, X: pd.DataFrame,
                                                  y: pd.DataFrame,
                                                  columns_cat: list,
                                                  sample_weight: pd.DataFrame) -> tuple:
        """Optimises OHE categorical columns based on Fbeta score"""

        rule_descriptions, X_rules = argo_utils.generate_empty_data_structures()
        rule_descriptions_list = []
        X_rules_list = []
        for col in columns_cat:
            rule_descriptions_col_list, X_rules_col_list = [], []
            for value in ['True', 'False']:
                rule_name = self._generate_rule_name()
                rule_logic = f"(X['{col}']=={value})"
                rule_strings = {rule_name: rule_logic}
                ara = ArgoRuleApplier(
                    rule_strings=rule_strings, opt_func=self.one_cond_rule_opt_func)
                X_rule = ara.apply(X=X, y=y, sample_weight=sample_weight)
                rule_description = ara.rule_descriptions
                if rule_description.iloc[0]['Precision'] == 0:
                    continue
                rule_descriptions_col_list.append(rule_description)
                X_rules_col_list.append(X_rule)
            rule_descriptions_col = pd.concat(
                rule_descriptions_col_list, axis=0)
            X_rules_col = pd.concat(X_rules_col_list, axis=1)
            # Keep only best performing condition per feature
            rule_descriptions_col = rule_descriptions_col.sort_values(
                'OptMetric', ascending=False).head(1)
            X_rules_col = X_rules_col[rule_descriptions_col.index]
            # Re-calculate OptMetric value using opt_func (rather than
            # one_cond_rule_opt_func)
            rule_descriptions_col['OptMetric'][0] = self.opt_func(
                y_true=y, y_pred=X_rules_col.squeeze(), sample_weight=sample_weight)
            rule_descriptions_list.append(rule_descriptions_col)
            X_rules_list.append(X_rules_col)
        rule_descriptions = pd.concat(rule_descriptions_list, axis=0)
        X_rules = pd.concat(X_rules_list, axis=1)
        # Remove rules with zero variance and precision == 0
        rule_descriptions, X_rules = self._drop_zero_var_and_precision_rules(
            rule_descriptions=rule_descriptions, X_rules=X_rules)
        return rule_descriptions, X_rules

    def _generate_pairwise_rules(self, rule_descriptions: pd.DataFrame,
                                 X_rules: pd.DataFrame, y: pd.Series,
                                 rules_combinations: list,
                                 sample_weight: pd.Series) -> tuple:
        """Combines binary columns of rules using AND conditions"""

        pairwise_info_dict = self._return_pairwise_information(
            rules_combinations)
        pairwise_logics = list(pairwise_info_dict.keys())
        pairwise_info_list = list(pairwise_info_dict.values())
        rules_names_1, rules_names_2, pairwise_names = [], [], []
        for info_dict in pairwise_info_list:
            rules_names_1.append(info_dict['RuleName1'])
            rules_names_2.append(info_dict['RuleName2'])
            pairwise_names.append(info_dict['PairwiseRuleName'])
        X_rules_pairwise_df = self._generate_pairwise_df(
            X_rules, rules_names_1, rules_names_2, pairwise_names)
        pairwise_descriptions = argo_utils.return_binary_pred_perf_of_set_numpy(
            y_true=y, y_preds=X_rules_pairwise_df, y_preds_columns=pairwise_names,
            sample_weight=sample_weight, opt_func=self.opt_func)
        pairwise_descriptions.index.name = 'Rule'
        pairwise_descriptions['Logic'] = pairwise_logics
        pairwise_descriptions['nConditions'] = pairwise_descriptions['Logic'].apply(
            argo_utils.count_rule_conditions)
        pairwise_descriptions = pairwise_descriptions.reindex(
            ['Logic', 'Precision', 'Recall', 'nConditions', 'PercDataFlagged', 'OptMetric'], axis=1)
        pairwise_components = dict((info_dict['PairwiseRuleName'], info_dict['PairwiseComponents'])
                                   for _, info_dict in pairwise_info_dict.items())
        return pairwise_descriptions, X_rules_pairwise_df, pairwise_components

    def _drop_unnecessary_pairwise_rules(self, pairwise_descriptions: pd.DataFrame,
                                         X_rules_pairwise_df: pd.DataFrame,
                                         pairwise_to_orig_lookup: dict,
                                         rule_descriptions: pd.DataFrame) -> tuple:
        """
        Drops pairwise rules with precision == 0 or that have a precision less 
        than one of their component rules.
        """

        zero_var_rules = self._return_zero_variance_rules(
            X_rules=X_rules_pairwise_df)
        zero_prec_rules = self._return_zero_precision_rules(
            rule_descriptions=pairwise_descriptions)
        # Get rules with precision less than either of the individual rules
        pw_rules_less_prec = self._return_pairwise_rules_to_drop(
            pairwise_descriptions, pairwise_to_orig_lookup, rule_descriptions)
        rules_to_drop = list(
            set(zero_var_rules + zero_prec_rules + pw_rules_less_prec))
        pairwise_descriptions = pairwise_descriptions.drop(
            rules_to_drop, axis=0)
        X_rules_pairwise_df = X_rules_pairwise_df.drop(
            rules_to_drop, axis=1)

        return pairwise_descriptions, X_rules_pairwise_df

    def _generate_n_order_pairwise_rules(self, rule_descriptions: pd.DataFrame,
                                         X_rules: pd.DataFrame, y: pd.Series,
                                         remove_corr_rules: bool,
                                         sample_weight: pd.Series) -> tuple:
        """
        Loops through ruleset (starting with one condition rules) and combines 
        them pairwise to a given order.
        """

        n_loops = int(
            math.log(2 ** math.ceil(math.log(self.n_total_conditions, 2)), 2))
        for n_loop in range(1, n_loops + 1):
            if remove_corr_rules:
                rule_descriptions, X_rules = self._remove_corr_rules(
                    rule_descriptions=rule_descriptions, X_rules=X_rules)
            rules_combinations = self._get_rule_combinations_for_loop(
                rule_descriptions, n_loop, self.num_rules_keep)
            if len(rules_combinations) == 0:
                break
            rule_descriptions_pairwise, X_rules_pairwise, pairwise_components = self._generate_pairwise_rules(
                rule_descriptions, X_rules, y, rules_combinations, sample_weight)
            rule_descriptions_pairwise, X_rules_pairwise = self._drop_unnecessary_pairwise_rules(
                rule_descriptions_pairwise, X_rules_pairwise,  pairwise_components, rule_descriptions)
            X_rules = pd.concat(
                [X_rules, X_rules_pairwise], axis=1)
            rule_descriptions = pd.concat(
                [rule_descriptions, rule_descriptions_pairwise], axis=0)
        rule_descriptions = rule_descriptions[rule_descriptions['nConditions']
                                              <= self.n_total_conditions]
        X_rules = X_rules[rule_descriptions.index.tolist()]
        rule_descriptions, X_rules = argo_utils.sort_rule_dfs_by_opt_metric(
            rule_descriptions, X_rules)

        return rule_descriptions, X_rules

    def _return_pairwise_information(self, rules_combinations: list) -> dict:
        """
        Returns a dict of the pairwise rule logic and associated 
        information
        """

        def clean_rule_logic(rule_name: str) -> str:
            """Removes duplicate columns in combined rule logic"""
            rule_name_list = rule_name.split("&")
            rule_name_set = sorted(list(set(rule_name_list)))
            rule_name = '&'.join(rule_name_set)
            return rule_name

        pairwise_info_dict = {}
        rule_logics_list = []
        # Loop through rule combinations and calculate pairwise logic. Then
        # link the component rule names, logic and distinct components to the
        # pairwise logic
        for (rule_name_1, rule_name_2), (rule_logic_1, rule_logic_2) in rules_combinations:
            pairwise_rule_logic = clean_rule_logic(
                f'{rule_logic_1}&{rule_logic_2}')
            if rule_logics_list.count(pairwise_rule_logic) == 0:
                pairwise_rule_name = self._generate_rule_name()
                pairwise_info_dict[pairwise_rule_logic] = {
                    'RuleName1': rule_name_1,
                    'RuleName2': rule_name_2,
                    'PairwiseRuleName': pairwise_rule_name,
                    'PairwiseComponents': [rule_name_1, rule_name_2]
                }
                rule_logics_list.append(pairwise_rule_logic)
            else:
                pairwise_info_dict[pairwise_rule_logic]['PairwiseComponents'].extend(
                    [rule_name_1, rule_name_2])
                pairwise_info_dict[pairwise_rule_logic]['PairwiseComponents'] = list(set(
                    pairwise_info_dict[pairwise_rule_logic]['PairwiseComponents']))
        return pairwise_info_dict

    def _drop_zero_var_and_precision_rules(self,
                                           rule_descriptions: pd.DataFrame,
                                           X_rules: pd.DataFrame) -> tuple:
        """
        Drops zero variance and zero precisions rules from rule_descriptions 
        and X_rules
        """

        zero_var_rules = self._return_zero_variance_rules(X_rules=X_rules)
        zero_precision_rules = self._return_zero_precision_rules(
            rule_descriptions=rule_descriptions)
        rules_to_drop = list(set(zero_var_rules + zero_precision_rules))
        rule_descriptions = rule_descriptions.drop(
            rules_to_drop, axis=0)
        X_rules = X_rules.drop(rules_to_drop, axis=1)
        return rule_descriptions, X_rules

    def _generate_rule_name(self) -> str:
        """Generates rule name"""

        rule_name = f'RGO_Rule_{self.today}_{self._rule_name_counter}'
        self._rule_name_counter += 1
        return rule_name

    @staticmethod
    def _remove_corr_rules(rule_descriptions: pd.DataFrame,
                           X_rules: pd.DataFrame) -> tuple:
        """
        Remove correlated rules using the 
        AgglomerativeClusteringFeatureReduction class
        """

        cs = CosineSimilarity()
        rcr = AgglomerativeClusteringFeatureReduction(
            threshold=0.75, columns_performance=rule_descriptions['OptMetric'],
            strategy='bottom_up', similarity_function=cs.fit)
        X_rules = rcr.fit_transform(X_rules)
        rule_descriptions = rule_descriptions.loc[X_rules.columns]
        return rule_descriptions, X_rules

    @staticmethod
    def _set_iteration_range(X_col: np.array, column: str, operator: str,
                             n_points: int, ratio_window: int,
                             columns_int: list) -> tuple:
        """Sets the iteration range for a given column"""

        X_col_max = max(X_col)
        X_col_min = min(X_col)
        if column in columns_int and n_points > abs(X_col_max - X_col_min):
            x_min = X_col_min
            x_max = X_col_max
        elif operator == "<=":
            x_min = X_col_min
            x_max = x_min + (X_col_max - x_min) / ratio_window
        elif operator == ">=":
            x_max = X_col_max
            x_min = x_max - (x_max - X_col_min) / ratio_window
        return (x_min, x_max)

    @staticmethod
    def _set_iteration_array(column: str, columns_int: list, x_min: float,
                             x_max: float, n_points: int,
                             _round_to_n_sf: object) -> np.array:
        """Returns the iteration array for a given column"""

        if column in columns_int:
            x_min, x_max = int(x_min), int(x_max)
            if abs(x_min - x_max) < n_points:
                x_iter = np.array(range(x_min, x_max + 1))
            else:
                x_iter = np.ceil(np.linspace(x_min, x_max, n_points))
        else:
            x_iter = np.linspace(x_min, x_max, n_points)
            x_iter = np.array([_round_to_n_sf(x, 2) for x in x_iter])
        return x_iter

    @staticmethod
    def _calculate_opt_metric_across_range(x_iter: np.array, operator: str,
                                           X_col: np.array, y: np.array,
                                           opt_func: object,
                                           sample_weight: np.array) -> np.array:
        """
        Calculates the optimisation function at each point in the x_iter 
        range
        """

        opt_metric_iter = np.zeros(len(x_iter))
        for i, x in enumerate(x_iter):
            X_rule = eval(f'X_col {operator} {x}').astype(int)
            opt_metric_iter[i] = opt_func(
                y_true=y, y_pred=X_rule, sample_weight=sample_weight)
        return opt_metric_iter

    @staticmethod
    def _return_x_of_max_opt_metric(opt_metric_iter: np.array, operator: str,
                                    x_iter: np.array) -> np.number:
        """Returns the threshold value which maximises the FBeta score"""

        max_opt_metric = np.nanmax(opt_metric_iter)
        if max_opt_metric == 0:
            return None
        if operator == "<=":
            idx_max_opt_metric = min([i for i, j in enumerate(
                opt_metric_iter) if j == max_opt_metric])
        elif operator == ">=":
            idx_max_opt_metric = max([i for i, j in enumerate(
                opt_metric_iter) if j == max_opt_metric])
        return x_iter[idx_max_opt_metric]

    @staticmethod
    def _round_to_n_sf(x: float, n_sf: int) -> float:
        """Method for rounding a float to n significant figures"""

        if x == 0:
            return 0
        return round(x, -int(math.floor(math.log10(abs(x)))) + (n_sf - 1))

    @staticmethod
    def _get_rule_combinations_for_loop(rule_descriptions: pd.DataFrame,
                                        n_loop: int,
                                        num_rules_keep: int) -> list:
        """Generates pairwise combinations of rules"""

        # At beginning of each loop, filter list of rules to include only those
        # needed for pairwise calculation
        rules_n_conditions = rule_descriptions[(
            rule_descriptions['nConditions'] == 2 ** (n_loop - 1))]
        # Then sort resulting ruleset by OptMetric and take top num_rules_keep
        # rules for pairwise calculation
        rules_n_conditions = rules_n_conditions.sort_values(
            by='OptMetric', ascending=False)
        rules_n_conditions = rules_n_conditions.iloc[:num_rules_keep]
        # Get the rule names and their logic
        rule_names = rules_n_conditions.index.values
        rule_logic = rules_n_conditions['Logic'].values
        # Calculate distinct combinations of both the rule names and their
        # logic
        rules_logic_combinations = list(combinations(rule_logic, r=2))
        rules_name_combinations = list(combinations(rule_names, r=2))
        # Combine these into a list
        rules_combinations = list(
            zip(rules_name_combinations, rules_logic_combinations))
        return rules_combinations

    @staticmethod
    def _generate_pairwise_df(X_rules: pd.DataFrame, rules_names_1: list,
                              rules_names_2: list,
                              pairwise_names: list) -> pd.DataFrame:
        """
        Multiplies the component rules together to give the pairwise dataframe
        """

        X_rules_pairwise_arr = X_rules[rules_names_1].values * \
            X_rules[rules_names_2].values
        X_rules_pairwise_df = pd.DataFrame(
            X_rules_pairwise_arr, columns=pairwise_names, index=X_rules.index)
        return X_rules_pairwise_df

    @staticmethod
    def _return_pairwise_rules_to_drop(pairwise_descriptions: pd.DataFrame,
                                       pairwise_to_orig_lookup: dict,
                                       rule_descriptions: pd.DataFrame) -> list:
        """
        Drops pairwise rule if its precision is less than or equal to the 
        precision of one of its component rules
        """

        rules_to_drop = []
        for idx, row in pairwise_descriptions.iterrows():
            orig_rules = pairwise_to_orig_lookup[idx]
            max_orig_prec = rule_descriptions.loc[orig_rules, 'Precision'].max(
            )
            if row['Precision'] <= max_orig_prec:
                rules_to_drop.append(idx)
        return rules_to_drop

    @staticmethod
    def _return_zero_variance_rules(X_rules: pd.DataFrame) -> list:
        """Returns list of zero variance rules"""

        X_rules_std = X_rules.to_numpy().std(axis=0)
        mask = X_rules_std == 0
        zero_var_rules = X_rules.columns[mask].tolist()
        return zero_var_rules

    @staticmethod
    def _return_zero_precision_rules(rule_descriptions: pd.DataFrame) -> list:
        """Returns list of zero precision rules"""

        mask = rule_descriptions['Precision'].to_numpy() == 0
        zero_precision_rules = rule_descriptions.index[mask].tolist()
        return zero_precision_rules
