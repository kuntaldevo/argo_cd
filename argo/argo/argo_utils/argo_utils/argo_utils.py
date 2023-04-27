"""Shared functions for ARGO packages"""
import pandas as pd
import numpy as np
import math
import json


def convert_conditions_to_argo_string(list_of_conditions: list,
                                      columns_int: list,
                                      columns_cat: list) -> str:
    """
    Converts a list of conditions to the standard ARGO string format.

    Args:
        list_of_conditions (list): Each element contains a tuple of the 
            feature (str), operator (str) and value (numeric) for each 
            condition in the rule.
        columns_int (list): List of integer columns.
        columns_cat (list): List of OHE categorical columns.

    Returns:
        str: The ARGO-readable rule name.
    """

    def convert_values_for_columns_int(feature, operator, value):
        """
        Method for converting a condition containing an integer value from 
        float to int
        """
        if operator in ['>=', '>']:
            return feature, '>=', math.ceil(value)
        elif operator in ['<=', '<']:
            return feature, '<=', math.floor(value)
        else:
            return feature, operator, value

    conditions = []
    for feature, operator, value in list_of_conditions:
        if feature in columns_cat:
            if (operator == '<=' and value < 1) or (operator == '==' and value == 0):
                condition = f"(X['{feature}']==False)"
            elif (operator == '>' and value >= 0) or (operator == '==' and value == 1):
                condition = f"(X['{feature}']==True)"
        # If feature is an int, round the value
        elif feature in columns_int:
            feature, operator, value = convert_values_for_columns_int(
                feature, operator, value)
            condition = f"(X['{feature}']{operator}{value})"
        else:
            condition = f"(X['{feature}']{operator}{value})"
        conditions.append(condition)
    conditions.sort()
    name = '&'.join(conditions)
    return name


def clean_dup_features_from_conditions(list_of_conditions: list) -> list:
    """
    Removes unnecessary conditions from a rule (e.g. for a branch in a tree, 
    the same feature and condition can be referenced, the threshold value is 
    different. This method just takes the relevant threshold value).

    Args:
        list_of_conditions (list): Each element contains a tuple of the 
            feature, operator and value for each condition in the rule.

    Returns:
        list: Cleaned list of conditions
    """

    def dedupe_conditions(feature, operator):
        list_of_values = [
            val for feat, op, val in list_of_conditions if feat == feature and op == operator]
        if operator in ['<', '<=']:
            return feature, operator, min(list_of_values)
        if operator in ['>', '>=']:
            return feature, operator, max(list_of_values)

    unique_feat_op_list = {(feat, op) for feat, op, _ in list_of_conditions}
    list_of_conditions_cleaned = [dedupe_conditions(
        unique_feat, unique_op) for unique_feat, unique_op in unique_feat_op_list]
    list_of_conditions_cleaned.sort()
    return list_of_conditions_cleaned


def generate_empty_data_structures() -> tuple:
    """
    Creates data structures often used in classes in ARGO.

    Returns:
        tuple: Contains the rule_descriptions and X_rules dataframes
    """
    columns = [
        'Rule', 'Precision', 'Recall', 'nConditions', 'PercDataFlagged', 'OptMetric'
    ]
    rule_descriptions = pd.DataFrame(columns=columns)
    rule_descriptions.set_index('Rule', inplace=True)
    X_rules = pd.DataFrame([])
    return rule_descriptions, X_rules


def return_columns_types(X: pd.DataFrame) -> tuple:
    """
    Returns the integer, float and OHE categorical columns for a given dataset.

    Args:
        X (pd.DataFrame): Dataset.

    Returns:
        tuple: list of integer columns, list of float columns, list of OHE 
            categorical columns
    """
    int_cols = list(X.select_dtypes('Int64').columns)
    X_no_int64 = X.drop(int_cols, axis=1)
    int_mask = np.subtract(X_no_int64, X_no_int64.round()).sum() == 0
    int_cols = int_cols + list(X_no_int64.columns[int_mask])
    float_cols = list(X_no_int64.columns[~int_mask])
    ohe_cat_cols = []
    for col in int_cols:
        unique_values = X[col].unique()
        unique_values = np.sort(unique_values)
        unique_values = set(unique_values)
        if unique_values == {0, 1}:
            ohe_cat_cols.append(col)
    return int_cols, ohe_cat_cols, float_cols


def sort_rule_dfs_by_opt_metric(rule_descriptions: pd.DataFrame,
                                X_rules: pd.DataFrame) -> tuple:
    """
    Method for sorting and reindexing rule_descriptions and X_rules by 
    opt_metric
    """

    rule_descriptions.sort_values(
        by=['OptMetric'], ascending=False, inplace=True)
    X_rules = X_rules.reindex(rule_descriptions.index.tolist(), axis=1)
    return rule_descriptions, X_rules


def combine_rule_dfs(rule_descriptions_1: pd.DataFrame,
                     X_rules_1: pd.DataFrame,
                     rule_descriptions_2: pd.DataFrame,
                     X_rules_2: pd.DataFrame) -> tuple:
    """Combines the rule_description and X_rules object of two rule sets"""

    rule_descriptions = pd.concat(
        [rule_descriptions_1, rule_descriptions_2], axis=0)
    X_rules = pd.concat([X_rules_1, X_rules_2], axis=1)

    return rule_descriptions, X_rules


def return_opt_func_perf(opt_func: object, y_preds: np.ndarray, y_true=None,
                         sample_weight=None) -> np.ndarray:
    """
    Calculates the given optimisation function across one or multiple binary
    predictors.

    Args:
        opt_func (object): A function/method which calculates a custom metric 
            (e.g. Fbeta score) for each column.
        y_preds (np.ndarray): Set of binary integer predictors. Can also be a 
            single predictor.
        y_true (np.ndarray, optional): Binary integer target column. Defaults 
            to None.        
        sample_weight (np.ndarray, optional): Row-wise sample_weights to apply. 
            Defaults to None.

    Returns:
        np.ndarray: The optimisation metric for each record.
    """

    # Convert relavent args to numpy arrays
    if isinstance(y_preds, (pd.Series, pd.DataFrame)):
        y_preds = y_preds.values
    if isinstance(y_true, pd.Series) and y_true is not None:
        y_true = y_true.values
    if isinstance(sample_weight, pd.Series) and sample_weight is not None:
        sample_weight = sample_weight.values
    # If one binary predictor and labelled data
    if y_preds.ndim == 1 and y_true is not None:
        opt_metric_results = np.array([opt_func(
            y_true=y_true, y_pred=y_preds, sample_weight=sample_weight)])
    # If one binary predictor and unlabelled data
    elif y_preds.ndim == 1 and y_true is None:
        opt_metric_results = np.array([opt_func(y_pred=y_preds)])
    # If more than binary predictor and labelled data
    elif y_preds.ndim != 1 and y_true is not None:
        opt_metric_results = np.array([opt_func(
            y_true=y_true, y_pred=y_preds[:, i], sample_weight=sample_weight) for i in range(0, y_preds.shape[1])])
    # If more than binary predictor and unlabelled data
    elif y_preds.ndim != 1 and y_true is None:
        opt_metric_results = np.array(
            [opt_func(y_pred=y_preds[:, i]) for i in range(0, y_preds.shape[1])])
    return opt_metric_results


def return_binary_pred_perf_of_set_numpy(y_true: np.ndarray,
                                         y_preds: np.ndarray,
                                         y_preds_columns: list,
                                         sample_weight=None,
                                         opt_func=None) -> pd.DataFrame:
    """
    Calculates the performance of a set of binary predictors given a target 
    column.

    Args:
        y_true (np.ndarray): Binary integer target column.
        y_preds (np.ndarray): Set of binary integer predictors. Can also be a 
            single predictor.
        y_preds_columns (list): Column names for the y_preds array.
        sample_weight (np.ndarray, optional): Row-wise sample_weights to apply. 
            Defaults to None.
        opt_func (object, optional): A function/method which calculates a 
            custom metric (e.g. Fbeta score) for each column. Defaults to None.

    Returns:
        pd.DataFrame: Dataframe containing the performance metrics for each 
            binary predictor.
    """
    # Convert relavent args to numpy arrays
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_preds, (pd.Series, pd.DataFrame)):
        y_preds = y_preds.values
    if isinstance(sample_weight, pd.Series) and sample_weight is not None:
        sample_weight = sample_weight.values
    perc_data_flagged = y_preds.mean(0)
    # Calculate opt_metric
    if opt_func is not None:
        opt_metric_results = return_opt_func_perf(opt_func=opt_func, y_preds=y_preds,
                                                  y_true=y_true, sample_weight=sample_weight)
    else:
        opt_metric_results = None
    # Reshape y_true into same shape as pairwise array
    if y_preds.shape != y_true.shape:
        y_true_arr = np.tile(y_true, (y_preds.shape[1], 1)).T.astype(int)
    else:
        y_true_arr = y_true
    # Calculate intersection of pairwise binary cols and y_true
    if sample_weight is not None:
        if y_preds.shape != sample_weight.shape:
            sample_weight_arr = np.tile(
                sample_weight, (y_preds.shape[1], 1)).T.astype(int)
        else:
            sample_weight_arr = sample_weight
        y_preds = y_preds * sample_weight_arr
        binary_y_intersect = y_preds * y_true_arr
        y_true = y_true * sample_weight
    else:
        binary_y_intersect = y_preds * y_true_arr
    # Calculate num records flagged and ratio flagged per rule
    y_preds_sums = y_preds.sum(0)
    y_preds_sums = np.where(y_preds_sums == 0, np.nan, y_preds_sums)
    # Calculate num records flagged in intersection
    binary_y_intersect_sum = binary_y_intersect.sum(0)
    # Calculate precision, recall and fscore of pairwise rules
    precisions = np.nan_to_num(np.divide(binary_y_intersect_sum, y_preds_sums))
    recalls = np.nan_to_num(np.divide(binary_y_intersect_sum, y_true.sum()))
    results = pd.DataFrame({
        'Precision': precisions,
        'Recall': recalls,
        'PercDataFlagged': perc_data_flagged,
        'OptMetric': opt_metric_results,
    }, index=y_preds_columns)

    return results


def return_rule_descriptions_from_X_rules(X_rules: pd.DataFrame,
                                          X_rules_cols: list,
                                          y_true: None,
                                          sample_weight=None,
                                          opt_func=None) -> pd.DataFrame:
    """
    Calculates the performance metrics for the standard `rule_descriptions`
    dataframe, given a set of rule binary columns

    Args:
        X_rules (pd.DataFrame): Set of rule binary columns.
        X_rules_cols (list): Columns associated with `X_rules`.
        y_true (np.ndarray, optional): Binary integer target column. Defaults
            to None.
        sample_weight (np.ndarray, optional): Row-wise sample_weights to apply. 
            Defaults to None.
        opt_func (object, optional): A function/method which calculates a 
            custom metric (e.g. Fbeta score) for each rule. Defaults to None. 

    Returns:
        pd.DataFrame: The performance metrics for the standard 
            `rule_descriptions` dataframe.
    """

    if y_true is not None:
        rule_descriptions = return_binary_pred_perf_of_set_numpy(y_true=y_true,
                                                                 y_preds=X_rules,
                                                                 y_preds_columns=X_rules_cols,
                                                                 sample_weight=sample_weight,
                                                                 opt_func=opt_func
                                                                 )
        rule_descriptions.index.name = 'Rule'
    else:
        opt_metric_results = return_opt_func_perf(opt_func=opt_func,
                                                  y_preds=X_rules)
        perc_data_flagged = X_rules.mean(0)
        rule_descriptions = pd.DataFrame(data={
            'PercDataFlagged': perc_data_flagged,
            'OptMetric': opt_metric_results,
        },
            index=X_rules_cols)
        rule_descriptions.index.name = 'Rule'
    return rule_descriptions


def flatten_stringified_json_column(X_column: pd.Series) -> pd.DataFrame:
    """
    Flattens JSONs contained in a column to their own columns.

    Args:
        X_column (pd.Series): Contains the JSONs to be flattened.

    Returns:
        pd.DataFrame: Contains a column per key-value pair in the JSONs.
    """

    X_column.fillna('{}', inplace=True)
    X_flattened = pd.DataFrame(
        list(X_column.apply(lambda x: json.loads(x)).values))
    X_flattened.set_index(X_column.index.values, inplace=True)

    return X_flattened


def count_rule_conditions(rule_string: str) -> int:
    """
    Counts the number of conditions in a rule string.

    Args:
        rule_string (str): The standard ARGO string representation
            of the rule.

    Returns:
        int: Number of conditions in the rule.
    """
    n_conditions = rule_string.count("X['")
    return n_conditions


# def return_binary_pred_perf_of_set_pandas(y_true, y_preds,
#                                           sample_weight=None, opt_func=None):
#     total_num_flagged = y_preds_ks.sum().to_pandas()
#     total_perc_flagged = y_preds_ks.mean().to_pandas()
#     new_col_names = []
#     if sample_weight is not None:
#         y_preds_and_true = ks.concat([y_preds_ks, y_true_ks.rename(
#             'y_true'), sample_weight.rename('sample_weight')], axis=1)
#     else:
#         y_preds_and_true = ks.concat(
#             [y_preds_ks, y_true_ks.rename('y_true')], axis=1)
#     for col in y_preds_and_true:
#         if col == 'y_true' or col == 'sample_weight':
#             continue
#         new_col_name = f'{col}_equals_y_true'
#         if sample_weight is not None:
#             y_preds_and_true[new_col_name] = y_preds_and_true[col] * \
#                 y_preds_and_true['y_true'] * y_preds_and_true['sample_weight']
#         else:
#             y_preds_and_true[new_col_name] = y_preds_and_true[col] * \
#                 y_preds_and_true['y_true']
#         new_col_names.append(new_col_name)
#     preds_equal_true = y_preds_and_true[new_col_names].sum().to_pandas()
#     precisions = pd.Series(
#         preds_equal_true.values/total_num_flagged.values, y_preds.columns, name='Precision')
#     return precisions
