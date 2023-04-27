import pytest
import numpy as np
import pandas as pd
import json
import argo_utils.argo_utils as argo_utils
from sklearn.metrics import fbeta_score, precision_score, recall_score
import string
from rule_optimisation.optimisation_functions import FScore, AlertsPerDay


@pytest.fixture
def create_data():
    np.random.seed(0)
    y_preds = pd.DataFrame(np.random.randint(0, 2, size=(1000, 10)), columns=[
                           i for i in string.ascii_letters[:10]])
    y_true = pd.Series(np.random.randint(0, 2, 1000))
    sample_weight = y_true * 10
    return (y_true, y_preds, sample_weight)


@pytest.fixture
def create_dummy_rules():
    rule_descriptions = pd.DataFrame({
        'Precision': [1.0, 0.5, 0.75, 0.25, 0],
        'Recall': [0.5, 0.7, 1, 0.75, 0],
        'nConditions': [1, 2, 3, 4, 5],
        'PercDataFlagged': [0.2, 0.4, 0.6, 0.45, 0.1],
        'Type': ['Fraud', 'Fraud', 'Fraud', 'Fraud', 'Fraud'],
        'OptMetric': [0.75, 0.6, 0.85, 0.5, 0]
    },
        index=['A', 'B', 'C', 'D', 'E']
    )
    X_rules = pd.DataFrame({
        'A': [1, 0, 1, 1, 0],
        'B': [1, 0, 0, 1, 1],
        'C': [1, 0, 0, 0, 1],
        'D': [1, 1, 1, 0, 1],
        'E': [1, 0, 0, 0, 1]
    })
    return rule_descriptions, X_rules


def test_convert_conditions_to_argo_string():
    list_of_conditions = [
        ('A', '>=', 1),
        ('B', '<=', 1.5),
        ('C', '==', 1),
        ('D', '<=', 2.9),
        ('E', '>', 0.5)
    ]
    columns_int = ['B', 'C', 'E']
    columns_cat = ['C', 'E']
    rule_name = argo_utils.convert_conditions_to_argo_string(
        list_of_conditions, columns_int, columns_cat)
    assert rule_name == "(X['A']>=1)&(X['B']<=1)&(X['C']==True)&(X['D']<=2.9)&(X['E']==True)"


def test_clean_dup_features_from_conditions():
    list_of_conditions = [
        ('A', '>=', 1),
        ('A', '>=', 3),
        ('B', '<=', 2),
        ('B', '<=', 1),
        ('C', '>', 0.5),
        ('C', '<=', 1)
    ]
    expected_result = [
        ('A', '>=', 3),
        ('B', '<=', 1),
        ('C', '<=', 1),
        ('C', '>', 0.5)
    ]
    result = argo_utils.clean_dup_features_from_conditions(list_of_conditions)
    assert result == expected_result


def test_generate_empty_data_structures():
    rd, xr = argo_utils.generate_empty_data_structures()
    assert all(rd.columns == ['Precision', 'Recall',
                              'nConditions', 'PercDataFlagged', 'OptMetric'])
    assert rd.index.name == 'Rule'
    assert isinstance(xr, pd.DataFrame)


def test_return_columns_types():
    X = pd.DataFrame({
        'A': [2.5, 3.5, 1, 1, 2.5],
        'B': [1, 0, 0, 1, 1],
        'C': [1, 2, 0, 0, 1]
    })
    int_cols, cat_cols, float_cols = argo_utils.return_columns_types(X)
    assert int_cols == ['B', 'C']
    assert cat_cols == ['B']
    assert float_cols == ['A']


def test_sort_rule_dfs_by_opt_metric():
    rule_descriptions = pd.DataFrame({
        'Precision': [1.0, 0.5, 0.75, 0.25, 0],
        'Recall': [0.5, 0.7, 1, 0.75, 0],
        'nConditions': [1, 2, 3, 4, 5],
        'PercDataFlagged': [0.2, 0.4, 0.6, 0.45, 0.1],
        'OptMetric': [0.75, 0.6, 0.85, 0.5, 0]
    },
        index=['A', 'B', 'C', 'D', 'E']
    )
    X_rules = pd.DataFrame({
        'A': [1, 0, 1, 1, 0],
        'B': [1, 0, 0, 1, 1],
        'C': [1, 0, 0, 0, 1],
        'D': [1, 1, 1, 0, 1],
        'E': [1, 0, 0, 0, 1]
    })
    rd, xr = argo_utils.sort_rule_dfs_by_opt_metric(rule_descriptions, X_rules)
    assert rd.index.tolist() == ['C', 'A', 'B', 'D', 'E']
    assert xr.columns.tolist() == ['C', 'A', 'B', 'D', 'E']


def test_combine_rule_dfs(create_dummy_rules):
    rule_descriptions, X_rules = create_dummy_rules
    rd1 = rule_descriptions.loc[['A', 'B']]
    rd2 = rule_descriptions.loc[['C', 'D', 'E']]
    xr1 = X_rules[['A', 'B']]
    xr2 = X_rules[['C', 'D', 'E']]
    rd_comb, xr_comb = argo_utils.combine_rule_dfs(rd1, xr1, rd2, xr2)
    assert all(rd_comb == rule_descriptions)
    assert all(xr_comb == X_rules)


def test_return_opt_func_perf(create_data):
    y_true, y_preds, weights = create_data
    f1 = FScore(beta=1)
    exp_result = np.array([0.51171875, 0.51815505, 0.5184466, 0.53149606,
                           0.52912142, 0.53731343, 0.49447236, 0.52274927,
                           0.51637765, 0.50197628])
    exp_result_weighted = np.array([0.67875648, 0.68217054, 0.68725869,
                                    0.69230769, 0.68894602, 0.69230769,
                                    0.65079365, 0.69230769, 0.68894602,
                                    0.66492147])
    # Multiple preds
    opt_metric_results = argo_utils.return_opt_func_perf(
        opt_func=f1.fit, y_preds=y_preds, y_true=y_true,
        sample_weight=None)
    assert all(np.isclose(opt_metric_results, exp_result))
    # Multiple preds, weighted
    opt_metric_results_weighted = argo_utils.return_opt_func_perf(
        opt_func=f1.fit, y_preds=y_preds, y_true=y_true,
        sample_weight=weights)
    assert all(np.isclose(opt_metric_results_weighted, exp_result_weighted))
    # One pred
    opt_metric_results = argo_utils.return_opt_func_perf(
        opt_func=f1.fit, y_preds=y_preds.iloc[:, 0], y_true=y_true,
        sample_weight=None)
    assert all(np.isclose(opt_metric_results, np.array([exp_result[0]])))
    # One pred, weighted
    opt_metric_results_weighted = argo_utils.return_opt_func_perf(
        opt_func=f1.fit, y_preds=y_preds.iloc[:, 0], y_true=y_true,
        sample_weight=weights)
    assert all(np.isclose(opt_metric_results_weighted,
                          np.array([exp_result_weighted[0]])))


def test_return_opt_func_perf_unlabelled(create_data):
    _, y_preds, _ = create_data
    apd = AlertsPerDay(n_alerts_expected_per_day=10, no_of_days_in_file=10)
    exp_result = np.array([-1713.96, -1672.81, -1764., -1648.36, -1624.09,
                           -1560.25, -1482.25, -1789.29, -1831.84, -1616.04])
    # Multiple preds
    opt_metric_results = argo_utils.return_opt_func_perf(
        opt_func=apd.fit, y_preds=y_preds)
    assert all(np.isclose(opt_metric_results, exp_result))
    # One pred
    opt_metric_results = argo_utils.return_opt_func_perf(
        opt_func=apd.fit, y_preds=y_preds.iloc[:, 0])
    assert all(np.isclose(opt_metric_results, np.array([exp_result[0]])))


def test_return_rule_descriptions_from_X_rules(create_data):
    y_true, y_preds, _ = create_data
    f1 = FScore(beta=1)
    exp_results = pd.DataFrame(
        np.array([[0.50972763, 0.51372549, 0.514, 0.51171875],
                  [0.51866405, 0.51764706, 0.509, 0.51815505],
                  [0.51346154, 0.52352941, 0.52, 0.5184466],
                  [0.53359684, 0.52941176, 0.506, 0.53149606],
                  [0.53280318, 0.5254902, 0.503, 0.52912142],
                  [0.54545455, 0.52941176, 0.495, 0.53731343],
                  [0.50721649, 0.48235294, 0.485, 0.49447236],
                  [0.51625239, 0.52941176, 0.523, 0.52274927],
                  [0.50757576, 0.5254902, 0.528, 0.51637765],
                  [0.5059761, 0.49803922, 0.502, 0.50197628]]),
        columns=['Precision', 'Recall', 'PercDataFlagged', 'OptMetric'],
        index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    )
    exp_results.index.name = 'Rule'
    rule_descriptions = argo_utils.return_rule_descriptions_from_X_rules(X_rules=y_preds,
                                                                         X_rules_cols=y_preds.columns,
                                                                         y_true=y_true,
                                                                         sample_weight=None,
                                                                         opt_func=f1.fit)
    assert all(rule_descriptions == exp_results)
    rule_descriptions = argo_utils.return_rule_descriptions_from_X_rules(X_rules=y_preds,
                                                                         X_rules_cols=y_preds.columns,
                                                                         y_true=y_true,
                                                                         sample_weight=None)
    exp_results['OptMetric'] = None
    assert all(rule_descriptions == exp_results)


def test_return_rule_descriptions_from_X_rules_weighted(create_data):
    y_true, y_preds, weights = create_data
    f1 = FScore(beta=1)
    exp_results = pd.DataFrame(
        np.array([[1., 0.51372549, 0.514, 0.67875648],
                  [1., 0.51764706, 0.509, 0.68217054],
                  [1., 0.52352941, 0.52, 0.68725869],
                  [1., 0.52941176, 0.506, 0.69230769],
                  [1., 0.5254902, 0.503, 0.68894602],
                  [1., 0.52941176, 0.495, 0.69230769],
                  [1., 0.48235294, 0.485, 0.65079365],
                  [1., 0.52941176, 0.523, 0.69230769],
                  [1., 0.5254902, 0.528, 0.68894602],
                  [1., 0.49803922, 0.502, 0.66492147]]),
        columns=['Precision', 'Recall', 'PercDataFlagged', 'OptMetric'],
        index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    )
    exp_results.index.name = 'Rule'
    rule_descriptions = argo_utils.return_rule_descriptions_from_X_rules(X_rules=y_preds,
                                                                         X_rules_cols=y_preds.columns,
                                                                         y_true=y_true,
                                                                         sample_weight=weights,
                                                                         opt_func=f1.fit)
    assert all(rule_descriptions == exp_results)
    rule_descriptions = argo_utils.return_rule_descriptions_from_X_rules(X_rules=y_preds,
                                                                         X_rules_cols=y_preds.columns,
                                                                         y_true=y_true,
                                                                         sample_weight=weights)
    exp_results['OptMetric'] = None
    assert all(rule_descriptions == exp_results)


def test_return_rule_descriptions_from_X_rules_unlabelled(create_data):
    _, y_preds, _ = create_data
    apd = AlertsPerDay(n_alerts_expected_per_day=10, no_of_days_in_file=10)
    exp_results = pd.DataFrame(
        np.array([[5.14000e-01, -1.71396e+03],
                  [5.09000e-01, -1.67281e+03],
                  [5.20000e-01, -1.76400e+03],
                  [5.06000e-01, -1.64836e+03],
                  [5.03000e-01, -1.62409e+03],
                  [4.95000e-01, -1.56025e+03],
                  [4.85000e-01, -1.48225e+03],
                  [5.23000e-01, -1.78929e+03],
                  [5.28000e-01, -1.83184e+03],
                  [5.02000e-01, -1.61604e+03]]),
        columns=['PercDataFlagged', 'OptMetric'],
        index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    )
    exp_results.index.name = 'Rule'
    rule_descriptions = argo_utils.return_rule_descriptions_from_X_rules(X_rules=y_preds,
                                                                         X_rules_cols=y_preds.columns,
                                                                         y_true=None,
                                                                         sample_weight=None,
                                                                         opt_func=apd.fit)
    assert all(rule_descriptions == exp_results)


def test_return_binary_pred_perf_of_set_numpy(create_data):
    y_true, y_preds, weights = create_data
    f1 = FScore(beta=1)
    # Test multiple preds
    for w in [None, weights]:
        results = argo_utils.return_binary_pred_perf_of_set_numpy(
            y_true=y_true, y_preds=y_preds, y_preds_columns=y_preds.columns, sample_weight=w, opt_func=f1.fit)
        _test_y_preds(y_preds, results, y_true, w)
    # Test one pred
    y_pred = y_preds.iloc[:, 0]
    for w in [None, weights]:
        results = argo_utils.return_binary_pred_perf_of_set_numpy(
            y_true=y_true, y_preds=y_pred, y_preds_columns=y_preds.columns, sample_weight=w, opt_func=f1.fit)
        _test_y_preds(y_pred, results, y_true, w)


def _test_y_preds(y_preds, rule_descriptions, y, sample_weight):
    if y_preds.ndim == 1:
        y_preds = pd.DataFrame(y_preds)
    for col in y_preds:
        precision = precision_score(
            y, y_preds[col], sample_weight=sample_weight)
        recall = recall_score(y, y_preds[col], sample_weight=sample_weight)
        perc_data_flagged = y_preds[col].mean()
        opt_metric = fbeta_score(
            y, y_preds[col], beta=1, sample_weight=sample_weight)
        assert all(np.array([precision, recall, perc_data_flagged,
                             opt_metric]) == rule_descriptions.loc[col].values)


def test_flatten_stringified_json_column():
    X = pd.DataFrame({
        'sim_ll': [
            json.dumps({"A": 10, "B": -1}),
            json.dumps({"A": 10, "C": -2}),
            json.dumps({"B": -1, "D": -1}),
            json.dumps({"A": 10, "B": -1})
        ]
    })
    expected_X = pd.DataFrame({
        'A': [10, 10, np.nan, 10],
        'B': [-1, np.nan, -1, -1],
        'C': [np.nan, -2, np.nan, np.nan],
        'D': [np.nan, np.nan, -1, np.nan]
    })
    X_flattened = argo_utils.flatten_stringified_json_column(X['sim_ll'])
    assert all(X_flattened == expected_X)


def test_count_rule_conditions():
    rule_strings = {
        "(X['A']>=1.0)|(X['A'].isna())": 2,
        "(X['A']>=2)": 1
    }
    for rule_string, expected_num_conditions in rule_strings.items():
        num_conditions = argo_utils.count_rule_conditions(
            rule_string=rule_string)
        assert num_conditions == expected_num_conditions
