import pytest
import numpy as np
import pandas as pd
from rule_generation.rule_generator_opt import RuleGeneratorOpt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, fbeta_score
from itertools import product
import random
import math


@pytest.fixture
def create_data():
    def return_random_num(y, fraud_min, fraud_max, nonfraud_min, nonfraud_max, rand_func):
        data = [rand_func(fraud_min, fraud_max) if i == 1 else rand_func(
            nonfraud_min, nonfraud_max) for i in y]
        return data

    random.seed(0)
    np.random.seed(0)
    y = pd.Series(data=[0]*980 + [1]*20, index=list(range(0, 1000)))
    X = pd.DataFrame(data={
        "num_distinct_txn_per_email_1day": [round(max(i, 0)) for i in return_random_num(y, 2, 1, 1, 2, np.random.normal)],
        "num_distinct_txn_per_email_7day": [round(max(i, 0)) for i in return_random_num(y, 4, 2, 2, 3, np.random.normal)],
        "ip_country_us": [round(min(i, 1)) for i in [max(i, 0) for i in return_random_num(y, 0.3, 0.4, 0.5, 0.5, np.random.normal)]],
        "email_kb_distance": [min(i, 1) for i in [max(i, 0) for i in return_random_num(y, 0.2, 0.5, 0.6, 0.4, np.random.normal)]],
        "email_alpharatio":  [min(i, 1) for i in [max(i, 0) for i in return_random_num(y, 0.33, 0.1, 0.5, 0.2, np.random.normal)]],
    },
        index=list(range(0, 1000))
    )
    columns_int = [
        'num_distinct_txn_per_email_1day', 'num_distinct_txn_per_email_7day', 'ip_country_us']
    columns_cat = ['ip_country_us']
    columns_num = ['num_distinct_txn_per_email_1day',
                   'num_distinct_txn_per_email_7day', 'email_kb_distance', 'email_alpharatio']
    weights = y.apply(lambda x: 1000 if x == 1 else 1)
    return [X, y, columns_int, columns_cat, columns_num, weights]


@pytest.fixture
def create_smaller_data():
    random.seed(0)
    np.random.seed(0)
    y = pd.Series(data=[0]*5 + [1]*5, index=list(range(0, 10)))
    X = pd.DataFrame(data={
        'A': [5, 0, 5, 0, 5, 3, 4, 0, 0, 0],
        'B': [0, 1, 0, 1, 0, 1, 0.6, 0.7, 0, 0],
        'C_US': [1, 1, 1, 1, 1, 1, 0, 0, 1, 1]
    },
        index=list(range(0, 10))
    )
    columns_int = ['A']
    columns_cat = ['C']
    columns_num = ['A', 'B']
    weights = y.apply(lambda x: 1000 if x == 1 else 1)
    return [X, y, columns_int, columns_cat, columns_num, weights]


@pytest.fixture
def fs_instantiated():
    class FScore:
        def __init__(self, beta):
            self.beta = beta

        def fit(self, y_true, y_pred, sample_weight):
            return fbeta_score(y_true=y_true, y_pred=y_pred, beta=self.beta, sample_weight=sample_weight, zero_division=0)
    f = FScore(0.5)
    return f.fit


@pytest.fixture
def rg_instantiated(fs_instantiated):
    f0dot5 = fs_instantiated
    params = {
        'opt_func': f0dot5,
        'n_total_conditions': 4,
        'num_rules_keep': 50,
        'n_points': 10,
        'ratio_window': 2,
        'remove_corr_rules': False
    }
    rg = RuleGeneratorOpt(**params)
    rg.today = '20200204'
    return [rg, params]


@pytest.fixture
def return_dummy_rules():
    def _read(weight_is_none=True):
        if weight_is_none:
            rule_descriptions = pd.DataFrame(
                np.array([["(X['B']>=0.5)", 0.6, 0.6, 1, 0.5, 0.6],
                          ["(X['C_US']==True)", 0.375, 0.6,
                           1, 0.8, 0.4054054054054054],
                          ["(X['A']>=3)", 0.4, 0.4, 1, 0.5, 0.4000000000000001]]),
                columns=['Logic', 'Precision', 'Recall',
                         'nConditions', 'PercDataFlagged', 'OptMetric'],
                index=['RGO_Rule_20200204_1',
                       'RGO_Rule_20200204_2', 'RGO_Rule_20200204_0'],
            )
            rule_descriptions = rule_descriptions.astype({'Logic': object, 'Precision': float, 'Recall': float,
                                                          'nConditions': int, 'PercDataFlagged': float, 'OptMetric': float})
            rule_descriptions.index.name = 'Rule'
        else:
            rule_descriptions = pd.DataFrame(
                np.array([["(X['B']>=0.5)", 0.9993337774816788, 0.6, 1, 0.5,
                           0.8819379115710255],
                          ["(X['C_US']==True)", 0.9983361064891847, 0.6, 1, 0.8,
                           0.8813160987074031],
                          ["(X['A']>=3)", 0.9985022466300549, 0.4, 1, 0.5,
                           0.7685213648939442]]),
                columns=['Logic', 'Precision', 'Recall',
                         'nConditions', 'PercDataFlagged', 'OptMetric'],
                index=['RGO_Rule_20200204_1',
                       'RGO_Rule_20200204_2', 'RGO_Rule_20200204_0'],
            )
            rule_descriptions = rule_descriptions.astype({'Logic': object, 'Precision': float, 'Recall': float,
                                                          'nConditions': int, 'PercDataFlagged': float, 'OptMetric': float})
            rule_descriptions.index.name = 'Rule'
        X_rules = pd.DataFrame(
            np.array([[0, 1, 1],
                      [1, 1, 0],
                      [0, 1, 1],
                      [1, 1, 0],
                      [0, 1, 1],
                      [1, 1, 1],
                      [1, 0, 1],
                      [1, 0, 0],
                      [0, 1, 0],
                      [0, 1, 0]], dtype=np.int),
            columns=['RGO_Rule_20200204_1',
                     'RGO_Rule_20200204_2', 'RGO_Rule_20200204_0'],
        )
        rule_combinations = [(('RGO_Rule_20200204_1', 'RGO_Rule_20200204_2'), ("(X['B']>=0.5)", "(X['C_US']==True)")),
                             (('RGO_Rule_20200204_1', 'RGO_Rule_20200204_0'),
                              ("(X['B']>=0.5)", "(X['A']>=3)")),
                             (('RGO_Rule_20200204_2', 'RGO_Rule_20200204_0'), ("(X['C_US']==True)", "(X['A']>=3)"))]
        return rule_descriptions, X_rules, rule_combinations
    return _read


@pytest.fixture
def return_dummy_pairwise_rules():
    rule_descriptions = pd.DataFrame(
        {
            'Rule': ['A', 'B', 'C'],
            'Precision': [1, 0.5, 0]
        }
    )
    rule_descriptions.set_index('Rule', inplace=True)
    pairwise_descriptions = pd.DataFrame(
        {
            'Rule': ['A&B', 'B&C', 'A&C'],
            'Precision': [1, 0.75, 0]
        }
    )
    pairwise_descriptions.set_index('Rule', inplace=True)
    X_rules_pairwise = pd.DataFrame({
        'A&B': range(0, 1000),
        'B&C': range(0, 1000),
        'A&C': range(0, 1000),
    })
    pairwise_to_orig_lookup = {
        'A&B': ['A', 'B'],
        'A&C': ['A', 'C'],
        'B&C': ['B', 'C'],
    }
    return pairwise_descriptions, X_rules_pairwise, pairwise_to_orig_lookup, rule_descriptions


@pytest.fixture
def return_iteration_results():
    iteration_ranges = {
        ('num_distinct_txn_per_email_1day', '>='): (0, 7),
        ('num_distinct_txn_per_email_1day', '<='): (0, 7),
        ('num_distinct_txn_per_email_7day', '>='): (6.0, 12),
        ('num_distinct_txn_per_email_7day', '<='): (0, 6.0),
        ('email_kb_distance', '>='): (0.5, 1.0),
        ('email_kb_distance', '<='): (0.0, 0.5),
        ('email_alpharatio', '>='): (0.5, 1.0),
        ('email_alpharatio', '<='): (0.0, 0.5)
    }
    iteration_arrays = {('num_distinct_txn_per_email_1day', '>='): np.array([0, 1, 2, 3, 4, 5, 6, 7]),
                        ('num_distinct_txn_per_email_1day', '<='): np.array([0, 1, 2, 3, 4, 5, 6, 7]),
                        ('num_distinct_txn_per_email_7day',
                         '>='): np.array([6,  7,  8,  9, 10, 11, 12]),
                        ('num_distinct_txn_per_email_7day', '<='): np.array([0, 1, 2, 3, 4, 5, 6]),
                        ('email_kb_distance',
                         '>='): np.array([0.5, 0.56, 0.61, 0.67, 0.72, 0.78, 0.83, 0.89, 0.94, 1.]),
                        ('email_kb_distance',
                         '<='): np.array([0., 0.056, 0.11, 0.17, 0.22, 0.28, 0.33, 0.39, 0.44,
                                          0.5]),
                        ('email_alpharatio',
                         '>='): np.array([0.5, 0.56, 0.61, 0.67, 0.72, 0.78, 0.83, 0.89, 0.94, 1.]),
                        ('email_alpharatio',
                         '<='): np.array([0., 0.056, 0.11, 0.17, 0.22, 0.28, 0.33, 0.39, 0.44,
                                          0.5])}
    fscore_arrays = {('num_distinct_txn_per_email_1day',
                      '>='): np.array([0.02487562, 0.04244482, 0.05393401, 0.01704545, 0.,
                                       0., 0., 0.]),
                     ('num_distinct_txn_per_email_1day',
                      '<='): np.array([0., 0.00608766, 0.02689873, 0.0275634, 0.02590674,
                                       0.02520161, 0.0249004, 0.02487562]),
                     ('num_distinct_txn_per_email_7day',
                      '>='): np.array([0.0304878, 0.04934211, 0., 0., 0.,
                                       0., 0.]),
                     ('num_distinct_txn_per_email_7day',
                      '<='): np.array([0., 0.00903614, 0.01322751, 0.01623377, 0.0248139,
                                       0.02395716, 0.02275161]),
                     ('email_kb_distance',
                      '>='): np.array([0.01290878, 0.01420455, 0.01588983, 0.01509662, 0.0136612,
                                       0.01602564, 0.01798561, 0.0210084, 0.0245098, 0.0154321]),
                     ('email_kb_distance',
                      '<='): np.array([0.10670732, 0.08333333, 0.06835938, 0.06410256, 0.06048387,
                                       0.05901288, 0.05474453, 0.04573171, 0.04298942, 0.04079254]),
                     ('email_alpharatio',
                      '>='): np.array([0.00498008, 0.00327225, 0., 0., 0.,
                                       0., 0., 0., 0., 0.]),
                     ('email_alpharatio',
                      '<='): np.array([0., 0., 0., 0.02232143, 0.04310345,
                                       0.06161972, 0.06157635, 0.05662021, 0.05712366, 0.04429134])}
    return iteration_ranges, iteration_arrays, fscore_arrays


@pytest.fixture
def return_pairwise_info_dict():
    pairwise_info_dict = {"(X['B']>=0.5)&(X['C_US']==True)": {'RuleName1': 'RGO_Rule_20200204_1',
                                                              'RuleName2': 'RGO_Rule_20200204_2',
                                                              'PairwiseRuleName': 'RGO_Rule_20200204_0',
                                                              'PairwiseComponents': ['RGO_Rule_20200204_1', 'RGO_Rule_20200204_2']},
                          "(X['A']>=3)&(X['B']>=0.5)": {'RuleName1': 'RGO_Rule_20200204_1',
                                                        'RuleName2': 'RGO_Rule_20200204_0',
                                                        'PairwiseRuleName': 'RGO_Rule_20200204_1',
                                                        'PairwiseComponents': ['RGO_Rule_20200204_1', 'RGO_Rule_20200204_0']},
                          "(X['A']>=3)&(X['C_US']==True)": {'RuleName1': 'RGO_Rule_20200204_2',
                                                            'RuleName2': 'RGO_Rule_20200204_0',
                                                            'PairwiseRuleName': 'RGO_Rule_20200204_2',
                                                            'PairwiseComponents': ['RGO_Rule_20200204_2', 'RGO_Rule_20200204_0']}}
    # pairwise_info_dict = {"(X['B']>=0.5)&(X['C_US']==True)": {'RuleName1': 'RGO_Rule_20200204__20200204_1',
    #                                                           'RuleName2': 'RGO_Rule_20200204__20200204_2',
    #                                                           'PairwiseRuleName': 'RGO_Rule_20200204__20200204_0',
    #                                                           'PairwiseComponents': ['RGO_Rule_20200204__20200204_1', 'RGO_Rule_20200204__20200204_2']},
    #                       "(X['A']>=3)&(X['B']>=0.5)": {'RuleName1': 'RGO_Rule_20200204__20200204_1',
    #                                                     'RuleName2': 'RGO_Rule_20200204__20200204_0',
    #                                                     'PairwiseRuleName': 'RGO_Rule_20200204__20200204_1',
    #                                                     'PairwiseComponents': ['RGO_Rule_20200204__20200204_1', 'RGO_Rule_20200204__20200204_0']},
    #                       "(X['A']>=3)&(X['C_US']==True)": {'RuleName1': 'RGO_Rule_20200204__20200204_2',
    #                                                         'RuleName2': 'RGO_Rule_20200204__20200204_0',
    #                                                         'PairwiseRuleName': 'RGO_Rule_20200204__20200204_2',
    #                                                         'PairwiseComponents': ['RGO_Rule_20200204__20200204_2', 'RGO_Rule_20200204__20200204_0']}}
    return pairwise_info_dict


def test_fit(create_data, rg_instantiated, fs_instantiated):

    def _assert_test_fit(rule_descriptions_shape, X_rules_shape):
        X_rules = rg.fit(X, y, sample_weight=w)
        _assert_rule_descriptions_and_X_rules(
            rg.rule_descriptions, X_rules, rule_descriptions_shape, X_rules_shape, X, y, opt_func, w)

    X, y, _, _, _, weights = create_data
    rg, _ = rg_instantiated
    opt_func = fs_instantiated
    for w in [None, weights]:
        _assert_test_fit((86, 6), (1000, 86)) if w is None else _assert_test_fit(
            (59, 6), (1000, 59))


def test_apply(create_data, rg_instantiated, fs_instantiated):

    def _assert_test_apply(rule_descriptions_shape, X_rules_shape):
        _ = rg.fit(X, y, sample_weight=w)
        X_rules_applied = rg.apply(X, y, w)
        _assert_rule_descriptions_and_X_rules(
            rg.rule_descriptions_applied, X_rules_applied, rule_descriptions_shape, X_rules_shape, X, y, opt_func, w)

    X, y, _, _, _, weights = create_data
    rg, _ = rg_instantiated
    opt_func = fs_instantiated
    for w in [None, weights]:
        _assert_test_apply((86, 6), (1000, 86)) if w is None else _assert_test_apply(
            (59, 6), (1000, 59))


def test_generate_numeric_one_condition_rules(create_data, rg_instantiated, fs_instantiated):
    X, y, columns_int, _, columns_num, weights = create_data
    rg, _ = rg_instantiated
    opt_func = fs_instantiated
    for w in [None, weights]:
        rule_descriptions, X_rules = rg._generate_numeric_one_condition_rules(
            X, y, columns_num, columns_int, w)
        _assert_rule_descriptions_and_X_rules(
            rule_descriptions, X_rules, (8, 6), (1000, 8), X, y, opt_func, w)


def test_generate_categorical_one_condition_rules(create_data, rg_instantiated, fs_instantiated):
    X, y, _, columns_cat, _, weights = create_data
    rg, _ = rg_instantiated
    opt_func = fs_instantiated
    for w in [None, weights]:
        rule_descriptions, X_rules = rg._generate_categorical_one_condition_rules(
            X, y, columns_cat, w)
        _assert_rule_descriptions_and_X_rules(
            rule_descriptions, X_rules, (1, 6), (1000, 1), X, y, opt_func, w)


def test_generate_pairwise_rules(return_dummy_rules, create_smaller_data, rg_instantiated,
                                 fs_instantiated):

    def _assert_generate_pairwise_rules(rule_descriptions_shape, X_rules_shape):
        rule_descriptions, X_rules, rule_combinations = return_dummy_rules()
        rule_descriptions, X_rules, _ = rg._generate_pairwise_rules(
            rule_descriptions, X_rules, y, rule_combinations, w)
        _assert_rule_descriptions_and_X_rules(
            rule_descriptions, X_rules, rule_descriptions_shape, X_rules_shape, X, y, opt_func, w)

    X, y, _, _, _, weights = create_smaller_data
    rg, _ = rg_instantiated
    opt_func = fs_instantiated
    for w in [None, weights]:
        _assert_generate_pairwise_rules(
            (3, 6), (10, 3)) if w is None else _assert_generate_pairwise_rules(
                (3, 6), (10, 3))


def test_drop_unnecessary_pairwise_rules(rg_instantiated, return_dummy_pairwise_rules):
    rg, _ = rg_instantiated
    args = return_dummy_pairwise_rules
    pairwise_descriptions, X_rules_pairwise = rg._drop_unnecessary_pairwise_rules(
        *args)
    print(pairwise_descriptions.shape)
    assert pairwise_descriptions.shape == (1, 1)
    assert X_rules_pairwise.shape == (1000, 1)


def test_generate_n_order_pairwise_rules(return_dummy_rules, create_smaller_data, rg_instantiated,
                                         fs_instantiated):

    def _assert_generate_n_order_pairwise_rules(rule_descriptions_shape, X_rules_shape):
        rule_descriptions, X_rules, _ = return_dummy_rules(w is None)
        print(rule_descriptions, X_rules, y, rem_corr_rules)
        rule_descriptions, X_rules = rg._generate_n_order_pairwise_rules(
            rule_descriptions, X_rules, y, rem_corr_rules, w)
        _assert_rule_descriptions_and_X_rules(
            rule_descriptions, X_rules, rule_descriptions_shape, X_rules_shape, X, y, opt_func, w)

    shape_lookup = {
        (True, True): ((4, 6), (10, 4)),
        (True, False): ((4, 6), (10, 4)),
        (False, True): ((4, 6), (10, 4)),
        (False, False): ((4, 6), (10, 4))
    }
    X, y, _, _, _, weights = create_smaller_data
    rg, _ = rg_instantiated
    opt_func = fs_instantiated
    rg._rule_name_counter = 3
    for rem_corr_rules, w in list(product([True, False], [None, weights])):
        rule_descriptions_shape, X_rules_shape = shape_lookup[(
            rem_corr_rules, w is None)]
        _assert_generate_n_order_pairwise_rules(
            rule_descriptions_shape, X_rules_shape)


def test_set_iteration_range(create_data, rg_instantiated, return_iteration_results):
    iteration_ranges, _, _ = return_iteration_results
    X, _, columns_int, _, columns_num, _ = create_data
    rg, _ = rg_instantiated
    for col, op in product(columns_num, ['>=', '<=']):
        assert iteration_ranges[(col, op)] == rg._set_iteration_range(
            X[col].values, col, op, 10, 2, columns_int)


def test_set_iteration_array(create_data, rg_instantiated, return_iteration_results):
    iteration_ranges, iteration_arrays, _ = return_iteration_results
    _, _, columns_int, _, columns_num, _ = create_data
    rg, _ = rg_instantiated
    for col, op in product(columns_num, ['>=', '<=']):
        iter_min = iteration_ranges[(col, op)][0]
        iter_max = iteration_ranges[(col, op)][1]
        expected_array = iteration_arrays[(col, op)]
        actual_array = rg._set_iteration_array(
            col, columns_int, iter_min, iter_max, 10, rg._round_to_n_sf)
        assert [a == b for a, b in zip(expected_array, actual_array)]


def test_calculate_opt_metric_across_range(create_data, rg_instantiated, return_iteration_results,
                                           fs_instantiated):
    _, iteration_arrays, fscore_arrays = return_iteration_results
    X, y, _, _, _, _ = create_data
    rg, _ = rg_instantiated
    opt_func = fs_instantiated
    for (col, op), x_array in iteration_arrays.items():
        expected_array = fscore_arrays[(col, op)]
        actual_array = rg._calculate_opt_metric_across_range(
            x_array, op, X[col], y, opt_func, None)
        calculated_array = []
        for x in x_array:
            fscore = opt_func(y_true=y, y_pred=eval(
                f'X["{col}"]{op}{x}'), sample_weight=None)
            calculated_array.append(fscore)
        assert [a == b for a, b in zip(expected_array, actual_array)]
        assert [a == b for a, b in zip(calculated_array, actual_array)]


def test_round_to_n_sf(rg_instantiated):
    rg, _ = rg_instantiated
    x = 0.0001234
    assert 0.00012 == rg._round_to_n_sf(x, 2)
    x = 0
    assert 0 == rg._round_to_n_sf(x, 2)


def test_get_rule_combinations_for_loop(return_dummy_rules, rg_instantiated):
    rule_descriptions, _, expected_rule_comb = return_dummy_rules()
    rg, _ = rg_instantiated
    actual_rule_comb = rg._get_rule_combinations_for_loop(
        rule_descriptions, 1, 50)
    assert [a == b for a, b in zip(expected_rule_comb, actual_rule_comb)]
    num_rules = rule_descriptions.shape[0]
    assert len(actual_rule_comb) == math.factorial(num_rules) / \
        (math.factorial(2) * math.factorial(num_rules - 2))


def test_return_pairwise_information(return_pairwise_info_dict, return_dummy_rules, rg_instantiated):
    expected_pw_info_dict = return_pairwise_info_dict
    _, _, rule_comb = return_dummy_rules()
    rg, _ = rg_instantiated
    actual_pw_info_dict = rg._return_pairwise_information(rule_comb)
    assert actual_pw_info_dict == expected_pw_info_dict


def test_generate_pairwise_df(return_dummy_rules, return_pairwise_info_dict, rg_instantiated):
    _, X_rules, _ = return_dummy_rules()
    expected_pw_info_dict = return_pairwise_info_dict
    rules_names_1, rules_names_2, pairwise_names = [], [], []
    for info_dict in expected_pw_info_dict.values():
        rules_names_1.append(info_dict['RuleName1'])
        rules_names_2.append(info_dict['RuleName2'])
        pairwise_names.append(info_dict['PairwiseRuleName'])
    rg, _ = rg_instantiated
    X_rules_pw = rg._generate_pairwise_df(
        X_rules, rules_names_1, rules_names_2, pairwise_names)
    assert X_rules_pw.shape == (10, 3)
    assert [a == b for a, b in zip(X_rules_pw.columns, expected_pw_info_dict)]
    for _, info_dict in expected_pw_info_dict.items():
        X_rule_pw = X_rules_pw[info_dict['PairwiseRuleName']]
        X_rule_pw_calc = X_rules[info_dict['RuleName1']
                                 ] * X_rules[info_dict['RuleName2']]
        assert [a == b for a, b in zip(X_rule_pw, X_rule_pw_calc)]


def test_return_pairwise_rules_to_drop(rg_instantiated, return_dummy_pairwise_rules):
    rg, _ = rg_instantiated
    pairwise_descriptions, _, pairwise_to_orig_lookup, rule_descriptions = return_dummy_pairwise_rules
    assert rg._return_pairwise_rules_to_drop(
        pairwise_descriptions, pairwise_to_orig_lookup, rule_descriptions) == ['A&B', 'A&C']


def test_generate_rule_name(rg_instantiated):
    rg, _ = rg_instantiated
    rule_name = rg._generate_rule_name()
    assert rule_name == 'RGO_Rule_20200204_0'


def _calc_rule_metrics(rule, X, y, opt_func, sample_weight):
    X_rule = eval(rule).astype(int)
    prec = precision_score(
        y, X_rule, sample_weight=sample_weight, zero_division=0)
    rec = recall_score(y, X_rule, sample_weight=sample_weight,
                       zero_division=0)
    opt_value = opt_func(y, X_rule, sample_weight=sample_weight)
    perc_data_flagged = X_rule.mean()
    return [prec, rec, perc_data_flagged, opt_value, X_rule]


def _assert_rule_descriptions(rule_descriptions, X, y, opt_func, sample_weight):
    for _, row in rule_descriptions.iterrows():
        class_results = row.loc[['Precision', 'Recall',
                                 'PercDataFlagged', 'OptMetric']].values.astype(float)
        rule = row['Logic']
        test_results = _calc_rule_metrics(rule, X, y, opt_func, sample_weight)
        for i in range(0, len(class_results)):
            assert round(class_results[i], 6) == round(test_results[i], 6)


def _assert_X_rules(X_rules, rule_list, X, y, opt_func, sample_weight):
    for rule, rule_name in zip(rule_list, X_rules):
        class_result = X_rules[rule_name]
        test_result = _calc_rule_metrics(
            rule, X, y, opt_func, sample_weight)[-1]
        assert [a == b for a, b in zip(class_result, test_result)]


def _assert_rule_descriptions_and_X_rules(rule_descriptions, X_rules, rule_descriptions_shape,
                                          X_rules_shape, X, y, opt_func, sample_weight):
    assert rule_descriptions.shape == rule_descriptions_shape
    assert X_rules.shape == X_rules_shape
    _assert_rule_descriptions(rule_descriptions, X, y, opt_func, sample_weight)
    _assert_X_rules(
        X_rules, rule_descriptions['Logic'].values, X, y, opt_func, sample_weight)
