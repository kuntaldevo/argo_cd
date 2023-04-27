import pytest
from rule_filtering.rule_filters import FilterRules, GreedyFilter, FilterCorrelatedRules
from rule_optimisation.optimisation_functions import FScore, AlertsPerDay
from correlation_reduction.similarity_functions import CosineSimilarity, JaccardSimilarity
from correlation_reduction.correlation_reduction_methods import AgglomerativeClusteringFeatureReduction
import argo_utils.argo_utils as argo_utils
import numpy as np
import pandas as pd
import random
from itertools import product


@pytest.fixture
def create_data():
    def return_random_num(y, fraud_min, fraud_max, nonfraud_min, nonfraud_max, rand_func):
        data = [rand_func(fraud_min, fraud_max) if i == 1 else rand_func(
            nonfraud_min, nonfraud_max) for i in y]
        return data

    random.seed(0)
    np.random.seed(0)
    y = pd.Series(data=[0]*980 + [1]*20, index=list(range(0, 1000)))
    X_rules = pd.DataFrame(data={
        "Rule1": [0]*980 + [1]*6 + [0] * 14,
        "Rule2": [0]*987 + [1]*6 + [0] * 7,
        "Rule3": [0]*993 + [1]*6 + [0] * 1,
        "Rule4": [round(max(i, 0)) for i in return_random_num(y, 0.4, 1, 0.5, 0.6, np.random.uniform)],
        "Rule5": [round(max(i, 0)) for i in return_random_num(y, 0.2, 1, 0, 0.6, np.random.uniform)],
    },
        index=list(range(0, 1000))
    )
    weights = y.apply(lambda x: 10 if x == 1 else 1)
    return X_rules, y, weights


@pytest.fixture
def create_data_corr():
    np.random.seed(0)
    X_rules = pd.DataFrame({
        "A": np.random.randint(0, 2, 1000),
        "B": np.random.randint(0, 2, 1000),
        "C": np.random.randint(0, 2, 1000),
        "D": np.random.randint(0, 2, 1000),
        "E": np.random.randint(0, 2, 1000)
    })
    columns_performance = pd.Series(
        [0.1, 0.2, 0.5, 0.4, 0.9],
        ["A", "B", "C", "D", "E"]
    )
    return X_rules, columns_performance


@pytest.fixture
def expected_cols_to_keep_corr():
    expected_results = [
        ['E', 'D'],
        ['E', 'D'],
        ['E'],
        ['E'],
        ['D', 'A', 'B', 'C', 'D', 'E'],
        ['D', 'A', 'B', 'C', 'D', 'E'],
        ['A', 'B', 'C', 'D', 'E'],
        ['A', 'B', 'C', 'D', 'E']
    ]
    return expected_results


@pytest.fixture
def return_rule_descriptions(create_data):
    X_rules, y, weights = create_data
    f4 = FScore(beta=4)
    rd_no_weight = argo_utils.return_binary_pred_perf_of_set_numpy(
        y_true=y, y_preds=X_rules, y_preds_columns=X_rules.columns, opt_func=f4.fit)
    rd_weight = argo_utils.return_binary_pred_perf_of_set_numpy(
        y_true=y, y_preds=X_rules, y_preds_columns=X_rules.columns, sample_weight=weights, opt_func=f4.fit)
    return rd_no_weight, rd_weight


@pytest.fixture
def instantiate_FilterRules(return_rule_descriptions):
    rd_no_weight, rd_weight = return_rule_descriptions
    filters = {
        'Precision': {
            'Operator': '>=',
            'Value': 0.2
        },
        'OptMetric': {
            'Operator': '>=',
            'Value': 0.3
        }
    }
    f4 = FScore(beta=4)
    fr_w_rd_no_weight = FilterRules(
        filters=filters, rule_descriptions=rd_no_weight)
    fr_w_rd_weight = FilterRules(
        filters=filters, rule_descriptions=rd_weight)
    fr_wo_rd_no_weight = FilterRules(filters=filters, opt_func=f4.fit)
    fr_wo_rd_weight = FilterRules(filters=filters, opt_func=f4.fit)
    return fr_w_rd_no_weight, fr_w_rd_weight, fr_wo_rd_no_weight, fr_wo_rd_weight


@pytest.fixture
def instantiate_FilterRules_unlabelled():
    apd = AlertsPerDay(n_alerts_expected_per_day=10,
                       no_of_days_in_file=10)
    filters = {
        'OptMetric': {
            'Operator': '>=',
            'Value': -100
        }
    }
    fr = FilterRules(filters=filters, opt_func=apd.fit)
    rule_descriptions = pd.DataFrame(
        np.array([[-88.36],
                  [-88.36],
                  [-88.36],
                  [-8064.04],
                  [-98.01]]),
        columns=['OptMetric'],
        index=['Rule1', 'Rule2', 'Rule3', 'Rule4', 'Rule5']
    )
    return fr, rule_descriptions


@pytest.fixture
def instantiate_GreedyFilter(return_rule_descriptions):
    rd_no_weight, rd_weight = return_rule_descriptions
    f4 = FScore(beta=4)
    gf_w_rd_no_weight = GreedyFilter(
        opt_func=f4.fit, rule_descriptions=rd_no_weight, sorting_col='Precision', show_plots=False)
    gf_w_rd_weight = GreedyFilter(
        opt_func=f4.fit, rule_descriptions=rd_weight, sorting_col='Precision', show_plots=False)
    gf_wo_rd_no_weight = GreedyFilter(
        opt_func=f4.fit, sorting_col='Precision', show_plots=False)
    gf_wo_rd_weight = GreedyFilter(
        opt_func=f4.fit, sorting_col='Precision', show_plots=False)
    return gf_w_rd_no_weight, gf_w_rd_weight, gf_wo_rd_no_weight, gf_wo_rd_weight


@pytest.fixture
def expected_results_FilterRules(create_data):
    X_rules, _, _ = create_data
    expected_results = [
        X_rules[['Rule1', 'Rule2', 'Rule3']],
        X_rules[['Rule1', 'Rule2', 'Rule3', 'Rule5']]
    ]
    return expected_results


@pytest.fixture
def expected_results_GreedyFilter(create_data):
    X_rules, _, _ = create_data
    expected_results = [
        X_rules[['Rule1', 'Rule2', 'Rule3']],
        X_rules[['Rule1', 'Rule2', 'Rule3', 'Rule5']]
    ]
    return expected_results


@pytest.fixture
def expected_results_return_performance_top_n():
    top_n_no_weight = pd.DataFrame(
        {'Precision': {1: 1.0, 2: 1.0, 3: 1.0, 4: 0.09803921568627451, 5: 0.02},
         'Recall': {1: 0.3, 2: 0.6, 3: 0.9, 4: 1.0, 5: 1.0},
         'PercDataFlagged': {1: 0.006, 2: 0.012, 3: 0.018, 4: 0.204, 5: 1.0},
         'OptMetric': {1: 0.31288343558282206, 2: 0.6144578313253012, 3: 0.9053254437869824, 4: 0.648854961832061, 5: 0.25757575757575757}}
    )
    top_n_weight = pd.DataFrame(
        {'Precision': {1: 1.0, 2: 1.0, 3: 1.0, 4: 0.5208333333333334, 5: 0.1694915254237288},
         'Recall': {1: 0.3, 2: 0.6, 3: 0.9, 4: 1.0, 5: 1.0},
         'PercDataFlagged': {1: 0.006, 2: 0.012, 3: 0.018, 4: 0.204, 5: 1.0},
         'OptMetric': {1: 0.31288343558282206, 2: 0.6144578313253012, 3: 0.9053254437869824, 4: 0.9486607142857143, 5: 0.776255707762557}}
    )
    top_n_no_weight.index.name = 'Top n rules'
    top_n_weight.index.name = 'Top n rules'
    return top_n_no_weight, top_n_weight


class TestFilterRules:

    def test_fit(self, create_data, instantiate_FilterRules, expected_results_FilterRules):
        X_rules, y, weights = create_data
        expected_results_FilterRules = expected_results_FilterRules
        fr_w_rd_no_weight, fr_w_rd_weight, fr_wo_rd_no_weight, fr_wo_rd_weight = instantiate_FilterRules
        # Without weight, with rule_descriptions
        fr_w_rd_no_weight.fit(X_rules=X_rules, y=y)
        assert all(fr_w_rd_no_weight.rules_to_keep ==
                   expected_results_FilterRules[0].columns)
        # Without weight, without rule_descriptions
        fr_wo_rd_no_weight.fit(X_rules=X_rules, y=y)
        assert all(fr_wo_rd_no_weight.rules_to_keep ==
                   expected_results_FilterRules[0].columns)
        # With weight, with rule_descriptions
        fr_w_rd_weight.fit(X_rules=X_rules, y=y, sample_weight=weights)
        assert all(fr_w_rd_weight.rules_to_keep ==
                   expected_results_FilterRules[1].columns)
        # With weight, without rule_descriptions
        fr_wo_rd_weight.fit(X_rules=X_rules, y=y, sample_weight=weights)
        assert all(fr_wo_rd_weight.rules_to_keep ==
                   expected_results_FilterRules[1].columns)

    def test_fit_unlabelled(self, create_data, instantiate_FilterRules_unlabelled):
        X_rules, _, _ = create_data
        # Without rule_descriptions
        fr, _ = instantiate_FilterRules_unlabelled
        fr.fit(X_rules=X_rules)
        assert fr.rules_to_keep == ['Rule1', 'Rule2', 'Rule3', 'Rule5']
        # With rule_descriptions
        fr, rule_descriptions = instantiate_FilterRules_unlabelled
        fr.rule_descriptions = rule_descriptions
        fr.fit(X_rules=X_rules)
        assert fr.rules_to_keep == ['Rule1', 'Rule2', 'Rule3', 'Rule5']

    def test_transform(self, create_data, instantiate_FilterRules):
        X_rules, _, _ = create_data
        fr, _, _, _ = instantiate_FilterRules
        fr.rules_to_keep = ['Rule1']
        X_rules_filtered = fr.transform(X_rules)
        assert (all(X_rules_filtered == X_rules['Rule1'].to_frame()))

    def test_fit_transform(self, create_data, instantiate_FilterRules, expected_results_FilterRules):
        X_rules, y, weights = create_data
        expected_results_FilterRules = expected_results_FilterRules
        fr_w_rd_no_weight, fr_w_rd_weight, fr_wo_rd_no_weight, fr_wo_rd_weight = instantiate_FilterRules
        # Without weight, with rule_descriptions
        X_rules_filtered = fr_w_rd_no_weight.fit_transform(
            X_rules=X_rules, y=y)
        assert all(fr_w_rd_no_weight.rules_to_keep ==
                   expected_results_FilterRules[0].columns)
        assert all(X_rules_filtered == expected_results_FilterRules[0])
        # Without weight, without rule_descriptions
        X_rules_filtered = fr_wo_rd_no_weight.fit_transform(
            X_rules=X_rules, y=y)
        assert all(fr_wo_rd_no_weight.rules_to_keep ==
                   expected_results_FilterRules[0].columns)
        assert all(X_rules_filtered == expected_results_FilterRules[0])
        # With weight, with rule_descriptions
        X_rules_filtered = fr_w_rd_weight.fit_transform(
            X_rules=X_rules, y=y, sample_weight=weights)
        assert all(fr_w_rd_weight.rules_to_keep ==
                   expected_results_FilterRules[1].columns)
        assert all(X_rules_filtered == expected_results_FilterRules[1])
        # With weight, without rule_descriptions
        X_rules_filtered = fr_wo_rd_weight.fit_transform(
            X_rules=X_rules, y=y, sample_weight=weights)
        assert all(fr_wo_rd_weight.rules_to_keep ==
                   expected_results_FilterRules[1].columns)
        assert all(X_rules_filtered == expected_results_FilterRules[1])

    def test_fit_transform_unlabelled(self, create_data,
                                      instantiate_FilterRules_unlabelled,
                                      expected_results_FilterRules):
        X_rules, _, _ = create_data
        _, expected_results_FilterRules = expected_results_FilterRules
        # Without rule_descriptions
        fr, _ = instantiate_FilterRules_unlabelled
        X_rules_filtered = fr.fit_transform(X_rules=X_rules)
        assert fr.rules_to_keep == ['Rule1', 'Rule2', 'Rule3', 'Rule5']
        assert all(X_rules_filtered == expected_results_FilterRules)
        # With rule_descriptions
        fr, rule_descriptions = instantiate_FilterRules_unlabelled
        fr.rule_descriptions = rule_descriptions
        X_rules_filtered = fr.fit_transform(X_rules=X_rules)
        assert fr.rules_to_keep == ['Rule1', 'Rule2', 'Rule3', 'Rule5']
        assert all(X_rules_filtered == expected_results_FilterRules)

    def test_iterate_rule_descriptions(self, instantiate_FilterRules, expected_results_FilterRules):
        expected_results_FilterRules = expected_results_FilterRules
        fr_w_rd_no_weight, fr_w_rd_weight, _, _ = instantiate_FilterRules
        for i, fr in enumerate([fr_w_rd_no_weight, fr_w_rd_weight]):
            rules_to_keep = fr._iterate_rule_descriptions(
                fr.rule_descriptions, fr.filters)
            assert all([a == b for a, b in zip(
                rules_to_keep, expected_results_FilterRules[i])])


class TestGreedyFilter:

    def test_fit(self, create_data, instantiate_GreedyFilter, expected_results_GreedyFilter):
        X_rules, y, weights = create_data
        expected_results_GreedyFilter = expected_results_GreedyFilter
        gf_w_rd_no_weight, gf_w_rd_weight, gf_wo_rd_no_weight, gf_wo_rd_weight = instantiate_GreedyFilter
        # Without weight, with rule_descriptions
        gf_w_rd_no_weight.fit(X_rules=X_rules, y=y)
        assert all(gf_w_rd_no_weight.rules_to_keep ==
                   expected_results_GreedyFilter[0].columns)
        # Without weight, without rule_descriptions
        gf_wo_rd_no_weight.fit(X_rules=X_rules, y=y)
        assert all(gf_wo_rd_no_weight.rules_to_keep ==
                   expected_results_GreedyFilter[0].columns)
        # With weight, with rule_descriptions
        gf_w_rd_weight.fit(X_rules=X_rules, y=y, sample_weight=weights)
        assert all(gf_w_rd_weight.rules_to_keep ==
                   expected_results_GreedyFilter[1].columns)
        # With weight, without rule_descriptions
        gf_wo_rd_weight.fit(X_rules=X_rules, y=y, sample_weight=weights)
        assert all(gf_wo_rd_weight.rules_to_keep ==
                   expected_results_GreedyFilter[1].columns)

    def test_transform(self, create_data, instantiate_GreedyFilter):
        X_rules, _, _ = create_data
        gf, _, _, _ = instantiate_GreedyFilter
        gf.rules_to_keep = ['Rule1']
        X_rules_filtered = gf.transform(X_rules)
        assert (all(X_rules_filtered == X_rules['Rule1'].to_frame()))

    def test_fit_transform(self, create_data, instantiate_GreedyFilter, expected_results_GreedyFilter):
        X_rules, y, weights = create_data
        expected_results_GreedyFilter = expected_results_GreedyFilter
        gf_w_rd_no_weight, gf_w_rd_weight, gf_wo_rd_no_weight, gf_wo_rd_weight = instantiate_GreedyFilter
        # Without weight, with rule_descriptions
        X_rules_filtered = gf_w_rd_no_weight.fit_transform(
            X_rules=X_rules, y=y)
        assert all(gf_w_rd_no_weight.rules_to_keep ==
                   expected_results_GreedyFilter[0].columns)
        assert all(X_rules_filtered == expected_results_GreedyFilter[0])
        # Without weight, without rule_descriptions
        X_rules_filtered = gf_wo_rd_no_weight.fit_transform(
            X_rules=X_rules, y=y)
        assert all(gf_wo_rd_no_weight.rules_to_keep ==
                   expected_results_GreedyFilter[0].columns)
        assert all(X_rules_filtered == expected_results_GreedyFilter[0])
        # With weight, with rule_descriptions
        X_rules_filtered = gf_w_rd_weight.fit_transform(
            X_rules=X_rules, y=y, sample_weight=weights)
        assert all(gf_w_rd_weight.rules_to_keep ==
                   expected_results_GreedyFilter[1].columns)
        assert all(X_rules_filtered == expected_results_GreedyFilter[1])
        # With weight, without rule_descriptions
        X_rules_filtered = gf_wo_rd_weight.fit_transform(
            X_rules=X_rules, y=y, sample_weight=weights)
        assert all(gf_wo_rd_weight.rules_to_keep ==
                   expected_results_GreedyFilter[1].columns)
        assert all(X_rules_filtered == expected_results_GreedyFilter[1])

    def test_return_performance_top_n(self, create_data, instantiate_GreedyFilter, expected_results_return_performance_top_n):
        X_rules, y, weights = create_data
        gf_w_rd_no_weight, gf_w_rd_weight, _, _ = instantiate_GreedyFilter
        expected_results = expected_results_return_performance_top_n
        f4 = FScore(beta=4)
        for i, (gf, w) in enumerate(zip([gf_w_rd_no_weight, gf_w_rd_weight], [None, weights])):
            top_n_rule_descriptions = gf._return_performance_top_n(
                gf.rule_descriptions, X_rules, y, w, f4.fit)
            assert all(top_n_rule_descriptions == expected_results[i])

    def test_return_top_rules_by_opt_func(self, instantiate_GreedyFilter, expected_results_return_performance_top_n, expected_results_GreedyFilter, return_rule_descriptions):
        gf, _, _, _ = instantiate_GreedyFilter
        top_n_list = expected_results_return_performance_top_n
        rule_descriptions_list = return_rule_descriptions
        rule_descriptions_list = [rd.sort_values('Precision', ascending=False)
                                  for rd in rule_descriptions_list]
        expected_results = expected_results_GreedyFilter
        for i, (top_n, rule_descriptions) in enumerate(zip(top_n_list, rule_descriptions_list)):
            rules_to_keep = gf._return_top_rules_by_opt_func(
                top_n, rule_descriptions)
            assert all([a == b for a, b in zip(
                rules_to_keep, expected_results[i])])


class TestFilterCorrelatedRules:

    def test_fit(self, create_data_corr, expected_cols_to_keep_corr):
        X_rules, columns_performance = create_data_corr
        expected_results = expected_cols_to_keep_corr
        cs = CosineSimilarity()
        js = JaccardSimilarity()
        combinations = list(
            product([0.25, 0.75], ['top_down', 'bottom_up'], [cs.fit, js.fit]))
        for i, (threshold, strategy, similarity_function) in enumerate(combinations):
            crc = AgglomerativeClusteringFeatureReduction(threshold=threshold,
                                                          strategy=strategy, similarity_function=similarity_function,
                                                          columns_performance=columns_performance)
            fr = FilterCorrelatedRules(
                correlation_reduction_class=crc, rule_descriptions=columns_performance)
            fr.fit(X_rules=X_rules)
            assert fr.rules_to_keep == expected_results[i]

    def test_transform(self, create_data_corr, expected_cols_to_keep_corr):
        X_rules, columns_performance = create_data_corr
        expected_results = expected_cols_to_keep_corr
        cs = CosineSimilarity()
        js = JaccardSimilarity()
        combinations = list(
            product([0.25, 0.75], ['top_down', 'bottom_up'], [cs.fit, js.fit]))
        for i, (threshold, strategy, similarity_function) in enumerate(combinations):
            crc = AgglomerativeClusteringFeatureReduction(threshold=threshold,
                                                          strategy=strategy, similarity_function=similarity_function,
                                                          columns_performance=columns_performance)
            fr = FilterCorrelatedRules(
                correlation_reduction_class=crc, rule_descriptions=columns_performance)
            fr.fit(X_rules=X_rules)
            X_rules_filtered = fr.transform(X_rules)
            assert all(X_rules_filtered == X_rules[expected_results[i]])
            assert all(fr.rule_descriptions ==
                       columns_performance.loc[expected_results[i]])

    def test_fit_transform(self, create_data_corr, expected_cols_to_keep_corr):
        X_rules, columns_performance = create_data_corr
        expected_results = expected_cols_to_keep_corr
        cs = CosineSimilarity()
        js = JaccardSimilarity()
        combinations = list(
            product([0.25, 0.75], ['top_down', 'bottom_up'], [cs.fit, js.fit]))
        for i, (threshold, strategy, similarity_function) in enumerate(combinations):
            crc = AgglomerativeClusteringFeatureReduction(threshold=threshold,
                                                          strategy=strategy, similarity_function=similarity_function,
                                                          columns_performance=columns_performance)
            fr = FilterCorrelatedRules(
                correlation_reduction_class=crc, rule_descriptions=columns_performance)
            X_rules_filtered = fr.fit_transform(X_rules=X_rules)
            assert all(X_rules_filtered == X_rules[expected_results[i]])
            assert all(fr.rule_descriptions ==
                       columns_performance.loc[expected_results[i]])
