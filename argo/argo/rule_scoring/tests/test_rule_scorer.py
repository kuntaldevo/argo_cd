from itertools import product
import pytest
import numpy as np
import pandas as pd
import rule_scoring.rule_scoring_methods as rsm
import rule_scoring.rule_score_scalers as rss
from rule_scoring.rule_scorer import RuleScorer
from sklearn.metrics import precision_score


@pytest.fixture
def create_data():
    np.random.seed(0)
    X_rules = pd.DataFrame({
        'A': np.random.randint(0, 2, 1000),
        'B': np.random.randint(0, 2, 1000),
        'C': np.random.randint(0, 2, 1000),
    })
    y = pd.Series(np.random.randint(0, 2, 1000))
    weights = (y + 1) * 2
    return X_rules, y, weights


@pytest.fixture
def expected_results():
    expected_results = {
        ('LR', 'MMS(-100, -10)', 'No weights'): pd.Series({'A': -10.0, 'B': -100.0, 'C': -32.0}),
        ('LR', 'MMS(100, 10)', 'No weights'): pd.Series({'A': 10.0, 'B': 100.0, 'C': 32.0}),
        ('LR', 'CS(-100)', 'No weights'): pd.Series({'A': -69.0, 'B': -100.0, 'C': -77.0}),
        ('LR', 'CS(100)', 'No weights'): pd.Series({'A': 69.0, 'B': 100.0, 'C': 77.0}),
        ('PS', 'MMS(-100, -10)', 'No weights'): pd.Series({'A': -10.0, 'B': -100.0, 'C': -36.0}),
        ('PS', 'MMS(100, 10)', 'No weights'): pd.Series({'A': 10.0, 'B': 100.0, 'C': 36.0}),
        ('PS', 'CS(-100)', 'No weights'): pd.Series({'A': -91.0, 'B': -100.0, 'C': -94.0}),
        ('PS', 'CS(100)', 'No weights'): pd.Series({'A': 91.0, 'B': 100.0, 'C': 94.0}),
        ('RFS', 'MMS(-100, -10)', 'No weights'): pd.Series({'A': -100.0, 'B': -26.0, 'C': -10.0}),
        ('RFS', 'MMS(100, 10)', 'No weights'): pd.Series({'A': 100.0, 'B': 26.0, 'C': 10.0}),
        ('RFS', 'CS(-100)', 'No weights'): pd.Series({'A': -100.0, 'B': -73.0, 'C': -68.0}),
        ('RFS', 'CS(100)', 'No weights'): pd.Series({'A': 100.0, 'B': 73.0, 'C': 68.0}),
        ('LR', 'MMS(-100, -10)', 'Weights'): pd.Series({'A': -10, 'B': -100, 'C': -31}),
        ('LR', 'MMS(100, 10)', 'Weights'): pd.Series({'A': 10, 'B': 100, 'C': 31}),
        ('LR', 'CS(-100)', 'Weights'): pd.Series({'A': -69, 'B': -100, 'C': -76}),
        ('LR', 'CS(100)', 'Weights'): pd.Series({'A': 69, 'B': 100, 'C': 76}),
        ('PS', 'MMS(-100, -10)', 'Weights'): pd.Series({'A': -10, 'B': -100, 'C': -36}),
        ('PS', 'MMS(100, 10)', 'Weights'): pd.Series({'A': 10, 'B': 100, 'C': 36}),
        ('PS', 'CS(-100)', 'Weights'): pd.Series({'A': -94, 'B': -100, 'C': -96}),
        ('PS', 'CS(100)', 'Weights'): pd.Series({'A': 94, 'B': 100, 'C': 96}),
        ('RFS', 'MMS(-100, -10)', 'Weights'): pd.Series({'A': -100, 'B': -25, 'C': -10}),
        ('RFS', 'MMS(100, 10)', 'Weights'): pd.Series({'A': 100, 'B': 25, 'C': 10}),
        ('RFS', 'CS(-100)', 'Weights'): pd.Series({'A': -100, 'B': -74, 'C': -68}),
        ('RFS', 'CS(100)', 'Weights'): pd.Series({'A': 100, 'B': 74, 'C': 68})
    }
    return expected_results


def test_fit(create_data, expected_results):
    X_rules, y, weights = create_data
    expected_results = expected_results
    labels = product(['LR', 'PS', 'RFS'],
                     ['MMS(-100, -10)', 'MMS(100, 10)', 'CS(-100)', 'CS(100)'],
                     ['No weights', 'Weights'])
    score_scaler_comb = list(product([rsm.LogRegScorer(), rsm.PerformanceScorer(precision_score), rsm.RandomForestScorer()],
                                     [rss.MinMaxScaler(-100, -10), rss.MinMaxScaler(
                                         10, 100), rss.ConstantScaler(-100), rss.ConstantScaler(100)],
                                     [None, weights]))

    for label, (scorer, scaler, w) in zip(labels, score_scaler_comb):
        rs = RuleScorer(scorer, scaler)
        rs.fit(X_rules, y, w)
        assert all(rs.rule_scores == expected_results[label])


def test_transform(create_data, expected_results):
    X_rules, y, _ = create_data
    expected_result = expected_results
    expected_result = expected_result[('LR', 'MMS(-100, -10)', 'No weights')]
    rs = RuleScorer(rsm.LogRegScorer(), rss.MinMaxScaler(-100, -10))
    rs.fit(X_rules, y)
    X_scores = rs.transform(X_rules)
    assert all(X_scores == expected_result * X_rules)
    assert all(X_scores.min() == rs.rule_scores)


def test_fit_transform(create_data, expected_results):
    X_rules, y, weights = create_data
    expected_results = expected_results
    labels = product(['LR', 'PS', 'RFS'],
                     ['MMS(-100, -10)', 'MMS(100, 10)', 'CS(-100)', 'CS(100)'],
                     ['No weights', 'Weights'])
    score_scaler_comb = list(product([rsm.LogRegScorer(), rsm.PerformanceScorer(precision_score), rsm.RandomForestScorer()],
                                     [rss.MinMaxScaler(-100, -10), rss.MinMaxScaler(
                                         10, 100), rss.ConstantScaler(-100), rss.ConstantScaler(100)],
                                     [None, weights]))

    for label, (scorer, scaler, w) in zip(labels, score_scaler_comb):
        rs = RuleScorer(scorer, scaler)
        X_scores = rs.fit_transform(X_rules, y, w)
        assert all(rs.rule_scores == expected_results[label])
        assert X_scores.shape == X_rules.shape
        assert all(X_scores == expected_results[label] * X_rules)
