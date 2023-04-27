import pytest
import pandas as pd
import numpy as np
import json
from rule_application.sim_rule_applier import SimRuleApplier
from rule_optimisation.optimisation_functions import FScore, AlertsPerDay
from sklearn.metrics import precision_score, recall_score, fbeta_score


@pytest.fixture
def create_data():
    np.random.seed(0)
    X = pd.DataFrame({
        'eid': list(range(0, 10)),
        'sim_ll': [
            json.dumps({'A': 10, 'B': -1}),
            json.dumps({'A': 10, 'C': -2}),
            json.dumps({'B': -1, 'D': -1}),
            json.dumps({'A': 10, 'B': -1}),
            json.dumps({'A': 10, 'D': -1}),
            json.dumps({'B': -1, 'E': 2}),
            json.dumps({'A': 10, 'B': -1, 'D': -1}),
            json.dumps({'A': 10, 'B': -1}),
            json.dumps({'A': 10, 'B': -1}),
            json.dumps({'A': 10, 'B': -1}),
        ]
    })
    X.set_index('eid', inplace=True)
    y = pd.Series(np.random.randint(0, 2, 10), list(
        range(0, 10)), name='sim_is_fraud')
    weights = y * 10
    return X, y, weights


@pytest.fixture
def fs_instantiated():
    f1 = FScore(1)
    return f1


@pytest.fixture
def instantiate_class(fs_instantiated):
    f1 = fs_instantiated
    sra = SimRuleApplier(f1.fit)
    return sra


@pytest.fixture
def expected_X_rules():
    X_rules = pd.DataFrame({
        "A": [1, 1, 0, 1, 1, 0, 1, 1, 1, 1],
        "B": [1, 0, 1, 1, 0, 1, 1, 1, 1, 1],
        "D": [0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
        "C": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        "E": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    }
    )
    return X_rules


@pytest.fixture
def _expected_rule_descriptions():
    rule_descriptions = pd.DataFrame(
        np.array([[0.75, 0.75, 0.8, 0.75],
                  [0.75, 0.75, 0.8, 0.75],
                  [1., 0.375, 0.3, 0.54545455],
                  [1., 0.125, 0.1, 0.22222222],
                  [1., 0.125, 0.1, 0.22222222]]),
        columns=['Precision', 'Recall', 'PercDataFlagged', 'OptMetric'],
        index=['A', 'B', 'D', 'C', 'E']
    )
    rule_descriptions.index.name = 'Rule'
    rule_descriptions_weighted = pd.DataFrame(
        np.array([[1., 0.75, 0.8, 0.85714286],
                  [1., 0.75, 0.8, 0.85714286],
                  [1., 0.375, 0.3, 0.54545455],
                  [1., 0.125, 0.1, 0.22222222],
                  [1., 0.125, 0.1, 0.22222222]]),
        columns=['Precision', 'Recall', 'PercDataFlagged', 'OptMetric'],
        index=['A', 'B', 'D', 'C', 'E']
    )
    rule_descriptions_weighted.index.name = 'Rule'
    rule_descriptions_unlabelled = pd.DataFrame(
        np.array([[0.8, -84.64],
                  [0.8, -84.64],
                  [0.3, -94.09],
                  [0.1, -98.01],
                  [0.1, -98.01]]),
        columns=['PercDataFlagged', 'OptMetric'],
        index=['A', 'B', 'D', 'C', 'E']
    )
    rule_descriptions_unlabelled.index.name = 'Rule'
    return rule_descriptions, rule_descriptions_weighted, rule_descriptions_unlabelled


def test_apply(create_data, instantiate_class, expected_X_rules,
               _expected_rule_descriptions):
    X, y, _ = create_data
    sra = instantiate_class
    expected_X_rules = expected_X_rules
    expected_rule_descriptions, _, _ = _expected_rule_descriptions
    X_rules = sra.apply(X, y, None)
    assert all(X_rules == expected_X_rules)
    assert all(sra.rule_descriptions == expected_rule_descriptions)


def test_apply_weighted(create_data, instantiate_class, expected_X_rules,
                        _expected_rule_descriptions):
    X, y, weights = create_data
    sra = instantiate_class
    expected_X_rules = expected_X_rules
    _, expected_rule_descriptions, _ = _expected_rule_descriptions
    X_rules = sra.apply(X, y, weights)
    assert all(X_rules == expected_X_rules)
    assert all(sra.rule_descriptions == expected_rule_descriptions)


def test_apply_unlabelled(create_data, instantiate_class, expected_X_rules,
                          _expected_rule_descriptions):
    X, _, _ = create_data
    sra = instantiate_class
    apd = AlertsPerDay(n_alerts_expected_per_day=10,
                       no_of_days_in_file=10)
    sra.opt_func = apd.fit
    expected_X_rules = expected_X_rules
    _, _, expected_rule_descriptions = _expected_rule_descriptions
    X_rules = sra.apply(X)
    assert all(X_rules == expected_X_rules)
    assert all(sra.rule_descriptions == expected_rule_descriptions)
    sra = instantiate_class
    sra.opt_func = None
    X_rules = sra.apply(X)
    X_rules = X_rules.reindex(expected_X_rules.columns, axis=1)
    assert all(X_rules == expected_X_rules)


def test_apply_filtered(create_data, instantiate_class, expected_X_rules,
                        _expected_rule_descriptions):
    X, y, _ = create_data
    expected_rule_descriptions, _, _ = _expected_rule_descriptions
    expected_rule_descriptions = expected_rule_descriptions.loc[['A', 'B']]
    sra = instantiate_class
    sra.rules = ['A', 'B']
    expected_X_rules = expected_X_rules
    expected_X_rules = expected_X_rules[['A', 'B']]
    X_rules = sra.apply(X, y)
    assert all(X_rules == expected_X_rules)
    assert all(sra.rule_descriptions == expected_rule_descriptions)


def test_apply_filtered_missing(create_data, instantiate_class,
                                expected_X_rules, _expected_rule_descriptions):
    X, y, _ = create_data
    expected_rule_descriptions, _, _ = _expected_rule_descriptions
    expected_rule_descriptions = expected_rule_descriptions.loc[['A', 'B']]
    sra = instantiate_class
    sra.rules = ['A', 'B', 'Z']
    expected_X_rules = expected_X_rules
    expected_X_rules = expected_X_rules[['A', 'B']]
    with pytest.warns(UserWarning,
                      match=f'Rules `Z` not found in `sim_ll` - unable to apply these rules.'):
        X_rules = sra.apply(X, y, None)
        assert all(X_rules == expected_X_rules)
        assert all(sra.rule_descriptions == expected_rule_descriptions)
        assert sra.rules_not_in_sim_ll == ['Z']


def test_error(create_data, instantiate_class):
    X, y, _ = create_data
    X.drop('sim_ll', axis=1, inplace=True)
    sra = instantiate_class
    with pytest.raises(Exception):
        sra.apply(X, y)
