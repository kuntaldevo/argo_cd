import pandas as pd
import numpy as np
import rule_optimisation.optimisation_functions as opt_funcs
import sklearn.metrics as sklearn_metrics
import pytest


@pytest.fixture
def create_data():
    np.random.seed(0)
    y_pred = np.random.randint(0, 2, 1000)
    y_true = np.random.randint(0, 2, 1000)
    weights = y_true * 10
    return y_true, y_pred, weights


def test_Precision(create_data):
    y_true, y_pred, weights = create_data
    precision = opt_funcs.Precision()
    for w in [None, weights]:
        prec_calc = precision.fit(y_true, y_pred, w)
        prec_exp = sklearn_metrics.precision_score(
            y_true, y_pred, sample_weight=w)
        assert prec_calc == prec_exp


def test_Recall(create_data):
    y_true, y_pred, weights = create_data
    recall = opt_funcs.Recall()
    for w in [None, weights]:
        recall_calc = recall.fit(y_true, y_pred, w)
        recall_exp = sklearn_metrics.recall_score(
            y_true, y_pred, sample_weight=w)
        assert recall_calc == recall_exp


def test_FScore(create_data):
    y_true, y_pred, weights = create_data
    f1 = opt_funcs.FScore(1)
    for w in [None, weights]:
        f1_calc = f1.fit(y_true, y_pred, w)
        f1_exp = sklearn_metrics.fbeta_score(
            y_true, y_pred, beta=1, sample_weight=w)
        assert f1_calc == f1_exp


def test_Revenue(create_data):
    np.random.seed(0)
    y_true, y_pred, _ = create_data
    amts = np.random.uniform(0, 1000, 1000)
    r = opt_funcs.Revenue(y_type='Fraud', chargeback_multiplier=2)
    rev_calc = r.fit(y_true, y_pred, amts)
    rev_exp = 40092.77587231972
    assert rev_calc == rev_exp


def test_AlertsPerDay(create_data):
    np.random.seed(0)
    _, y_pred, _ = create_data
    apd = opt_funcs.AlertsPerDay(
        n_alerts_expected_per_day=50, no_of_days_in_file=30)
    apd_calc = apd.fit(y_pred)
    apd_exp = -1102.2400000000002
    assert apd_calc == apd_exp


def test_PercVolume(create_data):
    np.random.seed(0)
    _, y_pred, _ = create_data
    pv = opt_funcs.PercVolume(perc_vol_expected=0.02)
    pv_calc = pv.fit(y_pred)
    pv_exp = -0.234256
    assert pv_calc == pv_exp
