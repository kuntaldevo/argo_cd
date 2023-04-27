import pytest
import numpy as np
import pandas as pd
from argo.rule_generation.rule_generation.rule_generator_dt import RuleGeneratorDT
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, fbeta_score
import random


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
        'tree_ensemble': RandomForestClassifier(n_estimators=10, random_state=0),
        'precision_threshold': 0,
        'num_cores': 4
    }
    rg = RuleGeneratorDT(**params)
    rg.today = '20200204'
    return [rg, params]


@pytest.fixture
def create_dummy_rules():
    rule_descriptions = pd.DataFrame({
        'Rule': ["(X['A']>1)", "(X['A']>1)"],
        'Logic': ["(X['A']>1)", "(X['A']>1)"],
        'Precision': [0, 0],
        'Recall': [0, 0],
        'nConditions': [0, 0],
        'PercDataFlagged': [0, 0],
        'FScore': [0, 0],
        'Beta': [0, 0]
    })
    rule_descriptions.set_index('Rule', inplace=True)
    X_rules = pd.concat([pd.DataFrame({"(X['A']>1)": np.random.randint(0, 1, 100)}),
                         pd.DataFrame({"(X['A']>1)": np.random.randint(0, 1, 100)})],
                        axis=1)
    return rule_descriptions, X_rules


@ pytest.fixture
def fit_decision_tree():
    def _fit(X, y, sample_weight=None):
        dt = DecisionTreeClassifier(random_state=0, max_depth=4)
        dt.fit(X, y, sample_weight=sample_weight)
        return dt
    return _fit


def test_fit_dt(create_data, fit_decision_tree):
    X, y, _, _, _, weights = create_data
    for w in [None, weights]:
        dt = fit_decision_tree(X, y, w)
        dt_test = DecisionTreeClassifier(random_state=0, max_depth=4)
        dt_test.fit(X, y, w)
        dt_preds = dt.predict_proba(X)[:, -1]
        dt_test_preds = dt_test.predict_proba(X)[:, -1]
        assert [a == b for a, b in zip(dt_preds, dt_test_preds)]


def test_fit(create_data, rg_instantiated):

    def _assert_test_fit(rule_descriptions_shape, X_rules_shape):
        X_rules = rg.fit(X, y, sample_weight=w)
        _assert_rule_descriptions_and_X_rules(
            rg.rule_descriptions, X_rules, rule_descriptions_shape, X_rules_shape, X, y, params['opt_func'], w)

    X, y, _, _, _, weights = create_data
    rg, params = rg_instantiated
    for w in [None, weights]:
        _assert_test_fit((40, 6), (1000, 40)) if w is None else _assert_test_fit(
            (14, 6), (1000, 14))


def test_apply(create_data, rg_instantiated):

    def _assert_test_apply(rule_descriptions_shape, X_rules_shape):
        _ = rg.fit(X, y, sample_weight=w)
        X_rules_applied = rg.apply(X, y, w)
        _assert_rule_descriptions_and_X_rules(
            rg.rule_descriptions_applied, X_rules_applied, rule_descriptions_shape, X_rules_shape, X, y, params['opt_func'], w)

    X, y, _, _, _, weights = create_data
    rg, params = rg_instantiated
    for w in [None, weights]:
        _assert_test_apply((40, 6), (1000, 40)) if w is None else _assert_test_apply(
            (14, 6), (1000, 14))


def test_drop_low_prec_trees_return_rules(create_data, rg_instantiated, fit_decision_tree):

    expected_rule_sets_sample_weight_None = set([
        "(X['email_alpharatio']<=0.43817)&(X['email_kb_distance']>0.00092)&(X['num_distinct_txn_per_email_1day']<=1)",
        "(X['email_alpharatio']<=0.43817)&(X['email_kb_distance']>0.00092)&(X['num_distinct_txn_per_email_1day']>=2)",
        "(X['email_alpharatio']<=0.43844)&(X['email_alpharatio']>0.43817)&(X['email_kb_distance']>0.00092)",
        "(X['email_alpharatio']<=0.52347)&(X['email_kb_distance']<=0.00092)&(X['num_distinct_txn_per_email_1day']<=2)&(X['num_distinct_txn_per_email_1day']>=2)",
        "(X['email_alpharatio']<=0.52347)&(X['email_kb_distance']<=0.00092)&(X['num_distinct_txn_per_email_1day']>=3)",
        "(X['email_alpharatio']>0.43844)&(X['email_kb_distance']<=0.29061)&(X['email_kb_distance']>0.00092)&(X['num_distinct_txn_per_email_7day']>=5)"
    ])

    expected_rule_sets_sample_weight_given = set([
        "(X['email_alpharatio']<=0.57456)&(X['num_distinct_txn_per_email_1day']<=3)&(X['num_distinct_txn_per_email_1day']>=1)&(X['num_distinct_txn_per_email_7day']>=1)"
    ])
    X, y, columns_int, columns_cat, _, weights = create_data
    dt = fit_decision_tree(X, y, None)
    rg, _ = rg_instantiated
    rule_sets = rg._drop_low_prec_trees_return_rules(
        X=X, y=y, decision_tree=dt, sample_weight=None, columns_int=columns_int, columns_cat=columns_cat)
    assert rule_sets == expected_rule_sets_sample_weight_None
    dt = fit_decision_tree(X, y, weights)
    rg, _ = rg_instantiated
    rule_sets = rg._drop_low_prec_trees_return_rules(
        X=X, y=y, decision_tree=dt, sample_weight=weights, columns_int=columns_int, columns_cat=columns_cat)
    assert rule_sets == expected_rule_sets_sample_weight_given
    rg, _ = rg_instantiated
    rg.precision_threshold = 1
    rule_sets = rg._drop_low_prec_trees_return_rules(
        X=X, y=y, decision_tree=dt, sample_weight=weights, columns_int=columns_int, columns_cat=columns_cat)
    assert rule_sets == set()


def test_extract_rules_from_ensemble(create_data, rg_instantiated):

    def _assert_extract_rules_from_ensemble(rule_descriptions_shape, X_rules_shape):
        rf = params['tree_ensemble']
        rf.fit(X, y, sample_weight=w)
        par_list = rg._extract_rules_from_ensemble(
            X, y, rf, params['num_cores'], w, columns_int, columns_cat)
        rule_descriptions = par_list[0]
        X_rules = par_list[1]
        _assert_rule_descriptions_and_X_rules(
            rule_descriptions, X_rules, rule_descriptions_shape, X_rules_shape, X, y, params['opt_func'], w)

    X, y, columns_int, columns_cat, _, weights = create_data
    rg, params = rg_instantiated
    for w in [None, weights]:
        _assert_extract_rules_from_ensemble(
            (40, 6), (1000, 40)) if w is None else _assert_extract_rules_from_ensemble((14, 6), (1000, 14))


def test_train_ensemble(rg_instantiated, create_data):
    X, y, _, _, _, weights = create_data
    rg, params = rg_instantiated
    rg.tree_ensemble.max_depth = params['n_total_conditions']
    for w in [None, weights]:
        rg_trained = rg._train_ensemble(
            X, y, tree_ensemble=rg.tree_ensemble, sample_weight=w)
        rf = RandomForestClassifier(
            max_depth=params['n_total_conditions'], random_state=0, n_estimators=100)
        rf.fit(X=X, y=y, sample_weight=w)
        rf_preds = rf.predict_proba(X)[:, -1]
        rg_preds = rg_trained.predict_proba(X)[:, -1]
        assert [a == b for a, b in zip(rf_preds, rg_preds)]


def test_generate_rule_name(rg_instantiated):
    rg, _ = rg_instantiated
    rule_name = rg._generate_rule_name()
    assert rule_name == 'RGDT_Rule_20200204_0'


def test_extract_rules_from_tree(fit_decision_tree, rg_instantiated, create_data):

    expected_rule_sets_sample_weight_None = set([
        "(X['email_alpharatio']<=0.43817)&(X['email_kb_distance']>0.00092)&(X['num_distinct_txn_per_email_1day']<=1)",
        "(X['email_alpharatio']<=0.43817)&(X['email_kb_distance']>0.00092)&(X['num_distinct_txn_per_email_1day']>=2)",
        "(X['email_alpharatio']<=0.43844)&(X['email_alpharatio']>0.43817)&(X['email_kb_distance']>0.00092)",
        "(X['email_alpharatio']<=0.52347)&(X['email_kb_distance']<=0.00092)&(X['num_distinct_txn_per_email_1day']<=2)&(X['num_distinct_txn_per_email_1day']>=2)",
        "(X['email_alpharatio']<=0.52347)&(X['email_kb_distance']<=0.00092)&(X['num_distinct_txn_per_email_1day']>=3)",
        "(X['email_alpharatio']>0.43844)&(X['email_kb_distance']<=0.29061)&(X['email_kb_distance']>0.00092)&(X['num_distinct_txn_per_email_7day']>=5)"
    ])

    expected_rule_sets_sample_weight_given = set([
        "(X['email_alpharatio']<=0.57456)&(X['num_distinct_txn_per_email_1day']<=3)&(X['num_distinct_txn_per_email_1day']>=1)&(X['num_distinct_txn_per_email_7day']>=1)"
    ])
    X, y, columns_int, columns_cat, _, weights = create_data
    dt = fit_decision_tree(X, y, None)
    rg, _ = rg_instantiated
    rule_sets = rg._extract_rules_from_tree(
        X=X, decision_tree=dt, precision_threshold=rg.precision_threshold, columns_int=columns_int,
        columns_cat=columns_cat)
    assert rule_sets == expected_rule_sets_sample_weight_None
    dt = fit_decision_tree(X, y, weights)
    rg, _ = rg_instantiated
    rule_sets = rg._extract_rules_from_tree(
        X=X, decision_tree=dt, precision_threshold=rg.precision_threshold, columns_int=columns_int,
        columns_cat=columns_cat)
    assert rule_sets == expected_rule_sets_sample_weight_given
    rg, _ = rg_instantiated
    rg.precision_threshold = 1
    rule_sets = rg._extract_rules_from_tree(
        X=X, decision_tree=dt, precision_threshold=rg.precision_threshold, columns_int=columns_int,
        columns_cat=columns_cat)
    assert rule_sets == set()


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
