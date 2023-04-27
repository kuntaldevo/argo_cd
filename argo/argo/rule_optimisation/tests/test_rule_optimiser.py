import pytest
from rule_optimisation.rule_optimiser import RuleOptimiser
from rule_optimisation.optimisation_functions import FScore, AlertsPerDay
from rules.rules import Rules
import numpy as np
import pandas as pd
from hyperopt import hp
from hyperopt.pyll import scope


@pytest.fixture
def _create_data():
    np.random.seed(0)
    X = pd.DataFrame({
        'A': np.random.randint(0, 10, 10000),
        'B': np.random.randint(0, 100, 10000),
        'C': np.random.uniform(0, 1, 10000),
        'D': [True, False] * 5000,
        'E': ['yes', 'no'] * 5000,
        'AllNa': [np.nan] * 10000,
        'ZeroVar': [1] * 10000
    })
    X.loc[10000] = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    X['A'] = X['A'].astype('Int64')
    X['B'] = X['B'].astype('Int64')
    X['D'] = X['D'].astype('boolean')
    y = pd.Series(np.random.randint(0, 2, 10001))
    sample_weight = np.where((X['A'] > 7).fillna(False) & (y == 0), 100, 1)
    return X, y, sample_weight


@pytest.fixture
def _create_inputs():
    rule_lambdas = {
        'integer': lambda **kwargs: "(X['A']>{A})".format(**kwargs),
        'float': lambda **kwargs: "(X['C']>{C})".format(**kwargs),
        'categoric': lambda **kwargs: "(X['E']=='yes')".format(**kwargs),
        'boolean': lambda **kwargs: "(X['D']==True)".format(**kwargs),
        'is_na': lambda **kwargs: "(X['A']>{A})|(X['A'].isna())".format(**kwargs),
        'mixed': lambda **kwargs: "((X['A']>{A})&(X['C']>{C})&(X['E']=='yes')&(X['D']==True))|(X['C']>{C%0})".format(**kwargs),
        'missing_col': lambda **kwargs: "(X['Z']>{Z})".format(**kwargs),
        'all_na': lambda **kwargs: "(X['AllNa']>{AllNa})".format(**kwargs),
        'zero_var': lambda **kwargs: "(X['ZeroVar']>{ZeroVar})".format(**kwargs),
        'already_optimal': lambda **kwargs: "(X['A']>{A})".format(**kwargs),
    }
    lambda_kwargs = {
        'integer': {'A': 9},
        'float': {'C': 1.5},
        'categoric': {},
        'boolean': {},
        'is_na': {'A': 9},
        'mixed': {'A': 1, 'C': 1.5, 'C%0': 2.5},
        'missing_col': {'Z': 1},
        'all_na': {'AllNa': 5},
        'zero_var': {'ZeroVar': 1},
        'already_optimal': {'A': 0}
    }
    return rule_lambdas, lambda_kwargs


@pytest.fixture
def _expected_results():
    opt_rule_strings = {
        'integer': "(X['A']>0)",
        'float': "(X['C']>0.003230558992660632)",
        'is_na': "(X['A']>0)|(X['A'].isna())",
        'mixed': "((X['A']>8)&(X['C']>0.2731178395058975)&(X['E']=='yes')&(X['D']==True))|(X['C']>0.003230558992660632)",
        'already_optimal': "(X['A']>0.0)",
        'categoric': "(X['E']=='yes')",
        'boolean': "(X['D']==True)",
        'all_na': "(X['AllNa']>5.0)",
        'zero_var': "(X['ZeroVar']>1.0)"
    }
    opt_rule_strings_weighted = {
        'integer': "(X['A']>0)",
        'float': "(X['C']>0.14437974242018892)",
        'is_na': "(X['A']>0)|(X['A'].isna())",
        'mixed': "((X['A']>3)&(X['C']>0.3449413915707924)&(X['E']=='yes')&(X['D']==True))|(X['C']>0.14437974242018892)",
        'already_optimal': "(X['A']>0.0)",
        'categoric': "(X['E']=='yes')",
        'boolean': "(X['D']==True)",
        'all_na': "(X['AllNa']>5.0)",
        'zero_var': "(X['ZeroVar']>1.0)"
    }
    orig_rule_performances = {
        'already_optimal': 0.6422306211224418,
        'integer': 0.0,
        'float': 0.0,
        'is_na': 0.0,
        'mixed': 0.0
    }
    orig_rule_performances_weighted = {
        'already_optimal': 0.07737844641675759,
        'integer': 0.0,
        'float': 0.0,
        'is_na': 0.0,
        'mixed': 0.0
    }
    opt_rule_performances = {
        'float': 0.6642155224279698,
        'mixed': 0.6642155224279698,
        'integer': 0.6422306211224418,
        'already_optimal': 0.6422306211224418,
        'is_na': 0.6421848260125499
    }
    opt_rule_performances_weighted = {
        'float': 0.0864948723631455,
        'mixed': 0.0864948723631455,
        'integer': 0.07737844641675759,
        'already_optimal': 0.07737844641675759,
        'is_na': 0.07737778159635708
    }
    return opt_rule_strings, opt_rule_strings_weighted, orig_rule_performances, \
        orig_rule_performances_weighted, opt_rule_performances, opt_rule_performances_weighted


@pytest.fixture
def _expected_results_unlabelled():
    opt_rule_strings = {
        'integer': "(X['A']>9.0)",
        'float': "(X['C']>0.9934712038306385)",
        'is_na': "(X['A']>9.0)|(X['A'].isna())",
        'mixed': "((X['A']>1.0)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
        'already_optimal': "(X['A']>8)",
        'categoric': "(X['E']=='yes')",
        'boolean': "(X['D']==True)",
        'all_na': "(X['AllNa']>5.0)",
        'zero_var': "(X['ZeroVar']>1.0)"
    }
    orig_rule_performances = {
        'is_na': -98.01,
        'integer': -100.0,
        'float': -100.0,
        'mixed': -100.0,
        'already_optimal': -797984.8899999999
    }
    opt_rule_performances = {
        'float': -16.0,
        'mixed': -100.0,
        'integer': -100.0,
        'already_optimal': -8892.49,
        'is_na': -98.01
    }
    return opt_rule_strings, orig_rule_performances, opt_rule_performances


@pytest.fixture
def _instantiate(_create_inputs):
    rule_lambdas, lambda_kwargs = _create_inputs
    f1 = FScore(beta=1)
    ro = RuleOptimiser(rule_lambdas=rule_lambdas,
                       lambda_kwargs=lambda_kwargs, opt_func=f1.fit, n_iter=30)
    return ro


def test_fit(_create_data, _instantiate, _expected_results):
    X, y, _ = _create_data
    exp_opt_rule_strings, _, exp_orig_rule_performances, _, exp_opt_rule_performances, _ = _expected_results
    ro = _instantiate
    with pytest.warns(UserWarning,
                      match="Rules `missing_col` use features that are missing from `X` - unable to optimise or apply these rules"):
        opt_rule_strings = ro.fit(X=X, y=y)
        assert opt_rule_strings == exp_opt_rule_strings
        assert ro.orig_rule_performances == exp_orig_rule_performances
        assert ro.opt_rule_performances == exp_opt_rule_performances
        assert ro.rule_names_missing_features == ['missing_col']
        assert ro.rule_names_no_opt_conditions == [
            'categoric', 'boolean', 'all_na']
        assert ro.rule_names_zero_var_features == ['zero_var']


def test_fit_weighted(_create_data, _instantiate, _expected_results):
    X, y, sample_weight = _create_data
    _, exp_opt_rule_strings, _, exp_orig_rule_performances, _, exp_opt_rule_performances = _expected_results
    ro = _instantiate
    with pytest.warns(UserWarning,
                      match="Rules `missing_col` use features that are missing from `X` - unable to optimise or apply these rules"):
        opt_rule_strings = ro.fit(X=X, y=y, sample_weight=sample_weight)
        assert opt_rule_strings == exp_opt_rule_strings
        assert ro.orig_rule_performances == exp_orig_rule_performances
        assert ro.opt_rule_performances == exp_opt_rule_performances
        assert ro.rule_names_missing_features == ['missing_col']
        assert ro.rule_names_no_opt_conditions == [
            'categoric', 'boolean', 'all_na']
        assert ro.rule_names_zero_var_features == ['zero_var']


def test_fit_unlabelled(_create_data, _instantiate, _expected_results_unlabelled):
    X, _, _ = _create_data
    exp_opt_rule_strings, exp_orig_rule_performances, exp_opt_rule_performances = _expected_results_unlabelled
    apd = AlertsPerDay(n_alerts_expected_per_day=10, no_of_days_in_file=10)
    ro = _instantiate
    ro.opt_func = apd.fit
    with pytest.warns(UserWarning,
                      match="Rules `missing_col` use features that are missing from `X` - unable to optimise or apply these rules"):
        opt_rule_strings = ro.fit(X=X)
        assert opt_rule_strings == exp_opt_rule_strings
        assert ro.orig_rule_performances == exp_orig_rule_performances
        assert ro.opt_rule_performances == exp_opt_rule_performances
        assert ro.rule_names_missing_features == ['missing_col']
        assert ro.rule_names_no_opt_conditions == [
            'categoric', 'boolean', 'all_na']
        assert ro.rule_names_zero_var_features == ['zero_var']


def test_apply(_create_data, _instantiate):
    ro = _instantiate
    X = pd.DataFrame({
        'A': [1, 2, 0, 1, 0, 2]
    })
    y = pd.Series([0, 1, 0, 0, 0, 1])
    exp_X_rules = pd.DataFrame({
        'Rule': [0, 1, 0, 0, 0, 1]
    })
    exp_rule_descriptions = pd.DataFrame(
        data=np.array([['Rule', 1.0, 1.0, 0.3333333333333333,
                        1.0, "(X['A'] > 1)", 1]], dtype=object),
        columns=['Rule', 'Precision', 'Recall', 'PercDataFlagged', 'OptMetric', 'Logic', 'nConditions'])
    exp_rule_descriptions.set_index('Rule', inplace=True)
    ro.opt_rule_strings = {
        'Rule': "(X['A'] > 1)",
    }
    X_rules = ro.apply(X, y)
    assert all(X_rules == exp_X_rules)
    assert all(ro.rule_descriptions == exp_rule_descriptions)


def test_optimise_rules(_create_data, _instantiate, _expected_results):
    X, y, sample_weight = _create_data
    ro = _instantiate
    exp_opt_rule_strings, exp_opt_rule_strings_weighted, _, _, _, _ = _expected_results
    for rule_name in ['missing_col', 'categoric', 'boolean', 'all_na', 'zero_var', 'already_optimal']:
        ro.rule_lambdas.pop(rule_name)
        ro.lambda_kwargs.pop(rule_name)
        exp_opt_rule_strings.pop(rule_name, None)
        exp_opt_rule_strings_weighted.pop(rule_name, None)
    int_cols = ['A', 'B', 'D']
    all_space_funcs = {
        'A': scope.int(hp.uniform('A', X['A'].min(), X['A'].max())),
        'C%0': hp.uniform('C%0', X['C'].min(), X['C'].max()),
        'C': hp.uniform('C', X['C'].min(), X['C'].max())
    }
    for exp_result, w in zip([exp_opt_rule_strings, exp_opt_rule_strings_weighted], [None, sample_weight]):
        opt_rule_strings = ro._optimise_rules(rule_lambdas=ro.rule_lambdas, lambda_kwargs=ro.lambda_kwargs,
                                              X=X, y=y, sample_weight=w,
                                              int_cols=int_cols, all_space_funcs=all_space_funcs)
        assert opt_rule_strings == exp_result


def test_return_int_cols(_instantiate):
    exp_int_cols = ['int', 'int_stored_as_float']
    X = pd.DataFrame({
        'int': [0, 1, 2, np.nan],
        'float': [0, 1.5, 2.5, np.nan],
        'int_stored_as_float': [1, 2, 3, np.nan]
    })
    X['int'] = X['int'].astype('Int64')
    ro = _instantiate
    int_cols = ro._return_int_cols(X=X)
    assert int_cols == exp_int_cols


def test_return_all_optimisable_rule_features(_instantiate, _create_data, _create_inputs):
    exp_all_features = ['A', 'C', 'C%0', 'Z', 'ZeroVar']
    exp_rule_name_no_opt_conds = ['all_na', 'boolean', 'categoric']
    ro = _instantiate
    X, _,  _ = _create_data
    _, lambda_kwargs = _create_inputs
    with pytest.warns(UserWarning, match="Rules `categoric`, `boolean`, `all_na` have no optimisable conditions - unable to optimise these rules"):
        all_features, rule_names_no_opt_conditions = ro._return_all_optimisable_rule_features(
            lambda_kwargs=lambda_kwargs, X=X)
        all_features.sort()
        rule_names_no_opt_conditions.sort()
        assert all_features == exp_all_features
        assert rule_names_no_opt_conditions == exp_rule_name_no_opt_conds


def test_return_all_space_funcs(_create_data, _instantiate):
    X, _, _ = _create_data
    exp_results = {
        'A': 'int',
        'C': 'float',
        'C%0': 'float'
    }
    ro = _instantiate
    all_rule_features = ['A', 'C', 'C%0']
    int_cols = ['A']
    space_funcs = ro._return_all_space_funcs(
        all_rule_features=all_rule_features, X=X, int_cols=int_cols)
    for rule_name, space_func in space_funcs.items():
        print(rule_name)
        assert space_func.name == exp_results[rule_name]


def test_return_rules_with_zero_var_features(_instantiate, _create_inputs):
    all_space_funcs = {
        'A': scope.int(hp.uniform('A', 0, 10)),
        'B': scope.int(hp.uniform('B', 0, 10)),
        'C': hp.uniform('C', 0, 10),
        'C%0': hp.uniform('C%0', 0, 10),
        'ZeroVar': 1
    }
    _, lambda_kwargs = _create_inputs
    lambda_kwargs.pop('missing_col')
    ro = _instantiate
    with pytest.warns(UserWarning, match="Rules `zero_var` have all zero variance features based on the dataset `X` - unable to optimise these rules"):
        rule_names_zero_var_features = ro._return_rules_with_zero_var_features(lambda_kwargs=lambda_kwargs,
                                                                               all_space_funcs=all_space_funcs,
                                                                               rule_names_no_opt_conditions=['all_na', 'boolean', 'categoric'])
    assert rule_names_zero_var_features == ['zero_var']


def test_return_optimisable_rules(_instantiate, _create_inputs):
    rule_lambdas, lambda_kwargs = _create_inputs
    r = Rules(rule_lambdas=rule_lambdas, lambda_kwargs=lambda_kwargs)
    r.filter_rules(exclude=['missing_col'])
    ro = _instantiate
    rules, non_optimisable_rules = ro._return_optimisable_rules(rules=r, rule_names_no_opt_conditions=[
        'all_na', 'boolean', 'categoric', 'already_optimal'], rule_names_zero_var_features=['zero_var'])
    rule_names_opt = list(rules.rule_lambdas.keys())
    rule_names_non_opt = list(non_optimisable_rules.rule_lambdas.keys())
    rule_names_opt.sort()
    rule_names_non_opt.sort()
    assert rule_names_opt == ['float', 'integer', 'is_na', 'mixed']
    assert rule_names_non_opt == [
        'all_na', 'already_optimal', 'boolean', 'categoric', 'zero_var']


def test_return_rules_missing_features(_create_data, _create_inputs, _instantiate):
    X, _, _ = _create_data
    rule_lambdas, lambda_kwargs = _create_inputs
    r = Rules(rule_lambdas=rule_lambdas, lambda_kwargs=lambda_kwargs)
    ro = _instantiate
    with pytest.warns(UserWarning, match="Rules `missing_col` use features that are missing from `X` - unable to optimise or apply these rules"):
        rule_names_missing_features, rule_features_in_X = ro._return_rules_missing_features(
            rules=r, X=X)
    assert rule_names_missing_features == ['missing_col']
    assert rule_features_in_X == {'A', 'AllNa', 'C', 'D', 'E', 'ZeroVar'}


def test_return_rule_performances(_create_data, _expected_results, _instantiate):
    f1 = FScore(beta=1)
    X, y, sample_weight = _create_data
    ro = _instantiate
    opt_rule_strings, opt_rule_strings_weighted, _, _, opt_rule_performances, opt_rule_performance_weighted = _expected_results
    for rule_name in ['categoric', 'boolean', 'all_na', 'zero_var']:
        opt_rule_strings.pop(rule_name)
        opt_rule_strings_weighted.pop(rule_name)
    rule_perf = ro._return_rule_performances(
        rule_strings=opt_rule_strings, X=X, y=y, sample_weight=None, opt_func=f1.fit)
    assert rule_perf == opt_rule_performances
    rule_perf_weighted = ro._return_rule_performances(
        rule_strings=opt_rule_strings_weighted, X=X, y=y, sample_weight=sample_weight, opt_func=f1.fit)
    assert rule_perf_weighted == opt_rule_performance_weighted


def test_return_rule_space_funcs(_instantiate):
    all_space_funcs = {
        'A': 'FuncA',
        'B': 'FuncB',
        'C': 'FuncC'
    }
    exp_result = {
        'A': 'FuncA',
        'C': 'FuncC'
    }
    rule_features = ['A', 'C']
    ro = _instantiate
    rule_space_funcs = ro._return_rule_space_funcs(
        all_space_funcs=all_space_funcs, rule_features=rule_features)
    assert rule_space_funcs == exp_result


def test_optimise_rule_thresholds(_expected_results, _create_data, _create_inputs, _instantiate):
    exp_opt_threshold = {'A': 0.8284904721469425}
    f1 = FScore(beta=1)
    X, y, sample_weight = _create_data
    rule_lambdas, _ = _create_inputs
    rule_lambda = rule_lambdas['integer']
    rule_space_funcs = {
        'A': scope.int(hp.uniform('A', X['A'].min(), X['A'].max())),
    }
    ro = _instantiate
    for w in [None, sample_weight]:
        opt_threshold = ro._optimise_rule_thresholds(
            rule_lambda=rule_lambda, rule_space_funcs=rule_space_funcs, X_=X, y=y, sample_weight=w,
            opt_func=f1.fit, n_iter=30, show_progressbar=False)
        assert opt_threshold == exp_opt_threshold


def test_return_orig_rule_if_better_perf(_instantiate):
    orig_rule_performances = {
        'Rule1': 0.5,
        'Rule2': 0.5,
        'Rule3': 0.5,
    }
    opt_rule_performances = {
        'Rule1': 0.3,
        'Rule2': 0.5,
        'Rule3': 0.7,
    }
    orig_rule_strings = {
        'Rule1': "(X['A']>1",
        'Rule2': "(X['A']>2",
        'Rule3': "(X['A']>3"
    }
    opt_rule_strings = {
        'Rule1': "(X['A']>0",
        'Rule2': "(X['A']>4",
        'Rule3': "(X['A']>5"
    }
    expected_opt_rule_strings = {
        'Rule1': "(X['A']>1",
        'Rule2': "(X['A']>2",
        'Rule3': "(X['A']>5"
    }
    expected_opt_rule_performances = {
        'Rule1': 0.5,
        'Rule2': 0.5,
        'Rule3': 0.7,
    }
    ro = _instantiate
    ro.opt_rule_strings = opt_rule_strings
    opt_rule_strings, opt_rule_performances = ro._return_orig_rule_if_better_perf(orig_rule_performances=orig_rule_performances, opt_rule_performances=opt_rule_performances,
                                                                                  orig_rule_strings=orig_rule_strings, opt_rule_strings=opt_rule_strings)
    assert opt_rule_strings == expected_opt_rule_strings
    assert opt_rule_performances == expected_opt_rule_performances


def test_convert_opt_int_values(_instantiate):
    exp_result = {
        'A': 0,
        'C%0': 1.2,
        'C': 2
    }
    opt_thresholds = {
        'A': 0.82,
        'C%0': 1.2,
        'C': 2
    }
    int_cols = ['A']
    ro = _instantiate
    opt_thresholds = ro._convert_opt_int_values(
        opt_thresholds=opt_thresholds, int_cols=int_cols)
    assert opt_thresholds == exp_result


def test_calculate_performance_comparison(_instantiate):
    ro = _instantiate
    orig_rule_performances = {
        'Rule1': 0.1,
        'Rule2': 0.2,
        'Rule3': 0.3
    }
    opt_rule_performances = {
        'Rule1': 0.2,
        'Rule2': 0.4,
        'Rule3': 0.3
    }
    exp_performance_comp = pd.DataFrame(
        np.array([[0.1, 0.2],
                  [0.2, 0.4],
                  [0.3, 0.3]]),
        columns=['OriginalRule', 'OptimisedRule']
    )
    exp_performance_comp.index = ['Rule1', 'Rule2', 'Rule3']
    exp_performance_diff = pd.Series({
        'Rule1': 0.1,
        'Rule2': 0.2,
        'Rule3': 0
    })
    performance_comp, performance_difference = ro._calculate_performance_comparison(orig_rule_performances=orig_rule_performances,
                                                                                    opt_rule_performances=opt_rule_performances)
    assert all(performance_comp == exp_performance_comp)
    assert all(performance_difference == exp_performance_diff)
