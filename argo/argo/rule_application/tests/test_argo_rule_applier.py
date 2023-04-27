import pandas as pd
import numpy as np
import random
import pytest
from sklearn.ensemble import RandomForestClassifier
from rule_optimisation.optimisation_functions import FScore, AlertsPerDay
from rule_application.argo_rule_applier import ArgoRuleApplier
from sklearn.metrics import precision_score, recall_score


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
    X = X.astype(float)
    weights = y.apply(lambda x: 1000 if x == 1 else 1)
    return [X, y, weights]


@pytest.fixture
def create_X_rules(create_data):
    X, _, _ = create_data
    X_rules = pd.concat([X['num_distinct_txn_per_email_7day'] >= 7,
                         X['email_alpharatio'] <= 0.5,
                         X['num_distinct_txn_per_email_1day'] >= 1,
                         X['email_kb_distance'] <= 0.5,
                         X['ip_country_us'] == False,
                         X['num_distinct_txn_per_email_1day'] <= 3,
                         X['num_distinct_txn_per_email_7day'] <= 5,
                         X['email_kb_distance'] >= 0.61,
                         X['email_alpharatio'] >= 0.5], axis=1)
    X_rules.columns = [f'Rule{i}' for i in range(1, 10)]
    return X_rules


@pytest.fixture
def return_dummy_rules():
    rules = {
        'Rule1': "X['num_distinct_txn_per_email_7day']>=7",
        'Rule2': "X['email_alpharatio']<=0.5",
        'Rule3': "X['num_distinct_txn_per_email_1day']>=1",
        'Rule4': "X['email_kb_distance']<=0.5",
        'Rule5': "X['ip_country_us']==False",
        'Rule6': "X['num_distinct_txn_per_email_1day']<=3",
        'Rule7': "X['num_distinct_txn_per_email_7day']<=5",
        'Rule8': "X['email_kb_distance']>=0.61",
        'Rule9': "X['email_alpharatio']>=0.5"
    }
    return rules


@pytest.fixture
def return_dummy_rules_np():
    rules = {
        'Rule1': "(X['num_distinct_txn_per_email_7day'].to_numpy(na_value=np.nan)>=7.0)",
        'Rule2': "(X['email_alpharatio'].to_numpy(na_value=np.nan)<=0.5)",
        'Rule3': "(X['num_distinct_txn_per_email_1day'].to_numpy(na_value=np.nan)>=1.0)",
        'Rule4': "(X['email_kb_distance'].to_numpy(na_value=np.nan)<=0.5)",
        'Rule5': "(X['ip_country_us'].to_numpy(na_value=np.nan)==False)",
        'Rule6': "(X['num_distinct_txn_per_email_1day'].to_numpy(na_value=np.nan)<=3.0)",
        'Rule7': "(X['num_distinct_txn_per_email_7day'].to_numpy(na_value=np.nan)<=5.0)",
        'Rule8': "(X['email_kb_distance'].to_numpy(na_value=np.nan)>=0.61)",
        'Rule9': "(X['email_alpharatio'].to_numpy(na_value=np.nan)>=0.5)"
    }
    return rules


@pytest.fixture
def return_exp_results():
    rule_descriptions = pd.DataFrame(
        np.array([[0.03578528827037773, 0.9, 0.503, 0.06883365200764818,
                   "X['email_alpharatio']<=0.5", 1],
                  [0.03424657534246575, 1.0, 0.584, 0.06622516556291391,
                   "X['num_distinct_txn_per_email_1day']>=1", 1],
                  [0.04225352112676056, 0.15, 0.071, 0.06593406593406594,
                   "X['num_distinct_txn_per_email_7day']>=7", 1],
                  [0.0330188679245283, 0.7, 0.424, 0.06306306306306306,
                   "X['email_kb_distance']<=0.5", 1],
                  [0.03213610586011342, 0.85, 0.529, 0.06193078324225866,
                   "X['ip_country_us']==False", 1],
                  [0.022172949002217297, 1.0, 0.902, 0.04338394793926247,
                   "X['num_distinct_txn_per_email_1day']<=3", 1],
                  [0.01927437641723356, 0.85, 0.882, 0.037694013303769404,
                   "X['num_distinct_txn_per_email_7day']<=5", 1],
                  [0.01284796573875803, 0.3, 0.467, 0.024640657084188913,
                   "X['email_kb_distance']>=0.61", 1],
                  [0.004024144869215292, 0.1, 0.497, 0.007736943907156674,
                   "X['email_alpharatio']>=0.5", 1]]),
        columns=['Precision', 'Recall', 'PercDataFlagged', 'OptMetric', 'Logic',
                 'nConditions'],
        index=['Rule2', 'Rule3', 'Rule1', 'Rule4', 'Rule5', 'Rule6', 'Rule7', 'Rule8',
               'Rule9']
    )
    rule_descriptions.index.name = 'Rule'

    rule_descriptions_weighted = pd.DataFrame(
        np.array([[0.9725734292939117, 1.0, 0.584, 0.9860960457548565,
                   "X['num_distinct_txn_per_email_1day']>=1", 1],
                  [0.9577626664112633, 1.0, 0.902, 0.9784257130277384,
                   "X['num_distinct_txn_per_email_1day']<=3", 1],
                  [0.9737625101433595, 0.9, 0.503, 0.9354293880732754,
                   "X['email_alpharatio']<=0.5", 1],
                  [0.9707629054362723, 0.85, 0.529, 0.9063766261462998,
                   "X['ip_country_us']==False", 1],
                  [0.9515813042261405, 0.85, 0.882, 0.8979268453717154,
                   "X['num_distinct_txn_per_email_7day']<=5", 1],
                  [0.9715475364330326, 0.7, 0.424, 0.8137169427492007,
                   "X['email_kb_distance']<=0.5", 1],
                  [0.9286488159727596, 0.3, 0.467, 0.4534976002418654,
                   "X['email_kb_distance']>=0.61", 1],
                  [0.9778357235984355, 0.15, 0.071, 0.2601005722212589,
                   "X['num_distinct_txn_per_email_7day']>=7", 1],
                  [0.8016032064128257, 0.1, 0.497, 0.17781729273171817,
                   "X['email_alpharatio']>=0.5", 1]]),
        columns=['Precision', 'Recall', 'PercDataFlagged', 'OptMetric', 'Logic',
                 'nConditions'],
        index=['Rule3', 'Rule6', 'Rule2', 'Rule5', 'Rule7', 'Rule4', 'Rule8', 'Rule1',
               'Rule9']
    )
    rule_descriptions_weighted.index.name = 'Rule'

    rule_descriptions_unlabelled = pd.DataFrame(
        np.array([[0.071, -4.409999999999998,
                   "X['num_distinct_txn_per_email_7day']>=7", 1],
                  [0.424, -1398.76, "X['email_kb_distance']<=0.5", 1],
                  [0.467, -1738.8900000000003, "X['email_kb_distance']>=0.61", 1],
                  [0.497, -1998.0900000000001, "X['email_alpharatio']>=0.5", 1],
                  [0.503, -2052.0899999999997, "X['email_alpharatio']<=0.5", 1],
                  [0.529, -2294.41, "X['ip_country_us']==False", 1],
                  [0.584, -2851.56, "X['num_distinct_txn_per_email_1day']>=1", 1],
                  [0.882, -6922.240000000001,
                   "X['num_distinct_txn_per_email_7day']<=5", 1],
                  [0.902, -7259.040000000001,
                   "X['num_distinct_txn_per_email_1day']<=3", 1]]
                 ),
        columns=['PercDataFlagged', 'OptMetric', 'Logic', 'nConditions'],
        index=['Rule1', 'Rule4', 'Rule8', 'Rule9', 'Rule2', 'Rule5', 'Rule3', 'Rule7',
               'Rule6']
    )
    rule_descriptions_unlabelled.index.name = 'Rule'

    X_rules_sums = pd.Series({'Rule2': 503,
                              'Rule3': 584,
                              'Rule1': 71,
                              'Rule4': 424,
                              'Rule5': 529,
                              'Rule6': 902,
                              'Rule7': 882,
                              'Rule8': 467,
                              'Rule9': 497})

    return rule_descriptions, rule_descriptions_weighted, rule_descriptions_unlabelled, X_rules_sums


@ pytest.fixture
def fs_instantiated():
    fs = FScore(1)
    return fs


@ pytest.fixture
def ara_instantiated(return_dummy_rules, fs_instantiated):
    fs = fs_instantiated
    rules = return_dummy_rules
    ara = ArgoRuleApplier(rules, fs.fit)
    return ara


@ pytest.fixture
def ara_instantiated_np(return_dummy_rules_np, fs_instantiated):
    fs = fs_instantiated
    rules = return_dummy_rules_np
    ara = ArgoRuleApplier(rules, fs.fit)
    return ara


def test_apply(create_data, ara_instantiated, ara_instantiated_np,
               return_exp_results):
    X, y, _ = create_data
    exp_rule_descriptions, _, _, exp_X_rules_sums = return_exp_results
    ara = ara_instantiated
    ara_np = ara_instantiated_np
    X_rules = ara.apply(X, y, None)
    X_rules_np = ara_np.apply(X, y, None)
    assert all(ara.rule_descriptions == exp_rule_descriptions)
    assert all(ara_np.rule_descriptions == exp_rule_descriptions)
    assert all(X_rules == X_rules_np)
    assert all(X_rules.sum().sort_index() == exp_X_rules_sums.sort_index())
    assert all(X_rules_np.sum().sort_index() == exp_X_rules_sums.sort_index())


def test_apply_weighted(create_data, ara_instantiated, ara_instantiated_np,
                        return_exp_results):
    X, y, weights = create_data
    _, exp_rule_descriptions, _, exp_X_rules_sums = return_exp_results
    ara = ara_instantiated
    ara_np = ara_instantiated_np
    X_rules = ara.apply(X, y, weights)
    X_rules_np = ara_np.apply(X, y, weights)
    assert all(ara.rule_descriptions == exp_rule_descriptions)
    assert all(ara_np.rule_descriptions == exp_rule_descriptions)
    assert all(X_rules == X_rules_np)
    assert all(X_rules.sum().sort_index() == exp_X_rules_sums.sort_index())
    assert all(X_rules_np.sum().sort_index() == exp_X_rules_sums.sort_index())


def test_apply_unlabelled(create_data, ara_instantiated, ara_instantiated_np,
                          return_exp_results):
    X, _, _ = create_data
    _, _, exp_rule_descriptions, exp_X_rules_sums = return_exp_results
    ara = ara_instantiated
    ara_np = ara_instantiated_np
    apd = AlertsPerDay(
        n_alerts_expected_per_day=5, no_of_days_in_file=10)
    ara.opt_func = apd.fit
    ara_np.opt_func = apd.fit
    X_rules = ara.apply(X)
    X_rules_np = ara_np.apply(X)
    assert all(ara.rule_descriptions == exp_rule_descriptions)
    assert all(ara_np.rule_descriptions == exp_rule_descriptions)
    assert all(X_rules == X_rules_np)
    assert all(X_rules.sum().sort_index() == exp_X_rules_sums.sort_index())
    assert all(X_rules_np.sum().sort_index() == exp_X_rules_sums.sort_index())
    ara = ara_instantiated
    ara_np = ara_instantiated_np
    ara.opt_func = None
    ara_np.opt_func = None
    X_rules = ara.apply(X)
    X_rules_np = ara_np.apply(X)
    assert all(X_rules == X_rules_np)
    assert all(X_rules.sum().sort_index() == exp_X_rules_sums.sort_index())
    assert all(X_rules_np.sum().sort_index() == exp_X_rules_sums.sort_index())


def test_get_X_rules(create_data, ara_instantiated, ara_instantiated_np):
    X, _, _ = create_data
    ara = ara_instantiated
    ara_np = ara_instantiated_np
    X_rules = ara._get_X_rules(X)
    X_rules_np = ara_np._get_X_rules(X)
    assert X_rules.shape == (1000, 9)
    assert X_rules.sum().sum() == 4859
    assert X_rules_np.shape == (1000, 9)
    assert X_rules_np.sum().sum() == 4859


def test_warning(create_data, fs_instantiated):
    X, y, _ = create_data
    fs = fs_instantiated
    rule_strings = {'Rule1': "(X['Z']>1)"}
    ara = ArgoRuleApplier(rule_strings=rule_strings, opt_func=fs.fit)
    with pytest.raises(KeyError) as e:
        ara.apply(X, y)
        assert str(e) == "Feature 'Z' in rule `Rule1` not found in `X`"
