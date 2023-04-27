import pytest
from rules.rules import Rules
import pandas as pd
import numpy as np


@pytest.fixture
def _data():
    np.random.seed(0)
    X = pd.DataFrame(
        {
            'A': np.random.uniform(0, 100, 100),
            'B': ['foo', 'bar'] * 50,
            'C': [1.0, 2.0, 3.0, np.nan] * 25,
            'D': ['foo', 'bar', np.nan, 'no'] * 25,
            'E': [1.0, 2.0, 3.0, 4.0] * 25,
            'F': [1.0, 2.0, 3.0, 5.0] * 25,
            'G': ['foo', '', np.nan, ''] * 25,

        }
    )
    return X


@pytest.fixture
def _rule_dicts():
    rule_dicts = {'Rule1': {'condition': 'AND',
                            'rules': [{'condition': 'OR',
                                       'rules': [{'field': 'payer_id_sum_approved_txn_amt_per_paypalid_1day',
                                                  'operator': 'greater_or_equal',
                                                  'value': 60.0},
                                                 {'field': 'payer_id_sum_approved_txn_amt_per_paypalid_7day',
                                                  'operator': 'greater',
                                                  'value': 120.0},
                                                 {'field': 'payer_id_sum_approved_txn_amt_per_paypalid_30day',
                                                  'operator': 'less_or_equal',
                                                  'value': 500.0}]},
                                      {'field': 'num_items', 'operator': 'equal', 'value': 1.0}]},
                  'Rule2': {'condition': 'AND',
                            'rules': [{'field': 'ml_cc_v0', 'operator': 'less', 'value': 0.315},
                                      {'condition': 'OR',
                                       'rules': [{'field': 'method_clean',
                                                  'operator': 'equal',
                                                  'value': 'checkout'},
                                                 {'field': 'method_clean',
                                                  'operator': 'begins_with', 'value': 'checkout'},
                                                 {'field': 'method_clean',
                                                  'operator': 'ends_with', 'value': 'checkout'},
                                                 {'field': 'method_clean',
                                                  'operator': 'contains', 'value': 'checkout'},
                                                 {'field': 'ip_address',
                                                  'operator': 'is_not_null', 'value': None},
                                                 {'field': 'ip_isp', 'operator': 'is_not_empty', 'value': None}]}]},
                  'Rule3': {'condition': 'AND',
                            'rules': [{'field': 'method_clean',
                                       'operator': 'not_begins_with',
                                       'value': 'checkout'},
                                      {'field': 'method_clean',
                                       'operator': 'not_ends_with', 'value': 'checkout'},
                                      {'field': 'method_clean',
                                       'operator': 'not_contains', 'value': 'checkout'},
                                      {'condition': 'OR',
                                       'rules': [{'field': 'ip_address', 'operator': 'is_null', 'value': None},
                                                 {'field': 'ip_isp', 'operator': 'is_empty', 'value': None}]}]},
                  'Rule4': {'condition': 'AND',
                            'rules': [{'field': 'forwarder_address', 'operator': 'equal', 'value': True},
                                      {'field': 'is_shipping_billing_address_same',
                                       'operator': 'equal',
                                       'value': False}]},
                  'Rule5': {'condition': 'AND',
                            'rules': [{'field': 'ad_price_type',
                                       'operator': 'not_in',
                                       'value': ['FREE', 'NEGOTIATION']},
                                      {'field': 'ad_price_type', 'operator': 'in', 'value': ['FOO', 'BAR']}]},
                  'Rule6': {'condition': 'AND',
                            'rules': [{'field': 'ip_country_iso_code',
                                       'operator': 'equal_field',
                                       'value': 'billing_country'},
                                      {'field': 'country_id',
                                       'operator': 'not_equal_field',
                                       'value': 'ip_country_iso_code'}]}}
    return rule_dicts


@pytest.fixture
def _rule_strings_pandas():
    rule_strings = {'Rule1': "((X['payer_id_sum_approved_txn_amt_per_paypalid_1day']>=60.0)|(X['payer_id_sum_approved_txn_amt_per_paypalid_7day']>120.0)|(X['payer_id_sum_approved_txn_amt_per_paypalid_30day']<=500.0))&(X['num_items']==1.0)",
                    'Rule2': "(X['ml_cc_v0']<0.315)&((X['method_clean']=='checkout')|(X['method_clean'].str.startswith('checkout', na=False))|(X['method_clean'].str.endswith('checkout', na=False))|(X['method_clean'].str.contains('checkout', na=False, regex=False))|(~X['ip_address'].isna())|(X['ip_isp'].fillna('')!=''))",
                    'Rule3': "(~X['method_clean'].str.startswith('checkout', na=False))&(~X['method_clean'].str.endswith('checkout', na=False))&(~X['method_clean'].str.contains('checkout', na=False, regex=False))&((X['ip_address'].isna())|(X['ip_isp'].fillna('')==''))",
                    'Rule4': "(X['forwarder_address']==True)&(X['is_shipping_billing_address_same']==False)",
                    'Rule5': "(~X['ad_price_type'].isin(['FREE', 'NEGOTIATION']))&(X['ad_price_type'].isin(['FOO', 'BAR']))",
                    'Rule6': "(X['ip_country_iso_code']==X['billing_country'])&(X['country_id']!=X['ip_country_iso_code'])"}
    return rule_strings


@pytest.fixture
def _rule_strings_numpy():
    rule_strings = {'Rule1': "((X['payer_id_sum_approved_txn_amt_per_paypalid_1day'].to_numpy(na_value=np.nan)>=60.0)|(X['payer_id_sum_approved_txn_amt_per_paypalid_7day'].to_numpy(na_value=np.nan)>120.0)|(X['payer_id_sum_approved_txn_amt_per_paypalid_30day'].to_numpy(na_value=np.nan)<=500.0))&(X['num_items'].to_numpy(na_value=np.nan)==1.0)",
                    'Rule2': "(X['ml_cc_v0'].to_numpy(na_value=np.nan)<0.315)&((X['method_clean'].to_numpy(na_value=np.nan)=='checkout')|(X['method_clean'].str.startswith('checkout', na=False))|(X['method_clean'].str.endswith('checkout', na=False))|(X['method_clean'].str.contains('checkout', na=False, regex=False))|(~pd.isna(X['ip_address'].to_numpy(na_value=np.nan)))|(X['ip_isp'].fillna('')!=''))",
                    'Rule3': "(~X['method_clean'].str.startswith('checkout', na=False))&(~X['method_clean'].str.endswith('checkout', na=False))&(~X['method_clean'].str.contains('checkout', na=False, regex=False))&((pd.isna(X['ip_address'].to_numpy(na_value=np.nan)))|(X['ip_isp'].fillna('')==''))",
                    'Rule4': "(X['forwarder_address'].to_numpy(na_value=np.nan)==True)&(X['is_shipping_billing_address_same'].to_numpy(na_value=np.nan)==False)",
                    'Rule5': "(~X['ad_price_type'].isin(['FREE', 'NEGOTIATION']))&(X['ad_price_type'].isin(['FOO', 'BAR']))",
                    'Rule6': "(X['ip_country_iso_code'].to_numpy(na_value=np.nan)==X['billing_country'].to_numpy(na_value=np.nan))&(X['country_id'].to_numpy(na_value=np.nan)!=X['ip_country_iso_code'].to_numpy(na_value=np.nan))"}
    return rule_strings


@pytest.fixture()
def _rule_lambdas_with_kwargs():
    rule_lambdas = {
        'Rule1': lambda **kwargs: "((X['payer_id_sum_approved_txn_amt_per_paypalid_1day']>={payer_id_sum_approved_txn_amt_per_paypalid_1day})|(X['payer_id_sum_approved_txn_amt_per_paypalid_7day']>{payer_id_sum_approved_txn_amt_per_paypalid_7day})|(X['payer_id_sum_approved_txn_amt_per_paypalid_30day']<={payer_id_sum_approved_txn_amt_per_paypalid_30day}))&(X['num_items']=={num_items})".format(**kwargs),
        'Rule2': lambda **kwargs: "(X['ml_cc_v0']<{ml_cc_v0})&((X['method_clean']=='checkout')|(X['method_clean'].str.startswith('checkout', na=False))|(X['method_clean'].str.endswith('checkout', na=False))|(X['method_clean'].str.contains('checkout', na=False, regex=False))|(~X['ip_address'].isna())|(X['ip_isp'].fillna('')!=''))".format(**kwargs),
        'Rule3': lambda **kwargs: "(~X['method_clean'].str.startswith('checkout', na=False))&(~X['method_clean'].str.endswith('checkout', na=False))&(~X['method_clean'].str.contains('checkout', na=False, regex=False))&((X['ip_address'].isna())|(X['ip_isp'].fillna('')==''))".format(**kwargs),
        'Rule4': lambda **kwargs: "(X['forwarder_address']==True)&(X['is_shipping_billing_address_same']==False)".format(**kwargs),
        'Rule5': lambda **kwargs: "(~X['ad_price_type'].isin(['FREE', 'NEGOTIATION']))&(X['ad_price_type'].isin(['FOO', 'BAR']))".format(**kwargs),
        'Rule6': lambda **kwargs: "(X['ip_country_iso_code']==X['billing_country'])&(X['country_id']!=X['ip_country_iso_code'])".format(**kwargs)
    }
    lambda_kwargs = {
        'Rule1': {'payer_id_sum_approved_txn_amt_per_paypalid_1day': 60.0, 'payer_id_sum_approved_txn_amt_per_paypalid_7day': 120.0, 'payer_id_sum_approved_txn_amt_per_paypalid_30day': 500, 'num_items': 1.0},
        'Rule2': {'ml_cc_v0': 0.315},
        'Rule3': {},
        'Rule4': {},
        'Rule5': {},
        'Rule6': {},
    }
    return rule_lambdas, lambda_kwargs


@pytest.fixture()
def _rule_lambdas_with_args():
    rule_lambdas = {
        'Rule1': lambda *args: "((X['payer_id_sum_approved_txn_amt_per_paypalid_1day']>={})|(X['payer_id_sum_approved_txn_amt_per_paypalid_7day']>{})|(X['payer_id_sum_approved_txn_amt_per_paypalid_30day']<={}))&(X['num_items']=={})".format(*args),
        'Rule2': lambda *args: "(X['ml_cc_v0']<{})&((X['method_clean']=='checkout')|(X['method_clean'].str.startswith('checkout', na=False))|(X['method_clean'].str.endswith('checkout', na=False))|(X['method_clean'].str.contains('checkout', na=False, regex=False))|(~X['ip_address'].isna())|(X['ip_isp'].fillna('')!=''))".format(*args),
        'Rule3': lambda *args: "(~X['method_clean'].str.startswith('checkout', na=False))&(~X['method_clean'].str.endswith('checkout', na=False))&(~X['method_clean'].str.contains('checkout', na=False, regex=False))&((X['ip_address'].isna())|(X['ip_isp'].fillna('')==''))".format(*args),
        'Rule4': lambda *args: "(X['forwarder_address']==True)&(X['is_shipping_billing_address_same']==False)".format(*args),
        'Rule5': lambda *args: "(~X['ad_price_type'].isin(['FREE', 'NEGOTIATION']))&(X['ad_price_type'].isin(['FOO', 'BAR']))".format(*args),
        'Rule6': lambda *args: "(X['ip_country_iso_code']==X['billing_country'])&(X['country_id']!=X['ip_country_iso_code'])".format(*args)
    }
    lambda_args = {
        'Rule1': [60.0, 120.0, 500, 1.0],
        'Rule2': [0.315],
        'Rule3': [],
        'Rule4': [],
        'Rule5': [],
        'Rule6': [],
    }
    return rule_lambdas, lambda_args


@pytest.fixture
def _system_dicts():
    system_dicts = {'Rule1': {'condition': 'AND',
                              'rules': [{'condition': 'OR',
                                         'rules': [{'id': 'payer_id.sum_approved_txn_amt_per_paypalid_1day',
                                                    'field': 'payer_id.sum_approved_txn_amt_per_paypalid_1day',
                                                    'type': 'double',
                                                    'operator': 'greater_or_equal',
                                                    'value': 'value.60.0'},
                                                   {'id': 'payer_id.sum_approved_txn_amt_per_paypalid_7day',
                                                    'field': 'payer_id.sum_approved_txn_amt_per_paypalid_7day',
                                                    'type': 'double',
                                                    'operator': 'greater',
                                                    'value': 'value.120.0'},
                                                   {'id': 'payer_id.sum_approved_txn_amt_per_paypalid_30day',
                                                    'field': 'payer_id.sum_approved_txn_amt_per_paypalid_30day',
                                                    'type': 'double',
                                                    'operator': 'less_or_equal',
                                                    'value': 'value.500.0'}],
                                         },
                                        {'id': 'num_items',
                                         'field': 'num_items',
                                         'type': 'integer',
                                         'operator': 'equal',
                                         'value': 'value.1'},
                                        ]},
                    'Rule2': {'condition': 'AND',
                              'rules': [{'id': 'ml_cc_v0',
                                         'field': 'ml_cc_v0',
                                         'type': 'double',
                                         'operator': 'less',
                                         'value': 'value.0.315'},
                                        {'condition': 'OR',
                                         'rules': [{'id': 'method_clean',
                                                    'field': 'method_clean',
                                                    'type': 'string',
                                                    'operator': 'equal',
                                                    'value': 'value.checkout'},
                                                   {'id': 'method_clean',
                                                    'field': 'method_clean',
                                                    'type': 'string',
                                                    'operator': 'begins_with',
                                                    'value': 'value.checkout'},
                                                   {'id': 'method_clean',
                                                    'field': 'method_clean',
                                                    'type': 'string',
                                                    'operator': 'ends_with',
                                                    'value': 'value.checkout'},
                                                   {'id': 'method_clean',
                                                    'field': 'method_clean',
                                                    'type': 'string',
                                                    'operator': 'contains',
                                                    'value': 'value.checkout'},
                                                   {'id': 'ip_address',
                                                    'field': 'ip_address',
                                                    'type': 'string',
                                                    'operator': 'is_not_null',
                                                    'value': None},
                                                   {'id': 'ip_isp',
                                                    'field': 'ip_isp',
                                                    'type': 'string',
                                                    'operator': 'is_not_empty',
                                                    'value': None}
                                                   ]}]},
                    'Rule3': {'condition': 'AND',
                              'rules': [{'id': 'method_clean',
                                         'field': 'method_clean',
                                         'type': 'string',
                                         'operator': 'not_begins_with',
                                         'value': 'value.checkout'},
                                        {'id': 'method_clean',
                                         'field': 'method_clean',
                                         'type': 'string',
                                         'operator': 'not_ends_with',
                                         'value': 'value.checkout'},
                                        {'id': 'method_clean',
                                         'field': 'method_clean',
                                         'type': 'string',
                                         'operator': 'not_contains',
                                         'value': 'value.checkout'},
                                        {'condition': 'OR',
                                         'rules': [{'id': 'ip_address',
                                                    'field': 'ip_address',
                                                    'type': 'string',
                                                    'operator': 'is_null',
                                                    'value': None},
                                                   {'id': 'ip_isp',
                                                    'field': 'ip_isp',
                                                    'type': 'string',
                                                    'operator': 'is_empty',
                                                    'value': None},
                                                   ],
                                         }]},
                    'Rule4': {'condition': 'AND',
                              'rules': [{'id': 'forwarder_address',
                                         'field': 'forwarder_address',
                                         'type': 'boolean',
                                         'input': 'radio',
                                         'operator': 'equal',
                                         'value': '1'},
                                        {'id': 'is_shipping_billing_address_same',
                                         'field': 'is_shipping_billing_address_same',
                                         'type': 'boolean',
                                         'input': 'radio',
                                         'operator': 'equal',
                                         'value': '0'}]},
                    'Rule5': {'condition': 'AND',
                              'rules': [{'id': 'ad_price_type',
                                         'field': 'ad_price_type',
                                         'type': 'string',
                                         'operator': 'not_in',
                                         'value': 'value.FREE,NEGOTIATION'},
                                        {'id': 'ad_price_type',
                                         'field': 'ad_price_type',
                                         'type': 'string',
                                         'operator': 'in',
                                         'value': 'value.FOO,BAR'}
                                        ]},
                    'Rule6': {'condition': 'AND',
                              'rules': [{'id': 'ip_country_iso_code',
                                         'field': 'ip_country_iso_code',
                                         'type': 'string',
                                         'operator': 'equal',
                                         'value': 'field.billing_country',
                                         },
                                        {'id': 'country_id',
                                         'field': 'country_id',
                                         'type': 'string',
                                         'operator': 'not_equal',
                                         'value': 'field.ip_country_iso_code',
                                         }]}}
    return system_dicts


@pytest.fixture
def _cassandra_info():
    field_datatypes = {
        'payer_id_sum_approved_txn_amt_per_paypalid_1day': 'DOUBLE',
        'payer_id_sum_approved_txn_amt_per_paypalid_7day': 'DOUBLE',
        'payer_id_sum_approved_txn_amt_per_paypalid_30day': 'DOUBLE',
        'num_items': 'INT',
        'ml_cc_v0': 'DOUBLE',
        'method_clean': 'TEXT',
        'method_clean': 'TEXT',
        'method_clean': 'TEXT',
        'method_clean': 'TEXT',
        'ip_address': 'TEXT',
        'ip_isp': 'TEXT',
        'method_clean': 'TEXT',
        'method_clean': 'TEXT',
        'method_clean': 'TEXT',
        'ip_address': 'TEXT',
        'ip_isp': 'TEXT',
        'forwarder_address': 'BOOLEAN',
        'is_shipping_billing_address_same': 'BOOLEAN',
        'ad_price_type': 'TEXT',
        'ad_price_type': 'TEXT',
        'ip_country_iso_code': 'TEXT',
        'country_id': 'TEXT',
        'billing_country': 'TEXT'
    }
    cassandra_field_names = {
        'payer_id_sum_approved_txn_amt_per_paypalid_1day': 'payer_id.sum_approved_txn_amt_per_paypalid_1day',
        'payer_id_sum_approved_txn_amt_per_paypalid_7day': 'payer_id.sum_approved_txn_amt_per_paypalid_7day',
        'payer_id_sum_approved_txn_amt_per_paypalid_30day': 'payer_id.sum_approved_txn_amt_per_paypalid_30day',
        'num_items': 'num_items',
        'ml_cc_v0': 'ml_cc_v0',
        'method_clean': 'method_clean',
        'method_clean': 'method_clean',
        'method_clean': 'method_clean',
        'method_clean': 'method_clean',
        'ip_address': 'ip_address',
        'ip_isp': 'ip_isp',
        'method_clean': 'method_clean',
        'method_clean': 'method_clean',
        'method_clean': 'method_clean',
        'ip_address': 'ip_address',
        'ip_isp': 'ip_isp',
        'forwarder_address': 'forwarder_address',
        'is_shipping_billing_address_same': 'is_shipping_billing_address_same',
        'ad_price_type': 'ad_price_type',
        'ad_price_type': 'ad_price_type',
        'ip_country_iso_code': 'ip_country_iso_code',
        'country_id': 'country_id',
        'billing_country': 'billing_country'
    }
    return field_datatypes, cassandra_field_names


@pytest.fixture
def _unsupported_system_dicts():
    system_dict = {'Rule1':
                   {'condition': 'OR',
                    'rules': [{'id': 'java_el',
                               'field': 'java_el',
                               'type': 'string',
                               'input': 'text',
                               'operator': 'java_el',
                               'value': '${table_values.$shipping_address_1.toLowerCase().matches(".*(24731 plumtree ct|36508 wells rd)\\\\b.*") }'}]}
                   }
    return system_dict


def test_as_rule_dicts_starting_with_rule_strings(_rule_strings_pandas, _rule_strings_numpy, _rule_dicts):
    rule_strings_pandas = _rule_strings_pandas
    rule_strings_numpy = _rule_strings_numpy
    expected_rule_dicts = _rule_dicts
    r = Rules(rule_strings=rule_strings_pandas)
    rule_dicts = r.as_rule_dicts()
    assert rule_dicts == expected_rule_dicts
    r = Rules(rule_strings=rule_strings_numpy)
    rule_dicts = r.as_rule_dicts()
    assert rule_dicts == expected_rule_dicts


def test_as_rule_dicts_starting_with_system_dicts(_system_dicts, _rule_dicts):
    system_dicts = _system_dicts
    expected_rule_dicts = _rule_dicts
    r = Rules(system_dicts=system_dicts)
    rule_dicts = r.as_rule_dicts()
    assert rule_dicts == expected_rule_dicts


def test_as_rule_dicts_starting_with_rule_lambdas_kwargs(_rule_lambdas_with_kwargs, _rule_dicts):
    rule_lambdas, lambda_kwargs = _rule_lambdas_with_kwargs
    expected_rule_dicts = _rule_dicts
    r = Rules(rule_lambdas=rule_lambdas, lambda_kwargs=lambda_kwargs)
    rule_dicts = r.as_rule_dicts()
    assert rule_dicts == expected_rule_dicts


def test_as_rule_dicts_starting_with_rule_lambdas_args(_rule_lambdas_with_args, _rule_dicts):
    rule_lambdas, lambda_args = _rule_lambdas_with_args
    expected_rule_dicts = _rule_dicts
    r = Rules(rule_lambdas=rule_lambdas, lambda_args=lambda_args)
    rule_dicts = r.as_rule_dicts()
    assert rule_dicts == expected_rule_dicts


def test_as_rule_strings_starting_with_rule_dicts(_rule_dicts, _rule_strings_pandas, _rule_strings_numpy):
    rule_dicts = _rule_dicts
    rule_strings_pandas = _rule_strings_pandas
    rule_strings_numpy = _rule_strings_numpy
    r = Rules(rule_dicts=rule_dicts)
    rule_strings = r.as_rule_strings(as_numpy=False)
    assert rule_strings == rule_strings_pandas
    r = Rules(rule_dicts=rule_dicts)
    rule_strings = r.as_rule_strings(as_numpy=True)
    assert rule_strings == rule_strings_numpy


def test_as_rule_strings_starting_with_system_dicts(_system_dicts, _rule_strings_pandas, _rule_strings_numpy):
    system_dicts = _system_dicts
    rule_strings_pandas = _rule_strings_pandas
    rule_strings_numpy = _rule_strings_numpy
    r = Rules(system_dicts=system_dicts)
    rule_strings = r.as_rule_strings(as_numpy=False)
    assert rule_strings == rule_strings_pandas
    r = Rules(system_dicts=system_dicts)
    rule_strings = r.as_rule_strings(as_numpy=True)
    assert rule_strings == rule_strings_numpy


def test_as_rule_strings_starting_with_rule_lambdas_kwargs(_rule_lambdas_with_kwargs, _rule_strings_pandas, _rule_strings_numpy):
    rule_lambdas, lambda_kwargs = _rule_lambdas_with_kwargs
    rule_strings_pandas = _rule_strings_pandas
    rule_strings_numpy = _rule_strings_numpy
    r = Rules(rule_lambdas=rule_lambdas, lambda_kwargs=lambda_kwargs)
    rule_strings = r.as_rule_strings(as_numpy=False)
    print(type(rule_strings))
    assert rule_strings == rule_strings_pandas
    r = Rules(rule_lambdas=rule_lambdas, lambda_kwargs=lambda_kwargs)
    rule_strings = r.as_rule_strings(as_numpy=True)
    assert rule_strings == rule_strings_numpy


def test_as_rule_strings_starting_with_rule_lambdas_args(_rule_lambdas_with_args, _rule_strings_pandas, _rule_strings_numpy):
    rule_lambdas, lambda_args = _rule_lambdas_with_args
    rule_strings_pandas = _rule_strings_pandas
    rule_strings_numpy = _rule_strings_numpy
    rule_lambdas, lambda_args = _rule_lambdas_with_args
    r = Rules(rule_lambdas=rule_lambdas, lambda_args=lambda_args)
    rule_strings = r.as_rule_strings(as_numpy=False)
    assert rule_strings == rule_strings_pandas
    r = Rules(rule_lambdas=rule_lambdas, lambda_args=lambda_args)
    rule_strings = r.as_rule_strings(as_numpy=True)
    assert rule_strings == rule_strings_numpy


def test_as_rule_lambdas_starting_with_rule_dicts_with_kwargs_True(_rule_dicts, _rule_strings_pandas, _rule_strings_numpy):
    rule_strings_pandas = _rule_strings_pandas
    rule_strings_numpy = _rule_strings_numpy
    rule_dicts = _rule_dicts
    r = Rules(rule_dicts=rule_dicts)
    rule_lambdas = r.as_rule_lambdas(as_numpy=False, with_kwargs=True)
    for rule_name, rule_lambda in rule_lambdas.items():
        rule_string = rule_lambda(**r.lambda_kwargs[rule_name])
        assert rule_string == rule_strings_pandas[rule_name]
    r = Rules(rule_dicts=rule_dicts)
    rule_lambdas = r.as_rule_lambdas(as_numpy=True, with_kwargs=True)
    for rule_name, rule_lambda in rule_lambdas.items():
        rule_string = rule_lambda(**r.lambda_kwargs[rule_name])
        assert rule_string == rule_strings_numpy[rule_name]


def test_as_rule_lambdas_starting_with_rule_dicts_with_kwargs_False(_rule_dicts, _rule_strings_pandas, _rule_strings_numpy):
    rule_strings_pandas = _rule_strings_pandas
    rule_strings_numpy = _rule_strings_numpy
    rule_dicts = _rule_dicts
    r = Rules(rule_dicts=rule_dicts)
    rule_lambdas = r.as_rule_lambdas(as_numpy=False, with_kwargs=False)
    for rule_name, rule_lambda in rule_lambdas.items():
        rule_string = rule_lambda(*r.lambda_args[rule_name])
        assert rule_string == rule_strings_pandas[rule_name]
    r = Rules(rule_dicts=rule_dicts)
    rule_lambdas = r.as_rule_lambdas(as_numpy=True, with_kwargs=False)
    for rule_name, rule_lambda in rule_lambdas.items():
        rule_string = rule_lambda(*r.lambda_args[rule_name])
        assert rule_string == rule_strings_numpy[rule_name]


def test_as_rule_lambdas_starting_with_system_dicts_with_kwargs_True(_system_dicts, _rule_strings_pandas, _rule_strings_numpy):
    rule_strings_pandas = _rule_strings_pandas
    rule_strings_numpy = _rule_strings_numpy
    system_dicts = _system_dicts
    r = Rules(system_dicts=system_dicts)
    rule_lambdas = r.as_rule_lambdas(as_numpy=False, with_kwargs=True)
    for rule_name, rule_lambda in rule_lambdas.items():
        rule_string = rule_lambda(**r.lambda_kwargs[rule_name])
        assert rule_string == rule_strings_pandas[rule_name]
    r = Rules(system_dicts=system_dicts)
    rule_lambdas = r.as_rule_lambdas(as_numpy=True, with_kwargs=True)
    for rule_name, rule_lambda in rule_lambdas.items():
        rule_string = rule_lambda(**r.lambda_kwargs[rule_name])
        assert rule_string == rule_strings_numpy[rule_name]


def test_as_rule_lambdas_starting_with_system_dicts_with_kwargs_False(_system_dicts, _rule_strings_pandas, _rule_strings_numpy):
    rule_strings_pandas = _rule_strings_pandas
    rule_strings_numpy = _rule_strings_numpy
    system_dicts = _system_dicts
    r = Rules(system_dicts=system_dicts)
    rule_lambdas = r.as_rule_lambdas(as_numpy=False, with_kwargs=False)
    for rule_name, rule_lambda in rule_lambdas.items():
        rule_string = rule_lambda(*r.lambda_args[rule_name])
        assert rule_string == rule_strings_pandas[rule_name]
    r = Rules(system_dicts=system_dicts)
    rule_lambdas = r.as_rule_lambdas(as_numpy=True, with_kwargs=False)
    for rule_name, rule_lambda in rule_lambdas.items():
        rule_string = rule_lambda(*r.lambda_args[rule_name])
        assert rule_string == rule_strings_numpy[rule_name]


def test_as_rule_lambdas_starting_with_rule_strings_with_kwargs_True(_rule_strings_pandas, _rule_strings_numpy):
    rule_strings_pandas = _rule_strings_pandas
    rule_strings_numpy = _rule_strings_numpy
    r = Rules(rule_strings=rule_strings_pandas)
    rule_lambdas = r.as_rule_lambdas(as_numpy=False, with_kwargs=True)
    for rule_name, rule_lambda in rule_lambdas.items():
        rule_string = rule_lambda(**r.lambda_kwargs[rule_name])
        assert rule_string == rule_strings_pandas[rule_name]
    r = Rules(rule_strings=rule_strings_numpy)
    rule_lambdas = r.as_rule_lambdas(as_numpy=True, with_kwargs=True)
    for rule_name, rule_lambda in rule_lambdas.items():
        rule_string = rule_lambda(**r.lambda_kwargs[rule_name])
        assert rule_string == rule_strings_numpy[rule_name]
    r = Rules(rule_strings=rule_strings_numpy)
    rule_lambdas = r.as_rule_lambdas(as_numpy=False, with_kwargs=True)
    for rule_name, rule_lambda in rule_lambdas.items():
        rule_string = rule_lambda(**r.lambda_kwargs[rule_name])
        assert rule_string == rule_strings_pandas[rule_name]
    r = Rules(rule_strings=rule_strings_numpy)
    rule_lambdas = r.as_rule_lambdas(as_numpy=True, with_kwargs=True)
    for rule_name, rule_lambda in rule_lambdas.items():
        rule_string = rule_lambda(**r.lambda_kwargs[rule_name])
        assert rule_string == rule_strings_numpy[rule_name]


def test_as_rule_lambdas_starting_with_rule_strings_with_kwargs_False(_rule_strings_pandas, _rule_strings_numpy):
    rule_strings_pandas = _rule_strings_pandas
    rule_strings_numpy = _rule_strings_numpy
    r = Rules(rule_strings=rule_strings_pandas)
    rule_lambdas = r.as_rule_lambdas(as_numpy=False, with_kwargs=False)
    for rule_name, rule_lambda in rule_lambdas.items():
        rule_string = rule_lambda(*r.lambda_args[rule_name])
        assert rule_string == rule_strings_pandas[rule_name]
    r = Rules(rule_strings=rule_strings_numpy)
    rule_lambdas = r.as_rule_lambdas(as_numpy=True, with_kwargs=False)
    for rule_name, rule_lambda in rule_lambdas.items():
        rule_string = rule_lambda(*r.lambda_args[rule_name])
        assert rule_string == rule_strings_numpy[rule_name]
    r = Rules(rule_strings=rule_strings_numpy)
    rule_lambdas = r.as_rule_lambdas(as_numpy=False, with_kwargs=False)
    for rule_name, rule_lambda in rule_lambdas.items():
        rule_string = rule_lambda(*r.lambda_args[rule_name])
        assert rule_string == rule_strings_pandas[rule_name]
    r = Rules(rule_strings=rule_strings_numpy)
    rule_lambdas = r.as_rule_lambdas(as_numpy=True, with_kwargs=False)
    for rule_name, rule_lambda in rule_lambdas.items():
        rule_string = rule_lambda(*r.lambda_args[rule_name])
        assert rule_string == rule_strings_numpy[rule_name]


def test_as_rule_lambdas_starting_with_rule_lambdas_with_kwargs_True(_rule_lambdas_with_kwargs, _rule_lambdas_with_args,
                                                                     _rule_strings_pandas, _rule_strings_numpy):
    rule_strings_pandas = _rule_strings_pandas
    rule_strings_numpy = _rule_strings_numpy
    # Starting with lambda_kwargs
    rule_lambdas, lambda_kwargs = _rule_lambdas_with_kwargs
    r = Rules(rule_lambdas=rule_lambdas, lambda_kwargs=lambda_kwargs)
    rule_lambdas_ = r.as_rule_lambdas(as_numpy=False, with_kwargs=True)
    for rule_name, rule_lambda in rule_lambdas_.items():
        rule_string = rule_lambda(**r.lambda_kwargs[rule_name])
        assert rule_string == rule_strings_pandas[rule_name]
    r = Rules(rule_lambdas=rule_lambdas, lambda_kwargs=lambda_kwargs)
    rule_lambdas_ = r.as_rule_lambdas(as_numpy=True, with_kwargs=True)
    for rule_name, rule_lambda in rule_lambdas_.items():
        rule_string = rule_lambda(**r.lambda_kwargs[rule_name])
        assert rule_string == rule_strings_numpy[rule_name]
    # Starting with lambda_args
    rule_lambdas, lambda_args = _rule_lambdas_with_args
    r = Rules(rule_lambdas=rule_lambdas, lambda_args=lambda_args)
    rule_lambdas_ = r.as_rule_lambdas(as_numpy=False, with_kwargs=True)
    for rule_name, rule_lambda in rule_lambdas_.items():
        rule_string = rule_lambda(**r.lambda_kwargs[rule_name])
        assert rule_string == rule_strings_pandas[rule_name]
    r = Rules(rule_lambdas=rule_lambdas, lambda_args=lambda_args)
    rule_lambdas_ = r.as_rule_lambdas(as_numpy=True, with_kwargs=True)
    for rule_name, rule_lambda in rule_lambdas_.items():
        rule_string = rule_lambda(**r.lambda_kwargs[rule_name])
        assert rule_string == rule_strings_numpy[rule_name]


def test_as_rule_lambdas_starting_with_rule_lambdas_with_kwargs_False(_rule_lambdas_with_kwargs, _rule_lambdas_with_args,
                                                                      _rule_strings_pandas, _rule_strings_numpy):
    rule_strings_pandas = _rule_strings_pandas
    rule_strings_numpy = _rule_strings_numpy
    # Starting with lambda_kwargs
    rule_lambdas, lambda_kwargs = _rule_lambdas_with_kwargs
    r = Rules(rule_lambdas=rule_lambdas, lambda_kwargs=lambda_kwargs)
    rule_lambdas_ = r.as_rule_lambdas(as_numpy=False, with_kwargs=False)
    for rule_name, rule_lambda in rule_lambdas_.items():
        rule_string = rule_lambda(*r.lambda_args[rule_name])
        assert rule_string == rule_strings_pandas[rule_name]
    r = Rules(rule_lambdas=rule_lambdas, lambda_kwargs=lambda_kwargs)
    rule_lambdas_ = r.as_rule_lambdas(as_numpy=True, with_kwargs=False)
    for rule_name, rule_lambda in rule_lambdas_.items():
        rule_string = rule_lambda(*r.lambda_args[rule_name])
        assert rule_string == rule_strings_numpy[rule_name]
    # Starting with lambda_args
    rule_lambdas, lambda_args = _rule_lambdas_with_args
    r = Rules(rule_lambdas=rule_lambdas, lambda_args=lambda_args)
    rule_lambdas_ = r.as_rule_lambdas(as_numpy=False, with_kwargs=False)
    for rule_name, rule_lambda in rule_lambdas_.items():
        rule_string = rule_lambda(*r.lambda_args[rule_name])
        assert rule_string == rule_strings_pandas[rule_name]
    r = Rules(rule_lambdas=rule_lambdas, lambda_args=lambda_args)
    rule_lambdas_ = r.as_rule_lambdas(as_numpy=True, with_kwargs=False)
    for rule_name, rule_lambda in rule_lambdas_.items():
        rule_string = rule_lambda(*r.lambda_args[rule_name])
        assert rule_string == rule_strings_numpy[rule_name]


def test_as_system_dicts_starting_with_rule_dicts(_rule_dicts, _system_dicts, _cassandra_info):
    rule_dicts = _rule_dicts
    expected_system_dicts = _system_dicts
    field_datatypes, cassandra_field_names = _cassandra_info
    r = Rules(rule_dicts=rule_dicts)
    system_dicts = r.as_system_dicts(field_datatypes, cassandra_field_names)
    assert system_dicts == expected_system_dicts


def test_as_system_dicts_starting_with_rule_strings(_rule_strings_pandas, _rule_strings_numpy, _system_dicts, _cassandra_info):
    rule_strings_pandas = _rule_strings_pandas
    rule_strings_numpy = _rule_strings_numpy
    expected_system_dicts = _system_dicts
    field_datatypes, cassandra_field_names = _cassandra_info
    r = Rules(rule_strings=rule_strings_pandas)
    system_dicts = r.as_system_dicts(field_datatypes, cassandra_field_names)
    assert system_dicts == expected_system_dicts
    r = Rules(rule_strings=rule_strings_numpy)
    system_dicts = r.as_system_dicts(field_datatypes, cassandra_field_names)
    assert system_dicts == expected_system_dicts


def test_as_system_dicts_starting_with_rule_lambdas_kwargs(_rule_lambdas_with_kwargs, _system_dicts, _cassandra_info):
    rule_lambdas, lambda_kwargs = _rule_lambdas_with_kwargs
    expected_system_dicts = _system_dicts
    field_datatypes, cassandra_field_names = _cassandra_info
    r = Rules(rule_lambdas=rule_lambdas, lambda_kwargs=lambda_kwargs)
    system_dicts = r.as_system_dicts(field_datatypes, cassandra_field_names)
    assert system_dicts == expected_system_dicts


def test_as_system_dicts_starting_with_rule_lambdas_args(_rule_lambdas_with_args, _system_dicts, _cassandra_info):
    rule_lambdas, lambda_args = _rule_lambdas_with_args
    expected_system_dicts = _system_dicts
    field_datatypes, cassandra_field_names = _cassandra_info
    r = Rules(rule_lambdas=rule_lambdas, lambda_args=lambda_args)
    system_dicts = r.as_system_dicts(field_datatypes, cassandra_field_names)
    assert system_dicts == expected_system_dicts


def test_filter_rules(_rule_strings_pandas, _cassandra_info):
    rule_strings = _rule_strings_pandas
    field_datatypes, cassandra_field_names = _cassandra_info
    r = Rules(rule_strings=rule_strings)
    r.as_rule_dicts()
    r.as_rule_lambdas(as_numpy=True, with_kwargs=True)
    r.as_system_dicts(field_datatypes, cassandra_field_names)
    inc_rule_names = ['Rule1', 'Rule2', 'Rule3']
    r.filter_rules(include=inc_rule_names)
    assert list(r.rule_strings.keys()) == inc_rule_names
    assert list(r.rule_dicts.keys()) == inc_rule_names
    assert list(r.rule_lambdas.keys()) == inc_rule_names
    assert list(r.lambda_kwargs.keys()) == inc_rule_names
    assert list(r.lambda_args.keys()) == inc_rule_names
    assert list(r.rule_features.keys()) == inc_rule_names
    assert list(r.system_dicts.keys()) == inc_rule_names
    exc_rule_names = ['Rule2', 'Rule3']
    r.filter_rules(exclude=exc_rule_names)
    assert list(r.rule_strings.keys()) == ['Rule1']
    assert list(r.rule_dicts.keys()) == ['Rule1']
    assert list(r.rule_lambdas.keys()) == ['Rule1']
    assert list(r.lambda_kwargs.keys()) == ['Rule1']
    assert list(r.lambda_args.keys()) == ['Rule1']
    assert list(r.rule_features.keys()) == ['Rule1']
    assert list(r.system_dicts.keys()) == ['Rule1']


def test_filter_rules_rule_strings_only(_rule_strings_pandas, _cassandra_info):
    rule_strings = _rule_strings_pandas
    r = Rules(rule_strings=rule_strings)
    inc_rule_names = ['Rule1', 'Rule2', 'Rule3']
    r.filter_rules(include=inc_rule_names)
    assert list(r.rule_strings.keys()) == inc_rule_names
    exc_rule_names = ['Rule2', 'Rule3']
    r.filter_rules(exclude=exc_rule_names)
    assert list(r.rule_strings.keys()) == ['Rule1']


def test_filter_rules_rule_dicts_only(_rule_dicts, _cassandra_info):
    rule_dicts = _rule_dicts
    r = Rules(rule_dicts=rule_dicts)
    inc_rule_names = ['Rule1', 'Rule2', 'Rule3']
    r.filter_rules(include=inc_rule_names)
    assert list(r.rule_dicts.keys()) == inc_rule_names
    exc_rule_names = ['Rule2', 'Rule3']
    r.filter_rules(exclude=exc_rule_names)
    assert list(r.rule_dicts.keys()) == ['Rule1']


def test_filter_rules_system_dicts_only(_system_dicts, _cassandra_info):
    system_dicts = _system_dicts
    r = Rules(system_dicts=system_dicts)
    inc_rule_names = ['Rule1', 'Rule2', 'Rule3']
    r.filter_rules(include=inc_rule_names)
    assert list(r.system_dicts.keys()) == inc_rule_names
    exc_rule_names = ['Rule2', 'Rule3']
    r.filter_rules(exclude=exc_rule_names)
    assert list(r.system_dicts.keys()) == ['Rule1']


def test_get_rule_features(_rule_dicts):
    exp_result = {'Rule1': {'num_items',
                            'payer_id_sum_approved_txn_amt_per_paypalid_1day',
                            'payer_id_sum_approved_txn_amt_per_paypalid_30day',
                            'payer_id_sum_approved_txn_amt_per_paypalid_7day'},
                  'Rule2': {'ip_address', 'ip_isp', 'method_clean', 'ml_cc_v0'},
                  'Rule3': {'ip_address', 'ip_isp', 'method_clean'},
                  'Rule4': {'forwarder_address', 'is_shipping_billing_address_same'},
                  'Rule5': {'ad_price_type'},
                  'Rule6': {'billing_country', 'country_id', 'ip_country_iso_code'}}
    rule_dicts = _rule_dicts
    r = Rules(rule_dicts=rule_dicts)
    rule_features = r.get_rule_features()
    assert rule_features == exp_result


def test_errors():
    with pytest.raises(ValueError):
        r = Rules()
    with pytest.raises(ValueError):
        r = Rules(rule_strings="X['A']>2")
        r._rule_dicts_to_system_dicts(None, None)
    with pytest.raises(ValueError):
        r = Rules(rule_strings="X['A']>2")
        r._system_dicts_to_rule_dicts()
    with pytest.raises(ValueError):
        r = Rules(rule_strings="X['A']>2")
        r._rule_dicts_to_rule_strings(as_numpy=False)
    with pytest.raises(ValueError):
        r = Rules(rule_dicts={'ABC': {}})
        r._rule_strings_to_rule_dicts()
    with pytest.raises(ValueError):
        r = Rules(rule_strings={'ABC': {}})
        r._rule_dicts_to_rule_lambdas(as_numpy=True, with_kwargs=True)
    with pytest.raises(Exception):
        r = Rules(rule_strings={'Rule1': {}})
        r.filter_rules(include=['Rule1'], exclude=['Rule1'])


def test_warnings(_unsupported_system_dicts):
    unsupported_system_dict = _unsupported_system_dicts
    with pytest.warns(UserWarning):
        r = Rules(system_dicts=unsupported_system_dict)
        r.as_rule_dicts()
