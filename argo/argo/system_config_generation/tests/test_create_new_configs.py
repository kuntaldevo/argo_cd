import pytest
import json
from system_config_generation.create_new_configs import CreateNewConfigs
import numpy as np


@pytest.fixture
def _data():
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
    scores = {
        'Rule1': np.int64(-100),
        'Rule2': np.int64(-50),
        'Rule3': np.int64(-25),
        'Rule4': np.int64(-12),
        'Rule5': np.int64(-6),
        'Rule6': np.int64(-3),
    }
    return system_dicts, scores


@pytest.fixture
def _rule_configs():
    rule_configs = {
        'Rule1': {'createdBy': 'argo@simility.com',
                  'createdOn': '2020-12-16T10:38:24',
                  'appPrefix': 'james_testing',
                  'entityName': 'transaction',
                  'name': 'Rule1',
                  'conditions': '{"condition": "AND", "rules": [{"condition": "OR", "rules": [{"id": "payer_id.sum_approved_txn_amt_per_paypalid_1day", "field": "payer_id.sum_approved_txn_amt_per_paypalid_1day", "type": "double", "operator": "greater_or_equal", "value": "value.60.0"}, {"id": "payer_id.sum_approved_txn_amt_per_paypalid_7day", "field": "payer_id.sum_approved_txn_amt_per_paypalid_7day", "type": "double", "operator": "greater", "value": "value.120.0"}, {"id": "payer_id.sum_approved_txn_amt_per_paypalid_30day", "field": "payer_id.sum_approved_txn_amt_per_paypalid_30day", "type": "double", "operator": "less_or_equal", "value": "value.500.0"}]}, {"id": "num_items", "field": "num_items", "type": "integer", "operator": "equal", "value": "value.1"}]}',
                  'score': -100,
                  'status': 'ACTIVE',
                  'isAutoGenerated': True},
        'Rule2': {'createdBy': 'argo@simility.com',
                  'createdOn': '2020-12-16T10:38:24',
                  'appPrefix': 'james_testing',
                  'entityName': 'transaction',
                  'name': 'Rule2',
                  'conditions': '{"condition": "AND", "rules": [{"id": "ml_cc_v0", "field": "ml_cc_v0", "type": "double", "operator": "less", "value": "value.0.315"}, {"condition": "OR", "rules": [{"id": "method_clean", "field": "method_clean", "type": "string", "operator": "equal", "value": "value.checkout"}, {"id": "method_clean", "field": "method_clean", "type": "string", "operator": "begins_with", "value": "value.checkout"}, {"id": "method_clean", "field": "method_clean", "type": "string", "operator": "ends_with", "value": "value.checkout"}, {"id": "method_clean", "field": "method_clean", "type": "string", "operator": "contains", "value": "value.checkout"}, {"id": "ip_address", "field": "ip_address", "type": "string", "operator": "is_not_null", "value": null}, {"id": "ip_isp", "field": "ip_isp", "type": "string", "operator": "is_not_empty", "value": null}]}]}',
                  'score': -50,
                  'status': 'ACTIVE',
                  'isAutoGenerated': True},
        'Rule3': {'createdBy': 'argo@simility.com',
                  'createdOn': '2020-12-16T10:38:24',
                  'appPrefix': 'james_testing',
                  'entityName': 'transaction',
                  'name': 'Rule3',
                  'conditions': '{"condition": "AND", "rules": [{"id": "method_clean", "field": "method_clean", "type": "string", "operator": "not_begins_with", "value": "value.checkout"}, {"id": "method_clean", "field": "method_clean", "type": "string", "operator": "not_ends_with", "value": "value.checkout"}, {"id": "method_clean", "field": "method_clean", "type": "string", "operator": "not_contains", "value": "value.checkout"}, {"condition": "OR", "rules": [{"id": "ip_address", "field": "ip_address", "type": "string", "operator": "is_null", "value": null}, {"id": "ip_isp", "field": "ip_isp", "type": "string", "operator": "is_empty", "value": null}]}]}',
                  'score': -25,
                  'status': 'ACTIVE',
                  'isAutoGenerated': True},
        'Rule4': {'createdBy': 'argo@simility.com',
                  'createdOn': '2020-12-16T10:38:24',
                  'appPrefix': 'james_testing',
                  'entityName': 'transaction',
                  'name': 'Rule4',
                  'conditions': '{"condition": "AND", "rules": [{"id": "forwarder_address", "field": "forwarder_address", "type": "boolean", "input": "radio", "operator": "equal", "value": "1"}, {"id": "is_shipping_billing_address_same", "field": "is_shipping_billing_address_same", "type": "boolean", "input": "radio", "operator": "equal", "value": "0"}]}',
                  'score': -12,
                  'status': 'ACTIVE',
                  'isAutoGenerated': True},
        'Rule5': {'createdBy': 'argo@simility.com',
                  'createdOn': '2020-12-16T10:38:24',
                  'appPrefix': 'james_testing',
                  'entityName': 'transaction',
                  'name': 'Rule5',
                  'conditions': '{"condition": "AND", "rules": [{"id": "ad_price_type", "field": "ad_price_type", "type": "string", "operator": "not_in", "value": "value.FREE,NEGOTIATION"}, {"id": "ad_price_type", "field": "ad_price_type", "type": "string", "operator": "in", "value": "value.FOO,BAR"}]}',
                  'score': -6,
                  'status': 'ACTIVE',
                  'isAutoGenerated': True},
        'Rule6': {'createdBy': 'argo@simility.com',
                  'createdOn': '2020-12-16T10:38:24',
                  'appPrefix': 'james_testing',
                  'entityName': 'transaction',
                  'name': 'Rule6',
                  'conditions': '{"condition": "AND", "rules": [{"id": "ip_country_iso_code", "field": "ip_country_iso_code", "type": "string", "operator": "equal", "value": "field.billing_country"}, {"id": "country_id", "field": "country_id", "type": "string", "operator": "not_equal", "value": "field.ip_country_iso_code"}]}',
                  'score': -3,
                  'status': 'ACTIVE',
                  'isAutoGenerated': True}
    }
    return rule_configs


def test_generate(_data, _rule_configs):
    system_dicts, scores = _data
    expected_rule_configs = _rule_configs
    c = CreateNewConfigs(conditions=system_dicts,
                         scores=scores,
                         app_prefix='james_testing',
                         entity='transaction',
                         make_active=True)
    rule_configs = c.generate()
    # Manually change createdOn date to match expected
    for rule_config in rule_configs.values():
        rule_config['createdOn'] = '2020-12-16T10:38:24'
    assert rule_configs == expected_rule_configs
