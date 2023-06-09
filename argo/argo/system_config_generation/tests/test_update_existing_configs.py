import pytest
from system_config_generation.update_existing_configs import UpdateExistingConfigs
import json
import numpy as np


@pytest.fixture
def _rule_configs():
    rule_configs = {'Rule1': {'createdBy': 'argo@simility.com',
                              'createdOn': '2020-12-16T15:00:17',
                              'appPrefix': 'james_testing',
                              'entityName': 'transaction',
                              'name': 'Rule1',
                              'conditions': '{"condition": "AND", "rules": [{"condition": "OR", "rules": [{"id": "payer_id.sum_approved_txn_amt_per_paypalid_1day", "field": "payer_id.sum_approved_txn_amt_per_paypalid_1day", "type": "double", "operator": "greater_or_equal", "value": "value.60.0"}, {"id": "payer_id.sum_approved_txn_amt_per_paypalid_7day", "field": "payer_id.sum_approved_txn_amt_per_paypalid_7day", "type": "double", "operator": "greater", "value": "value.120.0"}, {"id": "payer_id.sum_approved_txn_amt_per_paypalid_30day", "field": "payer_id.sum_approved_txn_amt_per_paypalid_30day", "type": "double", "operator": "less_or_equal", "value": "value.500.0"}]}, {"id": "num_items", "field": "num_items", "type": "integer", "operator": "equal", "value": "value.1"}]}',
                              'score': -100,
                              'status': 'ACTIVE',
                              'isAutoGenerated': True},
                    'Rule2': {'createdBy': 'argo@simility.com',
                              'createdOn': '2020-12-16T15:00:17',
                              'appPrefix': 'james_testing',
                              'entityName': 'transaction',
                              'name': 'Rule2',
                              'conditions': '{"condition": "AND", "rules": [{"id": "ml_cc_v0", "field": "ml_cc_v0", "type": "double", "operator": "less", "value": "value.0.315"}, {"condition": "OR", "rules": [{"id": "method_clean", "field": "method_clean", "type": "string", "operator": "equal", "value": "value.checkout"}, {"id": "method_clean", "field": "method_clean", "type": "string", "operator": "begins_with", "value": "value.checkout"}, {"id": "method_clean", "field": "method_clean", "type": "string", "operator": "ends_with", "value": "value.checkout"}, {"id": "method_clean", "field": "method_clean", "type": "string", "operator": "contains", "value": "value.checkout"}, {"id": "ip_address", "field": "ip_address", "type": "string", "operator": "is_not_null", "value": null}, {"id": "ip_isp", "field": "ip_isp", "type": "string", "operator": "is_not_empty", "value": null}]}]}',
                              'score': -100,
                              'status': 'ACTIVE',
                              'isAutoGenerated': True},
                    'Rule3': {'createdBy': 'argo@simility.com',
                              'createdOn': '2020-12-16T15:00:17',
                              'appPrefix': 'james_testing',
                              'entityName': 'transaction',
                              'name': 'Rule3',
                              'conditions': '{"condition": "AND", "rules": [{"id": "method_clean", "field": "method_clean", "type": "string", "operator": "not_begins_with", "value": "value.checkout"}, {"id": "method_clean", "field": "method_clean", "type": "string", "operator": "not_ends_with", "value": "value.checkout"}, {"id": "method_clean", "field": "method_clean", "type": "string", "operator": "not_contains", "value": "value.checkout"}, {"condition": "OR", "rules": [{"id": "ip_address", "field": "ip_address", "type": "string", "operator": "is_null", "value": null}, {"id": "ip_isp", "field": "ip_isp", "type": "string", "operator": "is_empty", "value": null}]}]}',
                              'score': -100,
                              'status': 'ACTIVE',
                              'isAutoGenerated': True},
                    'Rule4': {'createdBy': 'argo@simility.com',
                              'createdOn': '2020-12-16T15:00:17',
                              'appPrefix': 'james_testing',
                              'entityName': 'transaction',
                              'name': 'Rule4',
                              'conditions': '{"condition": "AND", "rules": [{"id": "forwarder_address", "field": "forwarder_address", "type": "boolean", "input": "radio", "operator": "equal", "value": "1"}, {"id": "is_shipping_billing_address_same", "field": "is_shipping_billing_address_same", "type": "boolean", "input": "radio", "operator": "equal", "value": "0"}]}',
                              'score': -100,
                              'status': 'ACTIVE',
                              'isAutoGenerated': True},
                    'Rule5': {'createdBy': 'argo@simility.com',
                              'createdOn': '2020-12-16T15:00:17',
                              'appPrefix': 'james_testing',
                              'entityName': 'transaction',
                              'name': 'Rule5',
                              'conditions': '{"condition": "AND", "rules": [{"id": "ad_price_type", "field": "ad_price_type", "type": "string", "operator": "not_in", "value": "value.FREE,NEGOTIATION"}, {"id": "ad_price_type", "field": "ad_price_type", "type": "string", "operator": "in", "value": "value.FOO,BAR"}]}',
                              'score': -100,
                              'status': 'ACTIVE',
                              'isAutoGenerated': True},
                    'Rule6': {'createdBy': 'argo@simility.com',
                              'createdOn': '2020-12-16T15:00:17',
                              'appPrefix': 'james_testing',
                              'entityName': 'transaction',
                              'name': 'Rule6',
                              'conditions': '{"condition": "AND", "rules": [{"id": "ip_country_iso_code", "field": "ip_country_iso_code", "type": "string", "operator": "equal", "value": "field.billing_country"}, {"id": "country_id", "field": "country_id", "type": "string", "operator": "not_equal", "value": "field.ip_country_iso_code"}]}',
                              'score': -100,
                              'status': 'ACTIVE',
                              'isAutoGenerated': True}}
    return rule_configs


@pytest.fixture
def _new_conditions():
    new_conditions = {'Rule1': {'condition': 'AND',
                                'rules': [{'condition': 'OR',
                                           'rules': [{'id': 'payer_id.sum_approved_txn_amt_per_paypalid_1day',
                                                      'field': 'payer_id.sum_approved_txn_amt_per_paypalid_1day',
                                                      'type': 'double',
                                                      'operator': 'greater_or_equal',
                                                      'value': 'value.120.0'},
                                                     {'id': 'payer_id.sum_approved_txn_amt_per_paypalid_7day',
                                                      'field': 'payer_id.sum_approved_txn_amt_per_paypalid_7day',
                                                      'type': 'double',
                                                      'operator': 'greater',
                                                      'value': 'value.240.0'},
                                                     {'id': 'payer_id.sum_approved_txn_amt_per_paypalid_30day',
                                                      'field': 'payer_id.sum_approved_txn_amt_per_paypalid_30day',
                                                      'type': 'double',
                                                      'operator': 'less_or_equal',
                                                      'value': 'value.1000.0'}],
                                           },
                                          {'id': 'num_items',
                                           'field': 'num_items',
                                           'type': 'integer',
                                           'operator': 'equal',
                                           'value': 'value.2'},
                                          ]},
                      'Rule2': {'condition': 'AND',
                                'rules': [{'id': 'ml_cc_v0',
                                           'field': 'ml_cc_v0',
                                           'type': 'double',
                                           'operator': 'less',
                                           'value': 'value.0.75'},
                                          {'condition': 'OR',
                                           'rules': [{'id': 'method_clean',
                                                      'field': 'method_clean',
                                                      'type': 'string',
                                                      'operator': 'equal',
                                                      'value': 'value.login'},
                                                     {'id': 'method_clean',
                                                      'field': 'method_clean',
                                                      'type': 'string',
                                                      'operator': 'begins_with',
                                                      'value': 'value.login'},
                                                     {'id': 'method_clean',
                                                      'field': 'method_clean',
                                                      'type': 'string',
                                                      'operator': 'ends_with',
                                                      'value': 'value.login'},
                                                     {'id': 'method_clean',
                                                      'field': 'method_clean',
                                                      'type': 'string',
                                                      'operator': 'contains',
                                                      'value': 'value.login'},
                                                     {'id': 'ip_address',
                                                      'field': 'ip_address',
                                                      'type': 'string',
                                                      'operator': 'is_null',
                                                      'value': None},
                                                     {'id': 'ip_isp',
                                                      'field': 'ip_isp',
                                                      'type': 'string',
                                                      'operator': 'is_empty',
                                                      'value': None}
                                                     ]}]},
                      'Rule3': {'condition': 'AND',
                                'rules': [{'id': 'method_clean',
                                           'field': 'method_clean',
                                           'type': 'string',
                                           'operator': 'not_begins_with',
                                           'value': 'value.login'},
                                          {'id': 'method_clean',
                                           'field': 'method_clean',
                                           'type': 'string',
                                           'operator': 'not_ends_with',
                                           'value': 'value.login'},
                                          {'id': 'method_clean',
                                           'field': 'method_clean',
                                           'type': 'string',
                                           'operator': 'not_contains',
                                           'value': 'value.login'},
                                          {'condition': 'OR',
                                           'rules': [{'id': 'ip_address',
                                                      'field': 'ip_address',
                                                      'type': 'string',
                                                      'operator': 'is_not_null',
                                                      'value': None},
                                                     {'id': 'ip_isp',
                                                      'field': 'ip_isp',
                                                      'type': 'string',
                                                      'operator': 'is_not_empty',
                                                      'value': None},
                                                     ],
                                           }]},
                      'Rule4': {'condition': 'AND',
                                'rules': [{'id': 'forwarder_address',
                                           'field': 'forwarder_address',
                                           'type': 'boolean',
                                           'input': 'radio',
                                           'operator': 'equal',
                                           'value': '0'},
                                          {'id': 'is_shipping_billing_address_same',
                                           'field': 'is_shipping_billing_address_same',
                                           'type': 'boolean',
                                           'input': 'radio',
                                           'operator': 'equal',
                                           'value': '1'}]},
                      'Rule5': {'condition': 'AND',
                                'rules': [{'id': 'ad_price_type',
                                           'field': 'ad_price_type',
                                           'type': 'string',
                                           'operator': 'in',
                                           'value': 'value.FREE,NEGOTIATION'},
                                          {'id': 'ad_price_type',
                                           'field': 'ad_price_type',
                                           'type': 'string',
                                           'operator': 'not_in',
                                           'value': 'value.FOO,BAR'}
                                          ]},
                      'Rule6': {'condition': 'AND',
                                'rules': [{'id': 'ip_country_iso_code',
                                           'field': 'ip_country_iso_code',
                                           'type': 'string',
                                           'operator': 'not_equal',
                                           'value': 'field.billing_country',
                                           },
                                          {'id': 'country_id',
                                           'field': 'country_id',
                                           'type': 'string',
                                           'operator': 'equal',
                                           'value': 'field.ip_country_iso_code',
                                           }]}}
    return new_conditions


@pytest.fixture
def _new_scores():
    new_scores = {
        'Rule1': np.int64(-50),
        'Rule2': np.int64(-50),
        'Rule3': np.int64(-50),
        'Rule4': np.int64(-50),
        'Rule5': np.int64(-50),
        'Rule6': np.int64(-50),
    }
    return new_scores


@pytest.fixture
def _expected_rule_configs():
    expected_rule_configs = {
        'Rule1': {'createdBy': 'argo@simility.com',
                  'createdOn': '2020-12-16T15:00:17',
                  'appPrefix': 'james_testing',
                  'entityName': 'transaction',
                  'name': 'Rule1',
                  'conditions': '{"condition": "AND", "rules": [{"condition": "OR", "rules": [{"id": "payer_id.sum_approved_txn_amt_per_paypalid_1day", "field": "payer_id.sum_approved_txn_amt_per_paypalid_1day", "type": "double", "operator": "greater_or_equal", "value": "value.120.0"}, {"id": "payer_id.sum_approved_txn_amt_per_paypalid_7day", "field": "payer_id.sum_approved_txn_amt_per_paypalid_7day", "type": "double", "operator": "greater", "value": "value.240.0"}, {"id": "payer_id.sum_approved_txn_amt_per_paypalid_30day", "field": "payer_id.sum_approved_txn_amt_per_paypalid_30day", "type": "double", "operator": "less_or_equal", "value": "value.1000.0"}]}, {"id": "num_items", "field": "num_items", "type": "integer", "operator": "equal", "value": "value.2"}]}',
                  'score': -50,
                  'status': 'ACTIVE',
                  'isAutoGenerated': True,
                  'modifiedBy': 'argo@simility.com',
                  'modifiedOn': '2020-12-21T15:08:02'},
        'Rule2': {'createdBy': 'argo@simility.com',
                  'createdOn': '2020-12-16T15:00:17',
                  'appPrefix': 'james_testing',
                  'entityName': 'transaction',
                  'name': 'Rule2',
                  'conditions': '{"condition": "AND", "rules": [{"id": "ml_cc_v0", "field": "ml_cc_v0", "type": "double", "operator": "less", "value": "value.0.75"}, {"condition": "OR", "rules": [{"id": "method_clean", "field": "method_clean", "type": "string", "operator": "equal", "value": "value.login"}, {"id": "method_clean", "field": "method_clean", "type": "string", "operator": "begins_with", "value": "value.login"}, {"id": "method_clean", "field": "method_clean", "type": "string", "operator": "ends_with", "value": "value.login"}, {"id": "method_clean", "field": "method_clean", "type": "string", "operator": "contains", "value": "value.login"}, {"id": "ip_address", "field": "ip_address", "type": "string", "operator": "is_null", "value": null}, {"id": "ip_isp", "field": "ip_isp", "type": "string", "operator": "is_empty", "value": null}]}]}',
                  'score': -50,
                  'status': 'ACTIVE',
                  'isAutoGenerated': True,
                  'modifiedBy': 'argo@simility.com',
                  'modifiedOn': '2020-12-21T15:08:02'},
        'Rule3': {'createdBy': 'argo@simility.com',
                  'createdOn': '2020-12-16T15:00:17',
                  'appPrefix': 'james_testing',
                  'entityName': 'transaction',
                  'name': 'Rule3',
                  'conditions': '{"condition": "AND", "rules": [{"id": "method_clean", "field": "method_clean", "type": "string", "operator": "not_begins_with", "value": "value.login"}, {"id": "method_clean", "field": "method_clean", "type": "string", "operator": "not_ends_with", "value": "value.login"}, {"id": "method_clean", "field": "method_clean", "type": "string", "operator": "not_contains", "value": "value.login"}, {"condition": "OR", "rules": [{"id": "ip_address", "field": "ip_address", "type": "string", "operator": "is_not_null", "value": null}, {"id": "ip_isp", "field": "ip_isp", "type": "string", "operator": "is_not_empty", "value": null}]}]}',
                  'score': -50,
                  'status': 'ACTIVE',
                  'isAutoGenerated': True,
                  'modifiedBy': 'argo@simility.com',
                  'modifiedOn': '2020-12-21T15:08:02'},
        'Rule4': {'createdBy': 'argo@simility.com',
                  'createdOn': '2020-12-16T15:00:17',
                  'appPrefix': 'james_testing',
                  'entityName': 'transaction',
                  'name': 'Rule4',
                  'conditions': '{"condition": "AND", "rules": [{"id": "forwarder_address", "field": "forwarder_address", "type": "boolean", "input": "radio", "operator": "equal", "value": "0"}, {"id": "is_shipping_billing_address_same", "field": "is_shipping_billing_address_same", "type": "boolean", "input": "radio", "operator": "equal", "value": "1"}]}',
                  'score': -50,
                  'status': 'ACTIVE',
                  'isAutoGenerated': True,
                  'modifiedBy': 'argo@simility.com',
                  'modifiedOn': '2020-12-21T15:08:02'},
        'Rule5': {'createdBy': 'argo@simility.com',
                  'createdOn': '2020-12-16T15:00:17',
                  'appPrefix': 'james_testing',
                  'entityName': 'transaction',
                  'name': 'Rule5',
                  'conditions': '{"condition": "AND", "rules": [{"id": "ad_price_type", "field": "ad_price_type", "type": "string", "operator": "in", "value": "value.FREE,NEGOTIATION"}, {"id": "ad_price_type", "field": "ad_price_type", "type": "string", "operator": "not_in", "value": "value.FOO,BAR"}]}',
                  'score': -50,
                  'status': 'ACTIVE',
                  'isAutoGenerated': True,
                  'modifiedBy': 'argo@simility.com',
                  'modifiedOn': '2020-12-21T15:08:02'},
        'Rule6': {'createdBy': 'argo@simility.com',
                  'createdOn': '2020-12-16T15:00:17',
                  'appPrefix': 'james_testing',
                  'entityName': 'transaction',
                  'name': 'Rule6',
                  'conditions': '{"condition": "AND", "rules": [{"id": "ip_country_iso_code", "field": "ip_country_iso_code", "type": "string", "operator": "not_equal", "value": "field.billing_country"}, {"id": "country_id", "field": "country_id", "type": "string", "operator": "equal", "value": "field.ip_country_iso_code"}]}',
                  'score': -50,
                  'status': 'ACTIVE',
                  'isAutoGenerated': True,
                  'modifiedBy': 'argo@simility.com',
                  'modifiedOn': '2020-12-21T15:08:02'}
    }
    return expected_rule_configs


@pytest.fixture
def _expected_rule_configs_no_new_scores():
    expected_rule_configs = {
        'Rule1': {'createdBy': 'argo@simility.com',
                  'createdOn': '2020-12-16T15:00:17',
                  'appPrefix': 'james_testing',
                  'entityName': 'transaction',
                  'name': 'Rule1',
                  'conditions': '{"condition": "AND", "rules": [{"condition": "OR", "rules": [{"id": "payer_id.sum_approved_txn_amt_per_paypalid_1day", "field": "payer_id.sum_approved_txn_amt_per_paypalid_1day", "type": "double", "operator": "greater_or_equal", "value": "value.120.0"}, {"id": "payer_id.sum_approved_txn_amt_per_paypalid_7day", "field": "payer_id.sum_approved_txn_amt_per_paypalid_7day", "type": "double", "operator": "greater", "value": "value.240.0"}, {"id": "payer_id.sum_approved_txn_amt_per_paypalid_30day", "field": "payer_id.sum_approved_txn_amt_per_paypalid_30day", "type": "double", "operator": "less_or_equal", "value": "value.1000.0"}]}, {"id": "num_items", "field": "num_items", "type": "integer", "operator": "equal", "value": "value.2"}]}',
                  'score': -100,
                  'status': 'ACTIVE',
                  'isAutoGenerated': True,
                  'modifiedBy': 'argo@simility.com',
                  'modifiedOn': '2020-12-21T15:08:02'},
        'Rule2': {'createdBy': 'argo@simility.com',
                  'createdOn': '2020-12-16T15:00:17',
                  'appPrefix': 'james_testing',
                  'entityName': 'transaction',
                  'name': 'Rule2',
                  'conditions': '{"condition": "AND", "rules": [{"id": "ml_cc_v0", "field": "ml_cc_v0", "type": "double", "operator": "less", "value": "value.0.75"}, {"condition": "OR", "rules": [{"id": "method_clean", "field": "method_clean", "type": "string", "operator": "equal", "value": "value.login"}, {"id": "method_clean", "field": "method_clean", "type": "string", "operator": "begins_with", "value": "value.login"}, {"id": "method_clean", "field": "method_clean", "type": "string", "operator": "ends_with", "value": "value.login"}, {"id": "method_clean", "field": "method_clean", "type": "string", "operator": "contains", "value": "value.login"}, {"id": "ip_address", "field": "ip_address", "type": "string", "operator": "is_null", "value": null}, {"id": "ip_isp", "field": "ip_isp", "type": "string", "operator": "is_empty", "value": null}]}]}',
                  'score': -100,
                  'status': 'ACTIVE',
                  'isAutoGenerated': True,
                  'modifiedBy': 'argo@simility.com',
                  'modifiedOn': '2020-12-21T15:08:02'},
        'Rule3': {'createdBy': 'argo@simility.com',
                  'createdOn': '2020-12-16T15:00:17',
                  'appPrefix': 'james_testing',
                  'entityName': 'transaction',
                  'name': 'Rule3',
                  'conditions': '{"condition": "AND", "rules": [{"id": "method_clean", "field": "method_clean", "type": "string", "operator": "not_begins_with", "value": "value.login"}, {"id": "method_clean", "field": "method_clean", "type": "string", "operator": "not_ends_with", "value": "value.login"}, {"id": "method_clean", "field": "method_clean", "type": "string", "operator": "not_contains", "value": "value.login"}, {"condition": "OR", "rules": [{"id": "ip_address", "field": "ip_address", "type": "string", "operator": "is_not_null", "value": null}, {"id": "ip_isp", "field": "ip_isp", "type": "string", "operator": "is_not_empty", "value": null}]}]}',
                  'score': -100,
                  'status': 'ACTIVE',
                  'isAutoGenerated': True,
                  'modifiedBy': 'argo@simility.com',
                  'modifiedOn': '2020-12-21T15:08:02'},
        'Rule4': {'createdBy': 'argo@simility.com',
                  'createdOn': '2020-12-16T15:00:17',
                  'appPrefix': 'james_testing',
                  'entityName': 'transaction',
                  'name': 'Rule4',
                  'conditions': '{"condition": "AND", "rules": [{"id": "forwarder_address", "field": "forwarder_address", "type": "boolean", "input": "radio", "operator": "equal", "value": "0"}, {"id": "is_shipping_billing_address_same", "field": "is_shipping_billing_address_same", "type": "boolean", "input": "radio", "operator": "equal", "value": "1"}]}',
                  'score': -100,
                  'status': 'ACTIVE',
                  'isAutoGenerated': True,
                  'modifiedBy': 'argo@simility.com',
                  'modifiedOn': '2020-12-21T15:08:02'},
        'Rule5': {'createdBy': 'argo@simility.com',
                  'createdOn': '2020-12-16T15:00:17',
                  'appPrefix': 'james_testing',
                  'entityName': 'transaction',
                  'name': 'Rule5',
                  'conditions': '{"condition": "AND", "rules": [{"id": "ad_price_type", "field": "ad_price_type", "type": "string", "operator": "in", "value": "value.FREE,NEGOTIATION"}, {"id": "ad_price_type", "field": "ad_price_type", "type": "string", "operator": "not_in", "value": "value.FOO,BAR"}]}',
                  'score': -100,
                  'status': 'ACTIVE',
                  'isAutoGenerated': True,
                  'modifiedBy': 'argo@simility.com',
                  'modifiedOn': '2020-12-21T15:08:02'},
        'Rule6': {'createdBy': 'argo@simility.com',
                  'createdOn': '2020-12-16T15:00:17',
                  'appPrefix': 'james_testing',
                  'entityName': 'transaction',
                  'name': 'Rule6',
                  'conditions': '{"condition": "AND", "rules": [{"id": "ip_country_iso_code", "field": "ip_country_iso_code", "type": "string", "operator": "not_equal", "value": "field.billing_country"}, {"id": "country_id", "field": "country_id", "type": "string", "operator": "equal", "value": "field.ip_country_iso_code"}]}',
                  'score': -100,
                  'status': 'ACTIVE',
                  'isAutoGenerated': True,
                  'modifiedBy': 'argo@simility.com',
                  'modifiedOn': '2020-12-21T15:08:02'}
    }
    return expected_rule_configs


@pytest.fixture
def _expected_rule_configs_no_new_conds():
    expected_rule_configs = {'Rule1': {'createdBy': 'argo@simility.com',
                                       'createdOn': '2020-12-16T15:00:17',
                                       'appPrefix': 'james_testing',
                                       'entityName': 'transaction',
                                       'name': 'Rule1',
                                       'conditions': '{"condition": "AND", "rules": [{"condition": "OR", "rules": [{"id": "payer_id.sum_approved_txn_amt_per_paypalid_1day", "field": "payer_id.sum_approved_txn_amt_per_paypalid_1day", "type": "double", "operator": "greater_or_equal", "value": "value.60.0"}, {"id": "payer_id.sum_approved_txn_amt_per_paypalid_7day", "field": "payer_id.sum_approved_txn_amt_per_paypalid_7day", "type": "double", "operator": "greater", "value": "value.120.0"}, {"id": "payer_id.sum_approved_txn_amt_per_paypalid_30day", "field": "payer_id.sum_approved_txn_amt_per_paypalid_30day", "type": "double", "operator": "less_or_equal", "value": "value.500.0"}]}, {"id": "num_items", "field": "num_items", "type": "integer", "operator": "equal", "value": "value.1"}]}',
                                       'score': -50,
                                       'status': 'ACTIVE',
                                       'isAutoGenerated': True,
                                       'modifiedBy': 'argo@simility.com',
                                       'modifiedOn': '2020-12-21T15:08:02', },
                             'Rule2': {'createdBy': 'argo@simility.com',
                                       'createdOn': '2020-12-16T15:00:17',
                                       'appPrefix': 'james_testing',
                                       'entityName': 'transaction',
                                       'name': 'Rule2',
                                       'conditions': '{"condition": "AND", "rules": [{"id": "ml_cc_v0", "field": "ml_cc_v0", "type": "double", "operator": "less", "value": "value.0.315"}, {"condition": "OR", "rules": [{"id": "method_clean", "field": "method_clean", "type": "string", "operator": "equal", "value": "value.checkout"}, {"id": "method_clean", "field": "method_clean", "type": "string", "operator": "begins_with", "value": "value.checkout"}, {"id": "method_clean", "field": "method_clean", "type": "string", "operator": "ends_with", "value": "value.checkout"}, {"id": "method_clean", "field": "method_clean", "type": "string", "operator": "contains", "value": "value.checkout"}, {"id": "ip_address", "field": "ip_address", "type": "string", "operator": "is_not_null", "value": null}, {"id": "ip_isp", "field": "ip_isp", "type": "string", "operator": "is_not_empty", "value": null}]}]}',
                                       'score': -50,
                                       'status': 'ACTIVE',
                                       'isAutoGenerated': True,
                                       'modifiedBy': 'argo@simility.com',
                                       'modifiedOn': '2020-12-21T15:08:02', },
                             'Rule3': {'createdBy': 'argo@simility.com',
                                       'createdOn': '2020-12-16T15:00:17',
                                       'appPrefix': 'james_testing',
                                       'entityName': 'transaction',
                                       'name': 'Rule3',
                                       'conditions': '{"condition": "AND", "rules": [{"id": "method_clean", "field": "method_clean", "type": "string", "operator": "not_begins_with", "value": "value.checkout"}, {"id": "method_clean", "field": "method_clean", "type": "string", "operator": "not_ends_with", "value": "value.checkout"}, {"id": "method_clean", "field": "method_clean", "type": "string", "operator": "not_contains", "value": "value.checkout"}, {"condition": "OR", "rules": [{"id": "ip_address", "field": "ip_address", "type": "string", "operator": "is_null", "value": null}, {"id": "ip_isp", "field": "ip_isp", "type": "string", "operator": "is_empty", "value": null}]}]}',
                                       'score': -50,
                                       'status': 'ACTIVE',
                                       'isAutoGenerated': True,
                                       'modifiedBy': 'argo@simility.com',
                                       'modifiedOn': '2020-12-21T15:08:02', },
                             'Rule4': {'createdBy': 'argo@simility.com',
                                       'createdOn': '2020-12-16T15:00:17',
                                       'appPrefix': 'james_testing',
                                       'entityName': 'transaction',
                                       'name': 'Rule4',
                                       'conditions': '{"condition": "AND", "rules": [{"id": "forwarder_address", "field": "forwarder_address", "type": "boolean", "input": "radio", "operator": "equal", "value": "1"}, {"id": "is_shipping_billing_address_same", "field": "is_shipping_billing_address_same", "type": "boolean", "input": "radio", "operator": "equal", "value": "0"}]}',
                                       'score': -50,
                                       'status': 'ACTIVE',
                                       'isAutoGenerated': True,
                                       'modifiedBy': 'argo@simility.com',
                                       'modifiedOn': '2020-12-21T15:08:02', },
                             'Rule5': {'createdBy': 'argo@simility.com',
                                       'createdOn': '2020-12-16T15:00:17',
                                       'appPrefix': 'james_testing',
                                       'entityName': 'transaction',
                                       'name': 'Rule5',
                                       'conditions': '{"condition": "AND", "rules": [{"id": "ad_price_type", "field": "ad_price_type", "type": "string", "operator": "not_in", "value": "value.FREE,NEGOTIATION"}, {"id": "ad_price_type", "field": "ad_price_type", "type": "string", "operator": "in", "value": "value.FOO,BAR"}]}',
                                       'score': -50,
                                       'status': 'ACTIVE',
                                       'isAutoGenerated': True,
                                       'modifiedBy': 'argo@simility.com',
                                       'modifiedOn': '2020-12-21T15:08:02', },
                             'Rule6': {'createdBy': 'argo@simility.com',
                                       'createdOn': '2020-12-16T15:00:17',
                                       'appPrefix': 'james_testing',
                                       'entityName': 'transaction',
                                       'name': 'Rule6',
                                       'conditions': '{"condition": "AND", "rules": [{"id": "ip_country_iso_code", "field": "ip_country_iso_code", "type": "string", "operator": "equal", "value": "field.billing_country"}, {"id": "country_id", "field": "country_id", "type": "string", "operator": "not_equal", "value": "field.ip_country_iso_code"}]}',
                                       'score': -50,
                                       'status': 'ACTIVE',
                                       'isAutoGenerated': True,
                                       'modifiedBy': 'argo@simility.com',
                                       'modifiedOn': '2020-12-21T15:08:02', }}
    return expected_rule_configs


def test_update(_rule_configs, _new_conditions, _new_scores, _expected_rule_configs):
    rule_configs = _rule_configs
    new_conditions = _new_conditions
    new_scores = _new_scores
    expected_rule_configs = _expected_rule_configs
    up = UpdateExistingConfigs(rule_configs=rule_configs,
                               updated_conditions=new_conditions, updated_scores=new_scores)
    rule_configs = up.update()
    # Manually change modifiedOn date to match expected
    for rule_config in rule_configs.values():
        rule_config['modifiedOn'] = '2020-12-21T15:08:02'
    assert rule_configs == expected_rule_configs


def test_update_no_scores(_rule_configs, _new_conditions,
                          _expected_rule_configs_no_new_scores):
    rule_configs = _rule_configs
    new_conditions = _new_conditions
    expected_rule_configs = _expected_rule_configs_no_new_scores
    up = UpdateExistingConfigs(rule_configs=rule_configs,
                               updated_conditions=new_conditions)
    rule_configs = up.update()
    # Manually change modifiedOn date to match expected
    for rule_config in rule_configs.values():
        rule_config['modifiedOn'] = '2020-12-21T15:08:02'
    assert rule_configs == expected_rule_configs


def test_update_no_conds(_rule_configs, _new_scores,
                         _expected_rule_configs_no_new_conds):
    rule_configs = _rule_configs
    new_scores = _new_scores
    expected_rule_configs = _expected_rule_configs_no_new_conds
    up = UpdateExistingConfigs(rule_configs=rule_configs,
                               updated_scores=new_scores)
    rule_configs = up.update()
    # Manually change modifiedOn date to match expected
    for rule_config in rule_configs.values():
        rule_config['modifiedOn'] = '2020-12-21T15:08:02'
    assert rule_configs == expected_rule_configs
