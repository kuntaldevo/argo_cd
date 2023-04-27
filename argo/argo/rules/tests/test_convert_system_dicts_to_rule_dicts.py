import pytest
from rules.convert_system_dicts_to_rule_dicts import ConvertSystemDictsToRuleDicts


@pytest.fixture
def _system_dicts():
    system_dicts = {'Rule1': {'condition': 'AND',
                              'rules': [{'condition': 'OR',
                                         'rules': [{'id': 'payer_id.sum_approved_txn_amt_per_paypalid_1day',
                                                    'field': 'payer_id.sum_approved_txn_amt_per_paypalid_1day',
                                                    'type': 'double',
                                                    'operator': 'greater_or_equal',
                                                    'value': 'value.60'},
                                                   {'id': 'payer_id.sum_approved_txn_amt_per_paypalid_7day',
                                                    'field': 'payer_id.sum_approved_txn_amt_per_paypalid_7day',
                                                    'type': 'double',
                                                    'operator': 'greater',
                                                    'value': 'value.120'},
                                                   {'id': 'payer_id.sum_approved_txn_amt_per_paypalid_30day',
                                                    'field': 'payer_id.sum_approved_txn_amt_per_paypalid_30day',
                                                    'type': 'double',
                                                    'operator': 'less_or_equal',
                                                    'value': 'value.500'}],
                                         'data': {}},
                                        {'id': 'num_items',
                                         'field': 'num_items',
                                         'type': 'integer',
                                         'operator': 'equal',
                                         'value': 'value.1'},
                                        ]},
                    'Rule2': {'condition': 'OR',
                              'rules': [{'condition': 'AND',
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
                                                              ]}]}]},
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
                                         'data': {}}]},
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
                                         'data': {}},
                                        {'id': 'country_id',
                                         'field': 'country_id',
                                         'type': 'string',
                                         'operator': 'not_equal',
                                         'value': 'field.ip_country_iso_code',
                                         'data': {}}]}}
    return system_dicts


@pytest.fixture
def _error_system_dicts():
    error_system_dicts = {'Rule7': {'condition': 'OR',
                                    'rules': [{'id': 'java_el',
                                               'field': 'java_el',
                                               'type': 'string',
                                               'input': 'text',
                                               'operator': 'java_el',
                                               'value': '${table_values.$shipping_address_1.toLowerCase().matches(".*(24731 plumtree ct|36508 wells rd\\\\b.*")}'}]}}
    return error_system_dicts


@pytest.fixture
def _expected_rule_dicts():
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
def _expected_rule_dict_with_unnecessary_grouping():
    rule_dict = {'condition': 'OR',
                 'rules': [{'condition': 'AND',
                            'rules': [{'field': 'ml_cc_v0', 'operator': 'less', 'value': 0.315},
                                      {'condition': 'OR',
                                       'rules': [{'field': 'method_clean',
                                                  'operator': 'equal',
                                                  'value': 'checkout'},
                                                 {'field': 'method_clean',
                                                  'operator': 'begins_with',
                                                  'value': 'checkout'},
                                                 {'field': 'method_clean',
                                                  'operator': 'ends_with', 'value': 'checkout'},
                                                 {'field': 'method_clean',
                                                  'operator': 'contains', 'value': 'checkout'},
                                                 {'field': 'ip_address',
                                                  'operator': 'is_not_null', 'value': None},
                                                 {'field': 'ip_isp', 'operator': 'is_not_empty', 'value': None}]}]}]}
    return rule_dict


@pytest.fixture
def _individual_system_conditions():
    system_conditions = [{'id': 'payer_id.sum_approved_txn_amt_per_paypalid_1day',
                          'field': 'payer_id.sum_approved_txn_amt_per_paypalid_1day',
                          'type': 'double',
                          'operator': 'greater_or_equal',
                          'value': 'value.60'},
                         {'id': 'payer_id.sum_approved_txn_amt_per_paypalid_7day',
                          'field': 'payer_id.sum_approved_txn_amt_per_paypalid_7day',
                          'type': 'double',
                          'operator': 'greater',
                          'value': 'value.120'},
                         {'id': 'payer_id.sum_approved_txn_amt_per_paypalid_30day',
                          'field': 'payer_id.sum_approved_txn_amt_per_paypalid_30day',
                          'type': 'double',
                          'operator': 'less_or_equal',
                          'value': 'value.500'},
                         {'id': 'num_items',
                          'field': 'num_items',
                          'type': 'integer',
                          'operator': 'equal',
                          'value': 'value.1'},
                         {'id': 'ml_cc_v0',
                          'field': 'ml_cc_v0',
                          'type': 'double',
                          'operator': 'less',
                          'value': 'value.0.315'},
                         {'id': 'method_clean',
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
                          'value': None},
                         {'id': 'method_clean',
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
                         {'id': 'ip_address',
                          'field': 'ip_address',
                          'type': 'string',
                          'operator': 'is_null',
                          'value': None},
                         {'id': 'ip_isp',
                          'field': 'ip_isp',
                          'type': 'string',
                          'operator': 'is_empty',
                          'value': None},
                         {'id': 'forwarder_address',
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
                          'value': '0'},
                         {'id': 'ad_price_type',
                          'field': 'ad_price_type',
                          'type': 'string',
                          'operator': 'not_in',
                          'value': 'value.FREE,NEGOTIATION'},
                         {'id': 'ad_price_type',
                          'field': 'ad_price_type',
                          'type': 'string',
                          'operator': 'in',
                          'value': 'value.FOO,BAR'},
                         {'id': 'ip_country_iso_code',
                          'field': 'ip_country_iso_code',
                          'type': 'string',
                          'operator': 'equal',
                          'value': 'field.billing_country',
                          'data': {}},
                         {'id': 'country_id',
                          'field': 'country_id',
                          'type': 'string',
                          'operator': 'not_equal',
                          'value': 'field.ip_country_iso_code',
                          'data': {}}]
    return system_conditions


@pytest.fixture
def _parsed_individual_conditions():
    parsed_conditions = [{'field': 'payer_id_sum_approved_txn_amt_per_paypalid_1day',
                          'operator': 'greater_or_equal',
                          'value': 60.0},
                         {'field': 'payer_id_sum_approved_txn_amt_per_paypalid_7day',
                          'operator': 'greater',
                          'value': 120.0},
                         {'field': 'payer_id_sum_approved_txn_amt_per_paypalid_30day',
                          'operator': 'less_or_equal',
                          'value': 500.0},
                         {'field': 'num_items',
                          'operator': 'equal',
                          'value': 1.0},
                         {'field': 'ml_cc_v0',
                          'operator': 'less',
                          'value': 0.315},
                         {'field': 'method_clean',
                          'operator': 'equal',
                          'value': 'checkout'},
                         {'field': 'method_clean',
                          'operator': 'begins_with',
                          'value': 'checkout'},
                         {'field': 'method_clean',
                          'operator': 'ends_with',
                          'value': 'checkout'},
                         {'field': 'method_clean',
                          'operator': 'contains',
                          'value': 'checkout'},
                         {'field': 'ip_address',
                          'operator': 'is_not_null',
                          'value': None},
                         {'field': 'ip_isp',
                          'operator': 'is_not_empty',
                          'value': None},
                         {'field': 'method_clean',
                          'operator': 'not_begins_with',
                          'value': 'checkout'},
                         {'field': 'method_clean',
                          'operator': 'not_ends_with',
                          'value': 'checkout'},
                         {'field': 'method_clean',
                          'operator': 'not_contains',
                          'value': 'checkout'},
                         {'field': 'ip_address',
                          'operator': 'is_null',
                          'value': None},
                         {'field': 'ip_isp',
                          'operator': 'is_empty',
                          'value': None},
                         {'field': 'forwarder_address',
                          'operator': 'equal',
                          'value': True},
                         {'field': 'is_shipping_billing_address_same',
                          'operator': 'equal',
                          'value': False},
                         {'field': 'ad_price_type',
                          'operator': 'not_in',
                          'value': ['FREE',
                                    'NEGOTIATION']},
                         {'field': 'ad_price_type',
                          'operator': 'in',
                          'value': ['FOO',
                                    'BAR']},
                         {'field': 'ip_country_iso_code',
                          'operator': 'equal_field',
                          'value': 'billing_country'},
                         {'field': 'country_id',
                          'operator': 'not_equal_field',
                          'value': 'ip_country_iso_code'}]
    return parsed_conditions


def test_convert(_system_dicts, _expected_rule_dicts):
    system_dicts = _system_dicts
    expected_rule_dicts = _expected_rule_dicts
    sp = ConvertSystemDictsToRuleDicts(system_dicts=system_dicts)
    rule_dicts = sp.convert()
    assert rule_dicts == expected_rule_dicts


def test_convert_rule(_system_dicts, _expected_rule_dicts):
    system_dicts = _system_dicts
    expected_rule_dicts = _expected_rule_dicts
    sp = ConvertSystemDictsToRuleDicts(system_dicts=system_dicts)
    for rule_name, system_dict in system_dicts.items():
        rule_dict = sp._convert_rule(system_dict=system_dict)
        assert rule_dict == expected_rule_dicts[rule_name]


def test_recurse_parse_conditions_dict(_system_dicts, _expected_rule_dict_with_unnecessary_grouping):
    system_dicts = _system_dicts
    system_dict = system_dicts['Rule2']
    expected_rule_dict = _expected_rule_dict_with_unnecessary_grouping
    sp = ConvertSystemDictsToRuleDicts(system_dicts={})
    rule_dict = sp._recurse_parse_conditions_dict(
        conditions_dict=system_dict)
    assert rule_dict == expected_rule_dict


def test_parse_individual_condition(_individual_system_conditions, _parsed_individual_conditions):
    system_conditions = _individual_system_conditions
    expected_individual_conditions = _parsed_individual_conditions
    parsed_conditions = []
    for system_condition in system_conditions:
        sp = ConvertSystemDictsToRuleDicts({})
        sp._parse_individual_condition(system_condition)
        parsed_conditions.append(system_condition)
    assert parsed_conditions == expected_individual_conditions


def test_parse_operator(_individual_system_conditions, _parsed_individual_conditions):
    system_conditions = _individual_system_conditions
    expected_individual_conditions = _parsed_individual_conditions
    for i, system_condition in enumerate(system_conditions):
        sp = ConvertSystemDictsToRuleDicts({})
        operator = system_condition['operator']
        value = system_condition['value']
        sp._parse_operator(condition=system_condition,
                           original_operator=operator, original_value=value)
        assert system_condition['operator'] == expected_individual_conditions[i]['operator']


def test_parse_operator_error(_error_system_dicts):
    system_condition = _error_system_dicts['Rule7']['rules'][0]
    sp = ConvertSystemDictsToRuleDicts({})
    operator = system_condition['operator']
    value = system_condition['value']
    with pytest.raises(Exception):
        sp._parse_operator(condition=system_condition,
                           original_operator=operator, original_value=value)


def test_parse_value(_individual_system_conditions, _parsed_individual_conditions):
    system_conditions = _individual_system_conditions
    expected_individual_conditions = _parsed_individual_conditions
    for i, system_condition in enumerate(system_conditions):
        sp = ConvertSystemDictsToRuleDicts({})
        operator = system_condition['operator']
        sp._parse_value(condition=system_condition,
                        original_operator=operator)
        assert system_condition['value'] == expected_individual_conditions[i]['value']


def test_parse_field_name(_individual_system_conditions, _parsed_individual_conditions):
    system_conditions = _individual_system_conditions
    expected_individual_conditions = _parsed_individual_conditions
    for i, system_condition in enumerate(system_conditions):
        sp = ConvertSystemDictsToRuleDicts({})
        sp._parse_field_name(condition=system_condition)
        assert system_condition['field'] == expected_individual_conditions[i]['field']


def test_remove_unnecessary_fields():
    dummy_dict = {
        'field': 'A',
        'operator': '>=',
        'value': 1,
        'id': 'A',
        'type': 'float',
        'data': {},
        'input': {}
    }
    expected_result = {
        'field': 'A',
        'operator': '>=',
        'value': 1
    }
    sp = ConvertSystemDictsToRuleDicts({})
    sp._remove_unnecessary_fields(dummy_dict)
    assert dummy_dict == expected_result


def test_remove_unnecessary_grouping():
    dummy_dict = {'condition': 'OR',
                  'rules': [{'condition': 'AND',
                               'rules': [
                                   {
                                       'field': 'A',
                                       'operator': '>=',
                                       'value': 1
                                   }]}]}
    expected_result = {'condition': 'AND',
                       'rules': [
                           {
                               'field': 'A',
                               'operator': '>=',
                               'value': 1
                           }]}
    sp = ConvertSystemDictsToRuleDicts({})
    dummy_dict = sp._remove_unnecessary_grouping(dummy_dict)
    assert dummy_dict == expected_result
