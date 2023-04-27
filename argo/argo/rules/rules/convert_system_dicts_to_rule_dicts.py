"""
Class for converting between system-ready and dict representations of 
rules
"""

import _pickle as cPickle
import warnings


class ConvertSystemDictsToRuleDicts:
    """
    Converts a set of rules (each being represented in the system-ready format) 
    into the standard ARGO dictionary format.   

    Attributes:
        rule_dicts (dict): Set of rules defined using the standard ARGO 
            dictionary format (values) and their names (keys).
        unparsed_rules (list): List of rules which could not be parsed to an 
            ARGO-ready format (due to unsupported operators).
    """

    def __init__(self, system_dicts: dict):
        """
        Args:
            system_dicts (dict): Set of rules defined using the system JSON 
                format (values) and their system name (keys).
        """

        self.system_dicts = cPickle.loads(cPickle.dumps(system_dicts, -1))
        self.rule_dicts = {}
        self.unparsed_rules = []
        self.operators_supported = {
            'any_element': False,
            'begins_with': True,
            'black_listed': False,
            'contains': True,
            'ends_with': True,
            'equal': True,
            'greater': True,
            'greater_or_equal': True,
            'in': True,
            'is_after_by': False,
            'is_before_by': False,
            'is_empty': True,
            'is_not_empty': True,
            'is_not_null': True,
            'is_null': True,
            'java_el': False,
            'less': True,
            'less_or_equal': True,
            'no_element': False,
            'not_begins_with': True,
            'not_ends_with': True,
            'not_contains': True,
            'not_equal': True,
            'not_in': True,
            'regex': False,
            'white_listed': False
        }
        self.numeric_types = [
            'integer',
            'double',
            'float'
        ]

    def convert(self) -> dict:
        """
        Converts a set of rules (each being represented in the system-ready 
        format) into the standard ARGO dictionary format.   

        Returns:
            dict: Set of rules defined using the standard ARGO dictionary 
                format (values) and their names (keys).
        """
        for rule_name, system_dict in self.system_dicts.items():
            try:
                rule_dict = self._convert_rule(system_dict=system_dict)
            except Exception as e:
                warnings.warn(f'{rule_name} : {e}')
                self.unparsed_rules.append(rule_name)
                continue
            self.rule_dicts[rule_name] = rule_dict
        return self.rule_dicts

    def _convert_rule(self, system_dict: dict) -> dict:
        """
        Converts a rule stored in the system-ready format into the standard 
        ARGO dictionary format.

        Returns:
            dict: Rule defined using the standard ARGO dictionary format.
        """

        rule_dict = self._recurse_parse_conditions_dict(
            system_dict)
        rule_dict = self._remove_unnecessary_grouping(rule_dict)
        return rule_dict

    def _recurse_parse_conditions_dict(self, conditions_dict: dict) -> dict:
        """Recursively parses the system rule JSON config"""

        for rule in conditions_dict['rules']:
            rule.pop('data', None)
            rule_keys = list(rule.keys())
            rule_keys.sort()
            if rule_keys == ['condition', 'rules']:
                self._recurse_parse_conditions_dict(rule)
            else:
                self._parse_individual_condition(rule)
        return conditions_dict

    def _parse_individual_condition(self, condition: dict) -> None:
        """
        Parses the final level of rule condition 'branch' into the ARGO rule 
        dictionary format
        """

        original_operator = condition['operator']
        original_value = condition['value']
        self._parse_field_name(condition)
        self._parse_operator(condition, original_operator, original_value)
        self._parse_value(condition, original_operator)
        self._remove_unnecessary_fields(condition)

    def _parse_operator(self, condition: dict, original_operator: str,
                        original_value: object) -> None:
        """
        Parses the operator of the system rule and injects it into the ARGO 
        rule dictionary.
        """

        is_operator_supported = self.operators_supported[original_operator]
        if not is_operator_supported:
            raise Exception(
                f'Operator `{original_operator}` is not currently supported in ARGO. Rule cannot be parsed.')
        if isinstance(original_value, str) and original_value.startswith('field.'):
            parsed_operator = f'{original_operator}_field'
        else:
            parsed_operator = original_operator
        condition['operator'] = parsed_operator

    def _parse_value(self, condition: dict, original_operator: str) -> None:
        """
        Parses the value of the system rule and injects it into the ARGO rule 
        dictionary.
        """

        original_value = condition['value']
        field_type = condition['type']
        if original_value is None:
            parsed_value = None
        elif original_value.startswith('field.'):
            parsed_value = original_value[6:].replace(".", "_")
        else:
            parsed_value = original_value.replace('value.', '')
            if field_type in self.numeric_types:
                parsed_value = float(parsed_value)
            elif field_type == 'boolean':
                parsed_value = bool(int(parsed_value))
            elif original_operator in ['in', 'not_in']:
                parsed_value = parsed_value.split(',')
        condition['value'] = parsed_value

    @staticmethod
    def _parse_field_name(condition: dict) -> None:
        """
        Parses the field name of the system rule and injects it into the ARGO 
        rule dictionary.
        """

        parsed_field = condition['field'].replace('.', '_')
        condition['field'] = parsed_field

    @staticmethod
    def _remove_unnecessary_fields(condition: dict) -> None:
        """Removes unnecessary fields from the system rule JSON config"""

        condition.pop('id')
        condition.pop('type')
        condition.pop('data', None)
        condition.pop('input', None)

    @staticmethod
    def _remove_unnecessary_grouping(rule_dict: dict) -> dict:
        """
        Removes unnecessary parent grouping from rule (e.g. if complete rule is 
        wrapped in an AND condition).
        """

        if len(rule_dict['rules']) == 1:
            rule_keys = list(rule_dict['rules'][0].keys())
            rule_keys.sort()
            if rule_keys == ['condition', 'rules']:
                rule_dict = rule_dict['rules'][0]
        return rule_dict
