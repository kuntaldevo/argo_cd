"""
Class for converting between dict and system-ready representations of 
rules
"""
import _pickle as cPickle
from typing import Union


class ConvertRuleDictsToSystemDicts:
    """
    Converts a set of rules (each being represented in the standard ARGO 
    dictionary format) into the system-ready format.   

    Attributes:
        system_dicts (dict): Set of rules defined using the system JSON format 
            (values) and their system name (keys).
    """

    def __init__(self, rule_dicts: dict, field_datatypes: dict,
                 cassandra_field_names: dict):
        """
        Args:
            rule_dicts (dict): Set of rules defined using the standard ARGO 
                dictionary format (values) and their names (keys).
            field_datatypes (dict): The Cassandra datatypes (values) for each 
                pipeline output field (keys).
            cassandra_field_names (dict): The Cassandra field names (values) 
                for each pipeline output field (keys).
        """

        self.rule_dicts = cPickle.loads(cPickle.dumps(rule_dicts, -1))
        self.field_datatypes = field_datatypes
        self.cassandra_field_names = cassandra_field_names
        self.system_dicts = {}
        self._datatype_lookup = {
            'FLOAT': 'float',
            'BIGINT': 'integer',
            'TEXT': 'string',
            'MAP': None,
            'INT': 'integer',
            'DOUBLE': 'double',
            'SET': None,
            'BOOLEAN': 'boolean',
            'TIMESTAMP': None,
            'BLOB': None,
            'LIST': None
        }

    def convert(self) -> dict:
        """
        Converts a set of rules (each being represented in the standard ARGO 
        dictionary format) into the system-ready format.

        Returns:
            dict: Set of rules defined using the system-ready format 
                (values) and their names (keys).
        """

        for rule_name, rule_dict in self.rule_dicts.items():
            system_dict = self._convert_rule(rule_dict=rule_dict)
            self.system_dicts[rule_name] = system_dict
        return self.system_dicts

    def _convert_rule(self, rule_dict: dict) -> dict:
        """
        Converts a rule stored in the standard ARGO dictionary format into the 
        system-ready format.
        """

        system_dict = self._recurse_parse_conditions_dict(rule_dict)
        return system_dict

    def _recurse_parse_conditions_dict(self, conditions_dict: dict) -> dict:
        """Recursively parses the ARGO rule dictionary"""

        for rule in conditions_dict['rules']:
            rule_keys = list(rule.keys())
            rule_keys.sort()
            if rule_keys == ['condition', 'rules']:
                self._recurse_parse_conditions_dict(rule)
            else:
                self._parse_individual_condition(rule)
        return conditions_dict

    def _parse_individual_condition(self, rule: dict) -> None:
        """
        Parses the individual condition of a rule into the system 
        format.
        """
        field = rule['field']
        operator = rule['operator']
        value = rule['value']
        new_field = self.cassandra_field_names[field]
        field_datatype = self.field_datatypes[field]
        new_datatype = self._datatype_lookup[field_datatype]
        rule['id'] = new_field
        rule['field'] = new_field
        rule['type'] = new_datatype
        if new_datatype == 'boolean':
            rule['input'] = 'radio'
            rule['operator'] = 'equal' if operator in [
                'equal', 'not_equal'] else operator
        if operator.endswith('_field'):
            rule['operator'] = operator[:-6]
        new_value = self._return_new_value(operator=operator,
                                           datatype=new_datatype, value=value)
        rule['value'] = new_value

    def _return_new_value(self, operator: str, datatype: str,
                          value: Union[str, int, float]) -> Union[str, int, float]:
        """Returns the value for the system-ready condition"""

        if value is None:
            new_value = None
        elif operator in ['in', 'not_in']:
            new_value = f'value.{",".join(value)}'
        elif operator.endswith('_field'):
            new_value = f'field.{self.cassandra_field_names[value]}'
        elif datatype == 'boolean':
            new_value = self._parse_boolean_value(operator, value)
        elif datatype == 'integer':
            new_value = f'value.{int(value)}'
        elif value is not None:
            new_value = f'value.{value}'
        else:
            new_value = value
        return new_value

    @staticmethod
    def _parse_boolean_value(operator: str,
                             value: Union[str, int, float]) -> str:
        """Parses the value for boolean operators"""

        python_op_lookup = {
            'equal': '==',
            'not_equal': '!='
        }
        try:
            bool_value = bool(float(value))
        except:
            bool_value = True if value == 'True' else False
        bool_eval = eval(f'True{python_op_lookup[operator]}{bool_value}')
        return '1' if bool_eval else '0'
