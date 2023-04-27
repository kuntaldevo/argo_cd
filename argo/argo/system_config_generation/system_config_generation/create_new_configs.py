"""
Class for creating new system-ready rule configs from a set of conditions
"""
from datetime import datetime
import json


class CreateNewConfigs:
    """
    Creates system-ready rule configurations for new rules. These can be used 
    to create the rules in the system using the create_rules_in_simility 
    module.

    Attributes:
        rule_configs (dict): Dictionary of system-ready rule JSON 
            configurations (values) and the rule name (keys).
    """

    def __init__(self, conditions: dict, scores: dict, app_prefix: str,
                 entity: str, make_active=True,
                 created_by='argo@simility.com'):
        """
        Args:
            conditions (dict): Set of rules defined using the system JSON 
                format (values) and their system name (keys).
            scores (dict): Set of scores (values) to be assigned to each rule 
                (keys).            
            app_prefix (str): App prefix of the Simility instance where the 
                rules will be created.
            entity (str): Entity of the Simility instance where the rules will 
                be created.
            make_active (bool, optional): If `True`, the configurations are set 
                such that, when they are sent to Simility, the rules are made 
                active. Defaults to `True`.
            created_by (str, optional): Label in the configuration which shows 
                who created the rule. Defaults to 'argo@simility.com'.
        """

        self.conditions = conditions
        self.scores = scores
        self.app_prefix = app_prefix
        self.entity = entity
        self.created_by = created_by
        now = datetime.now()
        self.created_on = now.strftime("%Y-%m-%dT%H:%M:%S")
        self.config_template = {
            'createdBy': self.created_by,
            'createdOn': self.created_on,
            'appPrefix': self.app_prefix,
            'entityName': self.entity,
            'name': None,
            'conditions': None,
            'score': None,
            'status': 'ACTIVE' if make_active else 'INACTIVE',
            'isAutoGenerated': True
        }
        self.rule_configs = {}

    def generate(self) -> dict:
        """
        Creates system-ready rule configurations for new rules. These can be 
        used to create the rules in the system using the 
        `create_rules_in_simility` module.

        Returns:
            dict: Dictionary of system-ready rule JSON configurations 
                (in Python dictionary format).
        """

        for rule_name in self.conditions.keys():
            rule_config = self._create_config(rule_name=rule_name)
            self.rule_configs[rule_name] = rule_config
        return self.rule_configs

    def _create_config(self, rule_name: str) -> dict:
        """Updates the config template with the necessary fields"""

        rule_config = self.config_template.copy()
        rule_config['name'] = rule_name
        rule_config['conditions'] = json.dumps(self.conditions[rule_name])
        rule_config['score'] = int(self.scores[rule_name])
        return rule_config