"""
Class for updating existing system-ready rule configs using a set of 
new conditions
"""
import copy
import json
from datetime import datetime


class UpdateExistingConfigs:
    """
    Updates a set of rule configurations with new conditions, new scores, 
    or both. These can be used to update rules in the system using the 
    update_rules_in_simility module.

    Attributes:
        updated_rule_configs (dict): Dictionary of system-ready rule 
            configurations (values) and the rule name (keys).
    """

    def __init__(self, rule_configs: dict, updated_conditions=None,
                 updated_scores=None, make_inactive=False,
                 modified_by='argo@simility.com'):
        """    
        Args:
            rule_configs (dict): The original system-ready rule configurations.
            updated_conditions (dict, optional): The new rule conditions to 
                update in the original system-ready rule configurations. 
                Defaults to `None`.
            updated_scores (dict, optional): The new rule scores to update
                in the original system-ready rule configurations. Defaults to 
                `None`.
            make_inactive (bool, optional): If set to `True`, the `status` 
                field in the rule configurations will be set to 'INACTIVE', 
                meaning that the rule will be inactivated in the system if the 
                configuration is updated in Simility.
            modified_by (str, optional): Label in the configuration which shows 
                who last modified the rule. Defaults to 'argo@simility.com'.

        Raises:
            ValueError: `updated_conditions` or `updated_scores` must be 
                provided.
        """

        self.rule_configs = copy.deepcopy(rule_configs)
        self.updated_conditions = updated_conditions
        self.updated_scores = updated_scores
        self.make_inactive = make_inactive
        self.modified_by = modified_by
        now = datetime.now()
        self.modified_on = now.strftime("%Y-%m-%dT%H:%M:%S")
        self.updated_rule_configs = {}

    def update(self) -> dict:
        """
        Updates a set of rule configurations with new conditions, new scores, 
        or both. These can be used to update rules in the system using the 
        `update_rules_in_simility` module.

        Returns:
            dict: Dictionary of system-ready rule JSON configurations 
                (in Python dictionary format).
        """

        for rule_name in self.rule_configs.keys():
            rule_config = self._update_config(rule_name=rule_name)
            self.updated_rule_configs[rule_name] = rule_config
        return self.updated_rule_configs

    def _update_config(self, rule_name: str) -> dict:
        """
        Updates the existing rule config with the new conditions or scores.
        """

        rule_config = self.rule_configs[rule_name]
        rule_config.pop('conditionsJson', None)
        rule_config.pop('conditionsV2', None)
        rule_config.pop('conditionsV2Json', None)
        rule_config.pop('conditionsCleaned', None)
        if self.updated_conditions is not None:
            updated_condition = self.updated_conditions[rule_name]
            updated_condition_str = json.dumps(updated_condition)
            rule_config['conditions'] = updated_condition_str
        if self.updated_scores is not None:
            updated_score = self.updated_scores[rule_name]
            rule_config['score'] = int(updated_score)
        if self.make_inactive:
            rule_config['status'] = 'INACTIVE'
        rule_config['modifiedBy'] = self.modified_by
        rule_config['modifiedOn'] = self.modified_on
        return rule_config
