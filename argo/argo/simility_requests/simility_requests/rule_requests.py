"""
Classes for creating, updating and returning rule related information
from Simility
"""
import requests
import json
import pandas as pd
import warnings
from simility_apis.create_rule_in_simility import CreateRuleInSimilityAPI
from simility_apis.update_rule_in_simility import UpdateRuleInSimilityAPI
from simility_apis.return_rule_info_from_simility import ReturnRuleInfoFromSimilityAPI
import simility_apis.set_password


class CreateRulesInSimility:

    """
    Creates rules in a Simility instance, given a dictionary of 
    system-ready rule configs stored as Python dictionaries (values)
    and the rule name (keys).
    """

    def __init__(self, url: str, app_prefix: str, user: str):
        """
        Args:
            url (str): The URL of the desired cluster (e.g. 
                'https://app.simility.com' for US-SaaS).
            app_prefix (str): Keyspace name on the cluster.
            user (str): Username to access the cluster.            
        """

        if url.endswith('/'):
            self.url = url[:-1]
        else:
            self.url = url
        self.app_prefix = app_prefix
        self.user = user

    def request(self, rule_configs: dict) -> None:
        """
        Creates rules in a Simility instance, given a dictionary of 
        system-ready rule configs stored as Python dictionaries (values)
        and the rule name (keys).

        Args:
            rule_configs (dict): Contains the system-ready rule configs 
                stored as Python dictionaries (values) and the rule 
                name (keys).

        Raises:
            Exception: If the API request returns an error, an exception is 
                thrown and the request error is shown.
        """
        cr = CreateRuleInSimilityAPI(
            url=self.url, app_prefix=self.app_prefix, user=self.user)
        for rule_name, rule_config in rule_configs.items():
            try:
                rule_config_str = json.dumps(rule_config)
                cr.request(rule_config=rule_config_str)
            except Exception as e:
                warnings.warn(f'Rule `{rule_name}` - {e}')


class UpdateRulesInSimility:

    """
    Updates existing rules in Simility, given a dictionary of 
    system-ready rule configs stored as Python dictionaries (values)
    and the rule name (keys).
    """

    def __init__(self, url: str, app_prefix: str, user: str):
        """
        Args:
            url (str): The URL of the desired cluster (e.g. 
                'https://app.simility.com' for US-SaaS).
            app_prefix (str): Keyspace name on the cluster.
            user (str): Username to access the cluster.            
        """

        if url.endswith('/'):
            self.url = url[:-1]
        else:
            self.url = url
        self.app_prefix = app_prefix
        self.user = user

    def request(self, rule_configs: dict) -> None:
        """
        Updates existing rules in Simility, given a dictionary of 
        system-ready rule configs stored as Python dictionaries (values)
        and the rule name (keys).

        Args:
            rule_configs (dict): Contains the system-ready rule configs 
            stored as Python dictionaries (values) and the rule 
            name (keys).

        Raises:
            Exception: If the API request returns an error, an exception is
            thrown and the request error is shown.
        """

        ur = UpdateRuleInSimilityAPI(
            url=self.url, app_prefix=self.app_prefix, user=self.user)
        for rule_name, rule_config in rule_configs.items():
            try:
                rule_config_str = json.dumps(rule_config)
                ur.request(rule_config=rule_config_str)
            except Exception as e:
                warnings.warn(f'Rule `{rule_name}` - {e}')


class ReturnRuleInfoFromSimility:
    """
    Returns the current rules and the information associated with them from a 
    Simility instance.
    """

    def __init__(self, url: str, app_prefix: str, entity: str, user: str,
                 keep_active_only=True):
        """
        Args:
            url (str): The URL of the desired cluster (e.g. 
                'https://app.simility.com' for US-SaaS).
            app_prefix (str): Keyspace name on the cluster.
            entity (str): The entity which the rules are built on.
            user (str): Username to access the cluster.
            keep_active_only (bool): Determines whether only active rules are 
                returned (set to `True` to return active rules only). Defaults
                to `True`.
        """

        if url.endswith('/'):
            self.url = url[:-1]
        else:
            self.url = url
        self.app_prefix = app_prefix
        self.entity = entity
        self.user = user
        self.keep_active_only = keep_active_only

    def request(self) -> pd.DataFrame:
        """
        Returns the current rules and the information associated with them from
        a Simility instance.

        Returns:
            pd.DataFrame: The rule configuration JSONs for each rule in a 
                Simility environment, formatted into a dataframe.

        Raises:
            Exception: If API request throws an error.
            Exception: If system rule config returned from the system is empty 
                (suggesting there are no rules present in the entity chosen)
        """

        response = self._request_for_rules(
            url=self.url, app_prefix=self.app_prefix, entity=self.entity,
            user=self.user)
        rules_df = self._convert_response_to_dataframe(
            response=response.text, keep_active_only=self.keep_active_only)
        return rules_df

    @staticmethod
    def _request_for_rules(url: str, app_prefix: str, entity: str,
                           user: str) -> str:
        """
        Returns the raw rule configuration JSONs related to the rules found 
        in a Simility environment.
        """

        rr = ReturnRuleInfoFromSimilityAPI(
            url=url, app_prefix=app_prefix, entity=entity, user=user)
        response = rr.request()
        return response

    @staticmethod
    def _convert_response_to_dataframe(response: str,
                                       keep_active_only: bool) -> pd.DataFrame:
        """Converts the raw string response from the API into a dataframe"""

        rules_json = json.loads(response)
        raw_rule_configs = pd.read_json(response)
        raw_rule_configs.rename(
            {'entries': 'jsonConfig'}, axis=1, inplace=True)
        rules_df = pd.DataFrame(rules_json['entries'])
        rules_df = pd.concat([rules_df, raw_rule_configs], axis=1)
        if rules_df.shape == (0, 0):
            raise Exception(
                'ERROR: the system rules config is empty, which suggests that there are no rules present on the chosen entity.')
        if keep_active_only:
            rules_df = rules_df[rules_df['status'] == 'ACTIVE']
        rules_df['modifiedOn'] = rules_df['modifiedOn'].apply(
            lambda x: pd.to_datetime(x, utc=True))
        rules_df.set_index('name', inplace=True)
        return rules_df


class ReturnRuleConditionsFromSimility:

    """
    Returns the current rules and their conditions from a Simility instance.
    """

    def __init__(self, url: str, app_prefix: str, entity: str, user: str,
                 keep_active_only=True):
        """
        Args:
            url (str): The URL of the desired cluster (e.g. 
                'https://app.simility.com' for US-SaaS).
            app_prefix (str): Keyspace name on the cluster.
            entity (str): The entity which the rules are built on.
            user (str): Username to access the cluster.
            keep_active_only (bool): Determines whether only active rules are 
                returned (set to `True` to return active rules only). Defaults 
                to `True`.
        """

        if url.endswith('/'):
            self.url = url[:-1]
        else:
            self.url = url
        self.app_prefix = app_prefix
        self.entity = entity
        self.user = user
        self.keep_active_only = keep_active_only

    def request(self) -> dict:
        """
        Returns the current rules and their conditions from a Simility 
        instance.

        Returns:
            dict: The system rule names (keys) and their conditions (values).
        """

        rri = ReturnRuleInfoFromSimility(url=self.url,
                                         app_prefix=self.app_prefix,
                                         user=self.user,
                                         entity=self.entity,
                                         keep_active_only=self.keep_active_only)
        rule_info = rri.request()
        rule_json_conditions = rule_info['jsonConfig'].to_dict()
        rule_conditions = dict((rule_name, json.loads(config['conditions']))
                               for rule_name, config in rule_json_conditions.items())
        return rule_conditions


class ReturnRuleConfigsFromSimility:
    """
    Returns the current rules and their system configs from a Simility 
    instance.
    """

    def __init__(self, url: str, app_prefix: str, entity: str, user: str, rules=None):
        """
        Args:
            url (str): The URL of the desired cluster (e.g. 
                'https://app.simility.com' for US-SaaS).
            app_prefix (str): Keyspace name on the cluster.
            entity (str): The entity which the rules are built on.
            user (str): Username to access the cluster.
            rules (list): Names of rules to include in output. If `None`, the 
                system configs of all of the rules in the system will be 
                returned. Defaults to `None`.
        """

        if url.endswith('/'):
            self.url = url[:-1]
        else:
            self.url = url
        self.app_prefix = app_prefix
        self.entity = entity
        self.user = user
        self.rules = rules

    def request(self) -> dict:
        """
        Returns the current rules and their conditions from a Simility 
        instance.

        Returns:
            dict: The system rule names (keys) and their system configs 
                (values).
        """

        rri = ReturnRuleInfoFromSimilityAPI(url=self.url,
                                            app_prefix=self.app_prefix,
                                            user=self.user,
                                            entity=self.entity)
        response_text = rri.request().text
        rule_json_configs = json.loads(response_text)['entries']
        if self.rules is None:
            rule_configs = dict(
                (rule_json_config['name'], rule_json_config) for rule_json_config in rule_json_configs)
        else:
            rule_configs = dict(
                (rule_json_config['name'], rule_json_config) for rule_json_config in rule_json_configs if rule_json_config['name'] in self.rules)
        return rule_configs
