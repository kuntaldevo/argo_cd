"""Class for defining a rule set and changing between representations"""

from rules.convert_rule_dicts_to_rule_strings import ConvertRuleDictsToRuleStrings
from rules.convert_rule_strings_to_rule_dicts import ConvertRuleStringsToRuleDicts
from rules.convert_rule_dicts_to_rule_lambdas import ConvertRuleDictsToRuleLambdas
from rules.convert_system_dicts_to_rule_dicts import ConvertSystemDictsToRuleDicts
from rules.convert_rule_dicts_to_system_dicts import ConvertRuleDictsToSystemDicts
from rules.convert_rule_lambdas_to_rule_strings import ConvertRuleLambdasToRuleStrings
from rules.get_rule_attributes import GetRuleFeatures
import warnings


class Rules:
    """
    Defines a set of rules using the following representations: system-ready, 
    string, dictionary, lambda expression.

    One of the above formats must be provided to define the rule set. The 
    rules can then be reformatted into one of the other representations.

    Attributes:
        system_dicts (dict): Set of rules defined using the system JSON format 
            (values) and their system name (keys).
        rule_dicts (dict): Set of rules defined using the standard ARGO 
            dictionary format (values) and their names (keys).
        rule_strings (dict): Set of rules defined using the standard ARGO 
            string format (values) and their names (keys).
        rule_lambdas (dict): Set of rules defined using the standard ARGO 
            lambda expression format (values) and their names (keys).
        lambda_kwargs (dict): For each rule (keys), a dictionary containing the 
            features used in the rule (keys) and the current values (values).            
        lambda_args (dict): For each rule (keys), a list containing the current 
            values used in the rule.
        rule_features (dict): For each rule (keys), a list containing the 
            features used in the rule.
    """

    def __init__(self, system_dicts=None, rule_dicts=None, rule_strings=None,
                 rule_lambdas=None, lambda_kwargs=None, lambda_args=None):
        """
        Provide one of the following arguments to define a rule set:

        Args:
            system_dicts (dict, optional): Set of rules defined using the 
                system JSON format (values) and their system name (keys).
            rule_dicts (dict, optional): Set of rules defined using the 
                standard ARGO dictionary format (values) and their system names 
                (keys). Defaults to `None`.
            rule_strings (dict, optional): Set of rules defined using the 
                standard ARGO string format (values) and their system names 
                (keys). Defaults to `None`.
            rule_lambdas (dict): Set of rules defined using the standard ARGO 
                lambda expression format (values) and their names (keys). Must 
                be given in conjunction with either `lambda_kwargs` or 
                `lambda_args`. Defaults to `None`.
            lambda_kwargs (dict): For each rule (keys), a dictionary containing 
                the features used in the rule (keys) and the current values 
                (values). Only populates when `.as_lambda()` is used with the 
                keyword argument `with_kwargs=True`. Defaults to `None`.
            lambda_args (dict): For each rule (keys), a list containing the
                current values used in the rule. Only populates when 
                `.as_lambda()` is used with the keyword argument 
                `with_kwargs=False`. Defaults to `None`.
        """

        if system_dicts is None and rule_dicts is None and \
                rule_strings is None and rule_lambdas is None:
            raise ValueError(
                '`system_dicts`, `rule_dicts`, `rule_strings` or `rule_lambdas` must be given')
        if rule_lambdas is not None and lambda_kwargs is None and \
                lambda_args is None:
            raise ValueError(
                '`lambda_kwargs` or `lambda_args` must be given when `rule_lambdas` is provided')
        self.system_dicts = {} if system_dicts is None else system_dicts
        self.rule_dicts = {} if rule_dicts is None else rule_dicts
        self.rule_strings = {} if rule_strings is None else rule_strings
        self.rule_lambdas = {} if rule_lambdas is None else rule_lambdas
        self.lambda_kwargs = {} if lambda_kwargs is None else lambda_kwargs
        self.lambda_args = {} if lambda_args is None else lambda_args
        self.rule_features = {}

    def as_rule_dicts(self) -> dict:
        """
        Converts rules into the standard ARGO dictionary format.

        Returns:
            dict: Rules in the standard ARGO dictionary format.
        """

        if self.rule_strings != {}:
            self._rule_strings_to_rule_dicts()
        elif self.system_dicts != {}:
            self._system_dicts_to_rule_dicts()
        elif self.rule_lambdas != {}:
            self._rule_lambdas_to_rule_strings()
            self._rule_strings_to_rule_dicts()
        return self.rule_dicts

    def as_rule_strings(self, as_numpy: bool) -> dict:
        """
        Converts rules into the standard ARGO string format.

        Args:
            as_numpy (bool): If `True`, the conditions in the string format 
                will uses Numpy rather than Pandas. These rules are generally 
                evaluated more quickly on larger dataset stored as Pandas 
                DataFrames.

        Returns:
            dict: Rules in the standard ARGO string format.
        """
        if self.rule_strings != {}:
            self._rule_strings_to_rule_dicts()
        elif self.system_dicts != {}:
            self._system_dicts_to_rule_dicts()
        elif self.rule_lambdas != {}:
            self._rule_lambdas_to_rule_strings()
            self._rule_strings_to_rule_dicts()
        self._rule_dicts_to_rule_strings(as_numpy=as_numpy)
        return self.rule_strings

    def as_rule_lambdas(self, as_numpy: bool, with_kwargs: bool) -> dict:
        """
        Converts rules into the standard ARGO lambda expression format.

        Args:
            as_numpy (bool): If `True`, the conditions in the string format 
                will uses Numpy rather than Pandas. These rules are generally 
                evaluated more quickly on larger dataset stored as Pandas 
                DataFrames.
            with_kwargs (bool): If `True`, the string in the lambda expression 
                is created such that the inputs are keyword arguments. If 
                `False`, the inputs are positional arguments.      

        Returns:
            dict: Rules in the standard ARGO lambda expression format.
        """

        if self.rule_lambdas != {}:
            self._rule_lambdas_to_rule_strings()
            self._rule_strings_to_rule_dicts()
        elif self.system_dicts != {}:
            self._system_dicts_to_rule_dicts()
        elif self.rule_strings != {}:
            self._rule_strings_to_rule_dicts()
        self._rule_dicts_to_rule_lambdas(
            as_numpy=as_numpy, with_kwargs=with_kwargs)
        return self.rule_lambdas

    def as_system_dicts(self, field_datatypes: dict,
                        cassandra_field_names: dict) -> dict:
        """
        Converts rules into the system-ready format.

        Args:
            field_datatypes (dict): The Cassandra datatypes (values) for each 
                pipeline output field (keys).
            cassandra_field_names (dict): The Cassandra field names (values) 
                for each pipeline output field (keys).

        Returns:
            dict: Rules in the system format.
        """
        if self.rule_strings != {}:
            self._rule_strings_to_rule_dicts()
        elif self.rule_lambdas != {}:
            self._rule_lambdas_to_rule_strings()
            self._rule_strings_to_rule_dicts()
        self._rule_dicts_to_system_dicts(field_datatypes=field_datatypes,
                                         cassandra_field_names=cassandra_field_names)
        return self.system_dicts

    def filter_rules(self, include=None, exclude=None) -> None:
        """
        Filters the rules stored in the class.

        Args:
            include (list, optional): The list of rule names to keep. 
                Defaults to `None`.
            exclude (list, optional): The list of rule names to drop. 
                Defaults to `None`.

        Raises:
            Exception: `include` and `exclude` cannot contain similar values.
        """

        if include is not None and exclude is not None:
            intersected = set.intersection(set(include), set(exclude))
            if len(intersected) > 0:
                raise Exception(
                    '`include` and `exclude` contain similar values')
        for d in [self.rule_strings, self.rule_dicts, self.rule_lambdas, self.system_dicts]:
            if d != {}:
                rule_names = list(d.keys())
                break
        for rule_name in rule_names:
            if (include is not None and rule_name not in include) or \
                    (exclude is not None and rule_name in exclude):
                self.rule_strings.pop(rule_name, None)
                self.rule_dicts.pop(rule_name, None)
                self.rule_lambdas.pop(rule_name, None)
                self.lambda_kwargs.pop(rule_name, None)
                self.lambda_args.pop(rule_name, None)
                self.rule_features.pop(rule_name, None)
                self.system_dicts.pop(rule_name, None)

    def get_rule_features(self) -> dict:
        """
        Returns the set of unique features present in each rule.

        Returns:
            dict: Set of unique features (values) in each rule 
                (keys).
        """

        if self.rule_dicts == {}:
            _ = self.as_rule_dicts()
        grf = GetRuleFeatures(rule_dicts=self.rule_dicts)
        self.rule_feature_set = grf.get()
        return self.rule_feature_set

    def _rule_dicts_to_system_dicts(self, field_datatypes: dict,
                                    cassandra_field_names: dict) -> None:
        """
        Convert the rules (each being represented in the standard ARGO 
        dictionary format) into the system format.
        """
        if self.rule_dicts == {}:
            raise ValueError('`rule_dicts` must be given')
        converter = ConvertRuleDictsToSystemDicts(
            rule_dicts=self.rule_dicts, field_datatypes=field_datatypes,
            cassandra_field_names=cassandra_field_names)
        self.system_dicts = converter.convert()

    def _system_dicts_to_rule_dicts(self) -> None:
        """
        Convert the rules (each being represented in the system format) into 
        the standard ARGO dictionary format.
        """
        if self.system_dicts == {}:
            raise ValueError('`system_dicts` must be given')
        converter = ConvertSystemDictsToRuleDicts(
            system_dicts=self.system_dicts)
        self.rule_dicts = converter.convert()
        self.unparsed_rules = converter.unparsed_rules

    def _rule_dicts_to_rule_strings(self, as_numpy: bool) -> None:
        """
        Converts the rules (each being represented in the standard ARGO 
        dictionary format) into the standard ARGO string format.        
        """

        if self.rule_dicts == {}:
            raise ValueError('`rule_dicts` must be given')
        converter = ConvertRuleDictsToRuleStrings(
            rule_dicts=self.rule_dicts)
        self.rule_strings = converter.convert(as_numpy=as_numpy)

    def _rule_strings_to_rule_dicts(self) -> None:
        """
        Converts the rules (each being represented in the standard ARGO string 
        format) into the standard ARGO dictionary format.        
        """

        if self.rule_strings == {}:
            raise ValueError('`rule_strings` must be given')
        converter = ConvertRuleStringsToRuleDicts(
            rule_strings=self.rule_strings)
        self.rule_dicts = converter.convert()

    def _rule_dicts_to_rule_lambdas(self, as_numpy: bool,
                                    with_kwargs: bool) -> None:
        """
        Converts the rules (each being represented in the standard ARGO 
        dictionary format) into the standard ARGO lambda expression format. 
        This is useful for optimising rules.
        """
        if self.rule_dicts == {}:
            raise ValueError('`rule_dicts` must be given')
        converter = ConvertRuleDictsToRuleLambdas(rule_dicts=self.rule_dicts)
        self.rule_lambdas = converter.convert(
            as_numpy=as_numpy, with_kwargs=with_kwargs)
        self.lambda_kwargs = converter.lambda_kwargs
        self.lambda_args = converter.lambda_args
        self.rule_features = converter.rule_features

    def _rule_lambdas_to_rule_strings(self) -> None:
        """
        Converts the rules (each being represented in the standard ARGO lambda 
        expression format) into the standard ARGO string format.
        """
        if self.rule_lambdas == {}:
            raise ValueError('`rule_lambdas` must be given')
        if self.lambda_kwargs == {} and self.lambda_args == {}:
            raise ValueError('`lambda_kwargs` or `lambda_args` must be given')
        converter = ConvertRuleLambdasToRuleStrings(
            rule_lambdas=self.rule_lambdas, lambda_kwargs=self.lambda_kwargs,
            lambda_args=self.lambda_args)
        self.rule_strings = converter.convert()
