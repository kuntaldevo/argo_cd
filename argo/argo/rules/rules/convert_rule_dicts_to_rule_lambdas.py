"""Class for converting between dict and lambda representations of rules"""
from rules.convert_rule_dicts_to_rule_strings import ConvertRuleDictsToRuleStrings


class ConvertRuleDictsToRuleLambdas:
    """
    Converts a set of rules (each being represented in the standard ARGO 
    dictionary format) into the standard ARGO lambda expression format.        

    Attributes:
        rule_lambdas (dict): Set of rules defined using the standard ARGO 
            lambda expression format (values) and their names (keys).
        lambda_kwargs (dict): For each rule (keys), a dictionary containing the
            features used in the rule (keys) and the current values (values). 
            Only populates when `with_kwargs=True`.
        lambda_args (dict): For each rule (keys), a list containing the current 
            values used in the rule. Only populates when `with_kwargs=False`.
        rule_features (dict): For each rule (keys), a list containing the 
            features used in the rule. Only populates when `with_kwargs=False`.        
    """

    def __init__(self, rule_dicts: dict):
        """
        Args:
            rule_dicts (dict): Set of rules defined using the standard ARGO 
                dictionary format (values) and their names (keys).
        """

        self.rule_dicts = rule_dicts

    def convert(self, as_numpy: bool, with_kwargs: bool) -> dict:
        """
        Converts a set of rules (each being represented in the standard ARGO 
        dictionary format) into the standard ARGO lambda expression format.        

        Args:
            as_numpy (bool): If True, the conditions in the string format will 
                uses Numpy rather than Pandas. These rules are generally 
                evaluated more quickly on larger dataset stored as Pandas 
                DataFrames.
            with_kwargs (bool): If True, the string in the lambda expression is
                created such that the inputs are keyword arguments. If False, 
                the inputs are positional arguments.                                       

        Returns:
            dict: Set of rules defined using the standard ARGO lambda 
                expression format (values) and their names (keys).
        """

        converter = ConvertRuleDictsToRuleStrings(rule_dicts=self.rule_dicts)
        self.rule_lambdas, self.lambda_kwargs, self.lambda_args, self.rule_features = converter._convert_to_lambda(
            as_numpy=as_numpy, with_kwargs=with_kwargs)
        return self.rule_lambdas
