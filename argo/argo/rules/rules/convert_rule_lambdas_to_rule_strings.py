"""Class for converting between lambda and string representations of rules"""


class ConvertRuleLambdasToRuleStrings:
    """
    Converts a set of rules (each being represented in the standard ARGO 
    lambda expression format) into the standard ARGO string format.   

    Attributes:
        rule_strings (dict): Set of rules defined using the standard ARGO 
            string format (values) and their names (keys).
    """

    def __init__(self, rule_lambdas: dict, lambda_kwargs=None,
                 lambda_args=None):
        """
        rule_lambdas (dict): Set of rules defined using the standard ARGO 
            lambda expression format (values) and their names (keys).
        lambda_kwargs (dict): For each rule (keys), a dictionary containing the 
            features used in the rule (keys) and the current values (values). 
            Defaults to None.            
        lambda_args (dict): For each rule (keys), a list containing the current
            values used in the rule. Defaults to None.        

        Raises:
            Exception: Either `lambda_kwargs` or `lambda_args` must be
                provided.
        """
        if lambda_kwargs is None and lambda_args is None:
            raise Exception(
                'Either `lambda_kwargs` or `lambda_args` must be provided')
        self.rule_lambdas = rule_lambdas
        self.lambda_kwargs = {} if lambda_kwargs is None else lambda_kwargs
        self.lambda_args = {} if lambda_args is None else lambda_args
        self.rule_strings = {}

    def convert(self) -> dict:
        """
        Converts a set of rules (each being represented in the standard ARGO 
        lambda expression format) into the standard ARGO string format.   

        Returns:
            dict: Set of rules defined using the standard ARGO string format 
                (values) and their names (keys).
        """
        for rule_name, rule_lambda in self.rule_lambdas.items():
            if self.lambda_kwargs != {}:
                rule_string = rule_lambda(**self.lambda_kwargs[rule_name])
            else:
                rule_string = rule_lambda(*self.lambda_args[rule_name])
            self.rule_strings[rule_name] = rule_string
        return self.rule_strings
