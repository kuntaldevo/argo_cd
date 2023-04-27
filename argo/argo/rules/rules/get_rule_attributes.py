"""Class for returning set of features present in each rule"""


class GetRuleFeatures:
    """
    Returns the set of unique features present in each rule.

    Attributes:
        rule_features (dict): Set of unique features (values) in each rule 
            (keys).
    """

    def __init__(self, rule_dicts: dict):
        """
        Args:
            rule_dicts (dict): Set of rules defined using the standard ARGO 
                dictionary format (values) and their names (keys).
        """

        self.rule_dicts = rule_dicts
        self.rule_features = {}

    def get(self) -> dict:
        """
        Returns the set of unique features present in each rule.

        Returns:
            dict: Set of unique features (values) in each rule (keys).
        """

        for rule_name, rule_dict in self.rule_dicts.items():
            rule_string = self._get_rule_features(
                rule_dict=rule_dict)
            self.rule_features[rule_name] = rule_string
        return self.rule_features

    def _get_rule_features(self, rule_dict: dict) -> set:
        """Gets the unique set of features in the rule"""

        feature_set = set()
        feature_set = self._recurse_add_rule_features(
            rules_list=rule_dict['rules'], feature_set=feature_set)
        return feature_set

    def _recurse_add_rule_features(self, rules_list: list,
                                   feature_set: set) -> list:
        """
        Loops through list of dictionary conditions - if it is a single
        condition, extracts the feature; if not, then it goes to the next 
        level down in the rule.
        """

        for rule in rules_list:
            rule_keys = list(rule.keys())
            rule_keys.sort()
            if rule_keys == ['condition', 'rules']:
                feature = self._recurse_add_rule_features(
                    rules_list=rule['rules'], feature_set=feature_set)
            else:
                feature = rule['field']
                feature_set.add(feature)
                # If field comparison, include field that is being compared to
                if rule['operator'].endswith('_field'):
                    feature_set.add(rule['value'])
        return feature_set
