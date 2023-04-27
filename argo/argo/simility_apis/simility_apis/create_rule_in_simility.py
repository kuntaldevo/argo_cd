"""Class for creating a rule in Simility"""
import requests
import simility_apis.set_password


class CreateRuleInSimilityAPI:

    """
    API for creating a rule in a Simility instance, given the stringified rule 
    JSON config.
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

    def request(self, rule_config: str) -> None:
        """
        API for creating a rule in a Simility instance, given the rule JSON 
        config.

        Args:
            rule_config (str): The configuration (stringified system JSON 
                config) of the rule that is to be created in the Simility 
                platform.

        Raises:
            Exception: If the request returns an error, an exception is 
                thrown and the request error is shown.
        """

        user_login = f'{self.app_prefix}.{self.user}'
        endpoint = f'{self.url}/server/rest/rule?runRule=false'
        response = requests.post(endpoint,
                                 auth=(user_login,
                                       simility_apis.set_password.PASSWORD),
                                 data=rule_config,
                                 headers={
                                     "content-type": "application/json"}
                                 )
        if not response.ok:
            raise Exception(f'The API request threw a {response} error.')
