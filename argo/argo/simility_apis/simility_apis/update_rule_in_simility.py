"""Class for updating a rule in Simility"""
import requests
import json
import simility_apis.set_password


class UpdateRuleInSimilityAPI:

    """
    Updates an existing rule in Simility, given its stringified rule JSON 
    config.
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
        self.user_login = f'{self.app_prefix}.{self.user}'

    def request(self, rule_config: str) -> None:
        """
        Updates an existing rule in Simility, given its stringified rule JSON 
        config.

        Args:
            rule_config (str): The stringified JSON configs of the rule to be 
                created in the Simility platform.

        Raises:
            Exception: If the request returns an error, an exception is 
                thrown and the request error is shown.
        """
        rule_config_dict = json.loads(rule_config)
        rule_name = rule_config_dict['name']
        rule_id = str(rule_config_dict['id'])
        header = {
            "Connection": "keep-alive",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.87 Safari/537.36",
            "Content-Type": "application/json",
            "Accept": "*/*",
            "X-Requested-With": "XMLHttpRequest",
            "Sec-Fetch-Dest": "empty",
            "Origin": self.url,
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-Mode": "cors",
            "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate",

        }
        endpoint = f'{self.url}/server/rest/rule/{rule_id}?runRule=false'
        response = requests.put(endpoint,
                                auth=(self.user_login,
                                      simility_apis.set_password.PASSWORD),
                                data=rule_config,
                                headers=header
                                )
        if not response.ok:
            raise Exception(
                f'The API request threw a {response} error.')
