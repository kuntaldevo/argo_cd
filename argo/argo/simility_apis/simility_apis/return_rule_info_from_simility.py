"""Class for returning rule related information from Simility"""
import requests
import pandas as pd
import simility_apis.set_password


class ReturnRuleInfoFromSimilityAPI:

    """
    API for returning the rule configuration JSONs related to the rules found 
    in a Simility environment.
    """

    def __init__(self, url: str, app_prefix: str, entity: str, user: str):
        """
        Args:
            url (str): The URL of the desired cluster (e.g. 
                'https://app.simility.com' for US-SaaS).
            app_prefix (str): Keyspace name on the cluster.
            entity (str): The entity which the rules are built on.
            user (str): Username to access the cluster.
        """

        if url.endswith('/'):
            self.url = url[:-1]
        else:
            self.url = url
        self.app_prefix = app_prefix
        self.entity = entity
        self.user = user

    def request(self) -> object:
        """
        API for returning the information related to each rule in a Simility 
        instance.

        Raises:
            Exception: If the request returns an error, an exception is 
                thrown and the request error is shown.

        Returns:
            object: The response containing the rule configuration JSONs for 
                each rule in a Simility environment. 
        """
        endpoint = f'{self.url}/server/rest/rule/list/{self.app_prefix}/{self.entity}'
        user_login = f'{self.app_prefix}.{self.user}'
        response = requests.get(endpoint, headers={
            'Content-Type': 'application/json'}, auth=(user_login, simility_apis.set_password.PASSWORD))
        if not response.ok:
            raise Exception('The API request threw a ' +
                            str(response) + ' error.')
        return response
