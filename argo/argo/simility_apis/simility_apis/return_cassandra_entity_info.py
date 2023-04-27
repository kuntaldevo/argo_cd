"""Class for returning Cassandra entity information from Simility"""
import pandas as pd
import requests
import simility_apis.set_password


class ReturnCassandraEntityInfoAPI:
    """
    API for returning the Cassandra entity information associated with a keyspace.
    """

    def __init__(self, url: str, app_prefix: str, user: str):
        """
        Args:
            url (str): The URL of the desired cluster (e.g. 
                'https://app.simility.com' for US-SaaS)
            app_prefix (str): Keyspace name on the cluster
            user (str): Username to access the cluster            
        """

        if url.endswith('/'):
            self.url = url[:-1]
        else:
            self.url = url
        self.app_prefix = app_prefix
        self.user = user

    def request(self) -> object:
        """
        API for returning the Cassandra entity information associated with a 
        keyspace.

        Raises:
            Exception: If the request returns an error, an exception is 
                thrown and the request error is shown.

        Returns:
            object: The response containing the entity information.
        """

        endpoint = f'{self.url}/server/rest/businessentity/entityinfo/{self.app_prefix}'
        user_login = f'{self.app_prefix}.{self.user}'
        response = requests.get(endpoint, headers={
            'Content-Type': 'application/json'}, auth=(user_login,
                                                       simility_apis.set_password.PASSWORD))
        if not response.ok:
            raise Exception(
                f'The API request threw a {response} error.')
        return response
