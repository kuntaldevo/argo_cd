"""Methods for reading different data formats"""
from simility_requests.cassandra_requests import ReturnPipelineOutputDatatypes
import pandas as pd

CASS_PYTHON_DTYPE_LOOKUP = {
    'DOUBLE': float,
    'TEXT': object,
    'INT': 'Int64',
    'BIGINT': 'Int64',
    'BOOLEAN': 'boolean',
    'TIMESTAMP': object,
    'LIST': object,
    'SET': object,
    'MAP': object,
    'FLOAT': float,
    'BLOB': object
}


class DataReader:

    """
    Contains methods for reading pipeline output data in different data formats
    while ensuring the Python datatypes match those used in Cassandra.
    """

    def __init__(self, url: str, app_prefix: str, user: str, base_entity: str,
                 cass_python_dtype_mapping=CASS_PYTHON_DTYPE_LOOKUP):
        """
        Args:
            url (str): The URL of the desired cluster (e.g. 
                'https://app.simility.com' for US-SaaS)
            app_prefix (str): Keyspace name on the cluster
            user (str): Username to access the cluster
            base_entity (str): Base entity name of the pipeline output data            
            cass_python_dtype_mapping (dict, optional): Mapping to use for 
                converting Cassandra datatypes (keys) to Pandas datatypes 
                (values). Defaults to CASS_PYTHON_DTYPE_LOOKUP.
        """
        rpodt = ReturnPipelineOutputDatatypes(
            url=url, app_prefix=app_prefix, user=user, base_entity=base_entity)
        cass_dtypes = rpodt.request()
        self.python_dtypes = dict((field, cass_python_dtype_mapping[cass_dtype])
                                  for field, cass_dtype in cass_dtypes.items())

    def read_csv(self, filepath: str, **kwargs) -> pd.DataFrame:
        """
        Reads a CSV file into a dataframe, using the Cassandra datatypes of 
        each field to set their datatypes in the dataframe.

        Args:
            filepath (str): Path to the CSV file.                    
            **kwargs (dict): Any keyword arguments to pass to the Pandas 
                read_csv method.
        Returns:
            pd.DataFrame: Dataframe of the CSV file with Python datatypes 
                aligning to the fields Cassandra datatypes.
        """
        data = pd.read_csv(filepath_or_buffer=filepath,
                           dtype=self.python_dtypes, **kwargs)
        return data
