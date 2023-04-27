"""Classes for returning Cassandra related information from Simility"""
import pandas as pd
import numpy as np
import requests
import json
from simility_apis.return_cassandra_entity_info import ReturnCassandraEntityInfoAPI
from operator import itemgetter


class ReturnCassandraDatatypes:
    """
    Returns the Cassandra datatypes of the fields present in both Cassandra 
    and pipeline output.
    """

    def __init__(self, url: str, app_prefix: str, user: str, base_entity: str):
        """
        Args:
            url (str): The URL of the desired cluster (e.g. 
                'https://app.simility.com' for US-SaaS)
            app_prefix (str): Keyspace name on the cluster
            user (str): Username to access the cluster
            base_entity (str): Base entity name of the pipeline output data            
        """

        if url.endswith('/'):
            self.url = url[:-1]
        else:
            self.url = url
        self.app_prefix = app_prefix
        self.user = user
        self.base_entity = base_entity

    def request(self) -> pd.DataFrame:
        """
        Returns the Cassandra datatypes of the fields present in both Cassandra 
        and pipeline output.

        Returns:
            pd.DataFrame: Contains the list of entities, their list of entity 
                fields, the pipeline output field names and the Cassandra 
                datatypes.
        """

        entities_list = self._request_for_entity_info(
            url=self.url, app_prefix=self.app_prefix, user=self.user)
        references = self._get_refs(
            entities_list=entities_list, base_entity=self.base_entity)
        entities_list = self._filter_entities_list(
            references=references, base_entity=self.base_entity,
            entities_list=entities_list)
        cassandra_dtypes = self._parse_entities_list(
            entities_list=entities_list, base_entity=self.base_entity,
            references=references)
        return cassandra_dtypes

    @staticmethod
    def _request_for_entity_info(url: str, app_prefix: str, user: str) -> list:
        """
        Fetches the entity information via the API and converts it into a list.

        Raises:
            Exception: If the request returns an error, an exception is 
                thrown and the request error is shown.
        """

        rcdt = ReturnCassandraEntityInfoAPI(
            url=url, app_prefix=app_prefix, user=user)
        dtypes_req = rcdt.request()
        entities_list = json.loads(dtypes_req.text)['entities']
        return entities_list

    @staticmethod
    def _get_refs(entities_list: list, base_entity: str) -> dict:
        """
        Fetches the reference fields used on entities running streaming 
        pipeline (i.e. base entities)
        """

        for entity in entities_list:
            entity_name = entity['entityName']
            if entity_name == base_entity:
                references = {}
                for field in entity['metas']:
                    if 'refBE' in field.keys():
                        references[field['name']] = field['refBE']
        return references

    @staticmethod
    def _filter_entities_list(references: dict, base_entity: str,
                              entities_list: list) -> list:
        """Filters out entities not connected to the base entity"""

        child_entities_to_keep = list(references.values())
        entities_to_keep = child_entities_to_keep + [base_entity]
        ind_to_keep = [i for i, entity in enumerate(
            entities_list) if entity['entityName'] in entities_to_keep]
        entities_list = itemgetter(*ind_to_keep)(entities_list)
        if isinstance(entities_list, dict):
            entities_list = [entities_list]
        return entities_list

    @staticmethod
    def _parse_entities_list(entities_list: list, base_entity: str,
                             references: dict) -> pd.DataFrame:
        """
        Method for parsing the raw entity information into the final 
        dataframe
        """

        cassandra_dtypes = pd.DataFrame(columns=[
            'Entity', 'ReferenceField', 'CassandraFieldName',
            'PipelineOutputFieldName', 'CassandraDatatype'
        ])
        cassandra_dtypes.set_index(
            ['Entity', 'ReferenceField', 'CassandraFieldName'], inplace=True)

        for entity in entities_list:
            entity_name = entity['entityName']
            # If entity is not a base entity, get list of ref fields from base
            # entity to this entity
            if entity_name != base_entity:
                ref_fields_from_be = [ref_field for (
                    ref_field, entity) in references.items() if entity in entity_name]
            for field_json in entity['fields']:
                field_name = field_json['name']
                field_dtype = field_json['type']
                # If entity is base entity, pipeline output field name =
                # Cassandra field name
                if entity_name == base_entity:
                    pipeline_output_field_name = field_name
                    cassandra_dtypes.loc[
                        (entity_name, '', field_name), :] = [pipeline_output_field_name, field_dtype]
                # Else entity must be child entity, so pipeline output field
                # name = base entity ref field + child entity Cassandra field
                # name
                else:
                    for ref_field_from_be in ref_fields_from_be:
                        pipeline_output_field_name = ref_field_from_be + '_' + field_name
                        cassandra_dtypes.loc[(entity_name, ref_field_from_be, field_name), :] = [
                            pipeline_output_field_name, field_dtype]
        return cassandra_dtypes


class ReturnPipelineOutputDatatypes:
    """
    Returns the Cassandra datatype associated with each field in pipeline 
    output.
    """

    def __init__(self, url: str, app_prefix: str, user: str, base_entity: str):
        """
        Args:
            url (str): The URL of the desired cluster (e.g. 
                'https://app.simility.com' for US-SaaS)
            app_prefix (str): Keyspace name on the cluster
            user (str): Username to access the cluster
            base_entity (str): Base entity name of the pipeline output data     
        """
        self.url = url
        self.app_prefix = app_prefix
        self.user = user
        self.base_entity = base_entity

    def request(self) -> dict:
        """
        Returns the Cassandra datatype associated with each field in pipeline 
        output.

        Returns:
            dict: The Cassandra datatype (values) associated with each 
                field in pipeline output (keys).
        """
        r = ReturnCassandraDatatypes(url=self.url, app_prefix=self.app_prefix,
                                     user=self.user, base_entity=self.base_entity)
        cass_dtypes = r.request()
        po_datatypes = cass_dtypes.reset_index()[['PipelineOutputFieldName', 'CassandraDatatype']].set_index(
            'PipelineOutputFieldName').squeeze().to_dict()
        return po_datatypes


class ReturnCassandraPipelineOutputMapping:
    """
    Returns the Cassandra field name associated with each pipeline output 
    field.
    """

    def __init__(self, url: str, app_prefix: str, user: str, base_entity: str):
        """
        Args:
            url (str): The URL of the desired cluster (e.g. 
                'https://app.simility.com' for US-SaaS)
            app_prefix (str): Keyspace name on the cluster
            user (str): Username to access the cluster
            base_entity (str): Base entity name of the pipeline output data 
        """
        self.url = url
        self.app_prefix = app_prefix
        self.user = user
        self.base_entity = base_entity

    def request(self) -> dict:
        """
        Returns the Cassandra field name associated with each pipeline output 
        field.

        Returns:
            dict: The Cassandra field name (values) associated with each 
                pipeline output field (keys).
        """
        r = ReturnCassandraDatatypes(
            url=self.url, app_prefix=self.app_prefix, user=self.user,
            base_entity=self.base_entity)
        cass_dtypes = r.request()
        cass_dtypes = cass_dtypes.reset_index()
        cassandra_pipeline_output_mapping = {}
        for _, row in cass_dtypes.iterrows():
            if row['ReferenceField'] == '':
                cassandra_pipeline_output_mapping[row['PipelineOutputFieldName']
                                                  ] = row['CassandraFieldName']
            else:
                cassandra_pipeline_output_mapping[row['PipelineOutputFieldName']
                                                  ] = row['ReferenceField'] + '.' + row['CassandraFieldName']
        return cassandra_pipeline_output_mapping
