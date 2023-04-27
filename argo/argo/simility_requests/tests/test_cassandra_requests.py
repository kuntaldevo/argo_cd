import pytest
import pickle
import json
from simility_requests.cassandra_requests import ReturnCassandraDatatypes,\
    ReturnPipelineOutputDatatypes, ReturnCassandraPipelineOutputMapping
import simility_apis.set_password
import httpretty


@pytest.fixture
def _params():
    params = {
        "url": 'http://sim-ds.us-central1.gcp.dev.paypalinc.com',
        "app_prefix": 'james_testing',
        "user": 'james@simility.com',
        "base_entity": 'transaction'
    }
    url = params['url']
    app_prefix = params['app_prefix']
    endpoint = f'{url}/server/rest/businessentity/entityinfo/{app_prefix}'
    return params, endpoint


@pytest.fixture
def _body():
    body = '{"appId": "1", "entities": [{"entity": {"name": "account_number"}, "entityName": "account_number", "fields": [{"name": "num_distinct_txn_per_account_number_1day", "repeated": false, "sources": "SIGNAL: postrel02_account_number_proto_agg", "type": "INT"}]}, {"entity": {"name": "transaction"}, "entityName": "transaction", "fields": [{"name": "eid", "repeated": false, "type": "TEXT"}, {"name": "is_existing_user", "repeated": false, "type": "BOOLEAN"}], "metas": [{"name": "account_number", "columnName": "account_number", "fieldType": "TEXT", "refBE": "account_number", "rid": 25459, "scope": "REFERENCE", "tableName": "transaction"}, {"name": "primary_address", "columnName": "primary_address", "fieldType": "TEXT", "refBE": "address", "rid": 25459, "scope": "REFERENCE", "tableName": "transaction"}]}, {"entity": {"name": "address"}, "entityName": "address", "fields": [{"name": "avg_order_total_per_primary_address_7day", "repeated": false, "sources": "SIGNAL: postrel02_address_proto_agg", "type": "DOUBLE"}]}]}'
    return body


class TestReturnCassandraDatatypes:
    @httpretty.activate
    def test_request(self, _params, _body):
        simility_apis.set_password.PASSWORD = 'abc'
        params, endpoint = _params
        api = ReturnCassandraDatatypes(**params)
        body = _body
        httpretty.enable()
        httpretty.register_uri(httpretty.GET, endpoint, body=body)
        cass_dtypes = api.request()
        assert list(cass_dtypes.columns) == [
            'PipelineOutputFieldName', 'CassandraDatatype']
        assert cass_dtypes.shape == (4, 2)
        httpretty.disable()

    @httpretty.activate
    def test_request_for_entity_info(self, _params, _body):
        simility_apis.set_password.PASSWORD = 'abc'
        params, endpoint = _params
        api = ReturnCassandraDatatypes(**params)
        body = _body
        httpretty.enable()
        httpretty.register_uri(httpretty.GET, endpoint, body=body)
        entities_list = api._request_for_entity_info(url='http://sim-ds.us-central1.gcp.dev.paypalinc.com',
                                                     app_prefix='james_testing',
                                                     user='james@simility.com')
        assert entities_list == json.loads(body)['entities']
        httpretty.disable()

    def test_get_refs(self, _body, _params):
        body = _body
        entities_list = json.loads(body)['entities']
        params, _ = _params
        api = ReturnCassandraDatatypes(**params)
        references = api._get_refs(
            entities_list=entities_list, base_entity='transaction')
        assert references == {
            'account_number': 'account_number', 'primary_address': 'address'}

    def test_filter_entities_list(self, _body, _params):
        body = _body
        entities_list = json.loads(body)['entities']
        references = {'account_number': 'account_number',
                      'primary_address': 'address'}
        params, _ = _params
        api = ReturnCassandraDatatypes(**params)
        filtered_entities_list = api._filter_entities_list(
            references=references, base_entity='transaction', entities_list=entities_list)
        filtered_entities_names = [f['entityName']
                                   for f in filtered_entities_list]
        filtered_entities_names.sort()
        assert filtered_entities_names == [
            'account_number', 'address', 'transaction']

    def test_parse_entities_list(self, _body, _params):
        body = _body
        entities_list = json.loads(body)['entities']
        filtered_entities_list = [e for e in entities_list if e['entityName'] in [
            'account_number', 'address', 'transaction']]
        references = {'account_number': 'account_number',
                      'primary_address': 'address'}
        params, _ = _params
        api = ReturnCassandraDatatypes(**params)
        cass_dtypes = api._parse_entities_list(
            entities_list=filtered_entities_list, base_entity='transaction', references=references)
        assert list(cass_dtypes.columns) == [
            'PipelineOutputFieldName', 'CassandraDatatype']
        assert cass_dtypes.shape == (4, 2)


class TestReturnPipelineOutputDatatypes:

    @httpretty.activate
    def test_request(self, _params, _body):
        simility_apis.set_password.PASSWORD = 'abc'
        params, endpoint = _params
        api = ReturnPipelineOutputDatatypes(**params)
        body = _body
        httpretty.enable()
        httpretty.register_uri(httpretty.GET, endpoint, body=body)
        po_datatypes = api.request()
        assert po_datatypes == {
            'eid': 'TEXT',
            'account_number_num_distinct_txn_per_account_number_1day': 'INT',
            'primary_address_avg_order_total_per_primary_address_7day': 'DOUBLE',
            'is_existing_user': 'BOOLEAN'
        }
        httpretty.disable()


class TestReturnCassandraPipelineOutputMapping:

    @httpretty.activate
    def test_request(self, _params, _body):
        simility_apis.set_password.PASSWORD = 'abc'
        params, endpoint = _params
        api = ReturnCassandraPipelineOutputMapping(**params)
        body = _body
        httpretty.enable()
        httpretty.register_uri(httpretty.GET, endpoint, body=body)
        cass_po_mapping = api.request()
        assert cass_po_mapping == {
            'eid': 'eid',
            'account_number_num_distinct_txn_per_account_number_1day': 'account_number.num_distinct_txn_per_account_number_1day',
            'primary_address_avg_order_total_per_primary_address_7day': 'primary_address.avg_order_total_per_primary_address_7day',
            'is_existing_user': 'is_existing_user'
        }
        httpretty.disable()
