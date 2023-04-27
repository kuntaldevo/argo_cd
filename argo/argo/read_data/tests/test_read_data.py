import pytest
from read_data.read_data import DataReader
import simility_apis.set_password
import httpretty
import numpy as np
import pandas as pd
import os


@pytest.fixture
def _instantiate():
    params = {
        "url": 'http://sim-ds.us-central1.gcp.dev.paypalinc.com',
        "app_prefix": 'james_testing',
        "user": 'james@simility.com',
        "base_entity": 'transaction',
    }
    url = params['url']
    app_prefix = params['app_prefix']
    endpoint = f'{url}/server/rest/businessentity/entityinfo/{app_prefix}'
    return params, endpoint


@pytest.fixture
def _body():
    body = '{"appId": "1", "entities": [{"entity": {"name": "account_number"}, "entityName": "account_number", "fields": [{"name": "num_distinct_txn_per_account_number_1day", "repeated": false, "sources": "SIGNAL: postrel02_account_number_proto_agg", "type": "INT"}]}, {"entity": {"name": "transaction"}, "entityName": "transaction", "fields": [{"name": "eid", "repeated": false, "type": "TEXT"}, {"name": "is_existing_user", "repeated": false, "type": "BOOLEAN"}], "metas": [{"name": "account_number", "columnName": "account_number", "fieldType": "TEXT", "refBE": "account_number", "rid": 25459, "scope": "REFERENCE", "tableName": "transaction"}, {"name": "primary_address", "columnName": "primary_address", "fieldType": "TEXT", "refBE": "address", "rid": 25459, "scope": "REFERENCE", "tableName": "transaction"}]}, {"entity": {"name": "address"}, "entityName": "address", "fields": [{"name": "avg_order_total_per_primary_address_7day", "repeated": false, "sources": "SIGNAL: postrel02_address_proto_agg", "type": "DOUBLE"}]}]}'
    return body


@pytest.fixture
def _generate_data():
    X = pd.DataFrame({
        'eid': ['txn1', 'txn2', 'txn3'],
        'account_number_num_distinct_txn_per_account_number_1day': [1, 2, np.nan],
        'primary_address_avg_order_total_per_primary_address_7day': [1.5, 3.0, np.nan],
        'is_existing_user': [True, False, np.nan]
    })
    X.set_index('eid', inplace=True)
    X.to_csv('dummy_data.csv')


@pytest.fixture
def _expected_result():
    X = pd.DataFrame(
        np.array([['txn1', 1, 1.5, True],
                  ['txn2', 2, 3.0, False],
                  ['txn3', pd.NA, np.nan, pd.NA]]),
        columns=['eid', 'account_number_num_distinct_txn_per_account_number_1day',
                 'primary_address_avg_order_total_per_primary_address_7day',
                 'is_existing_user'])
    X['account_number_num_distinct_txn_per_account_number_1day'] = X[
        'account_number_num_distinct_txn_per_account_number_1day'].astype('Int64')
    X['primary_address_avg_order_total_per_primary_address_7day'] = X[
        'primary_address_avg_order_total_per_primary_address_7day'].astype('float64')
    X['is_existing_user'] = X['is_existing_user'].astype('boolean')
    X.set_index('eid', inplace=True)
    return X


@httpretty.activate
def test_read_csv(_instantiate, _body, _expected_result, _generate_data):
    _generate_data
    simility_apis.set_password.PASSWORD = 'abc'
    params, endpoint = _instantiate
    body = _body
    expected_result = _expected_result
    httpretty.enable()
    httpretty.register_uri(httpretty.GET, endpoint, body=body)
    dr = DataReader(**params)
    data = dr.read_csv(filepath="./dummy_data.csv", index_col="eid")
    assert all(expected_result == data)
    httpretty.disable()
    os.system("rm ./dummy_data.csv")
