import pytest
import json
from simility_apis.return_cassandra_entity_info import ReturnCassandraEntityInfoAPI
import simility_apis.set_password
import httpretty


@pytest.fixture
def _instantiate():
    params = {
        "url": 'http://sim-ds.us-central1.gcp.dev.paypalinc.com',
        "app_prefix": 'james_testing',
        "user": 'james@simility.com'
    }
    url = params['url']
    app_prefix = params['app_prefix']
    endpoint = f'{url}/server/rest/businessentity/entityinfo/{app_prefix}'
    api = ReturnCassandraEntityInfoAPI(**params)
    return api, endpoint


@httpretty.activate
def test_request(_instantiate):
    simility_apis.set_password.PASSWORD = 'abc'
    api, endpoint = _instantiate
    httpretty.enable()
    httpretty.register_uri(httpretty.GET, endpoint,
                           body='{"appId": 31, "entities": [], "fieldTypes": []}')
    response = api.request()
    response_json = json.loads(response.text)
    assert list(response_json.keys()) == ['appId', 'entities', 'fieldTypes']
    httpretty.disable()


@httpretty.activate
def test_request_bad_connection(_instantiate):
    simility_apis.set_password.PASSWORD = 'abc'
    api, endpoint = _instantiate
    httpretty.enable()
    httpretty.register_uri(httpretty.GET, endpoint,
                           status=401)
    with pytest.raises(Exception) as e:
        api.request()
        assert e == 'The API request threw a 401 error.'
    httpretty.disable()


def test_request_no_password_error(_instantiate):
    api = _instantiate
    with pytest.raises(AttributeError) as e:
        api.request()
        assert e == "module 'simility_apis.set_password' has no attribute 'PASSWORD'"
