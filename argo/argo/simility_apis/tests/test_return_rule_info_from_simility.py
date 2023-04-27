import pytest
from simility_apis.return_rule_info_from_simility import ReturnRuleInfoFromSimilityAPI
import simility_apis.set_password
import httpretty
import json


@pytest.fixture
def _instantiate():
    params = {
        "url": 'http://sim-ds.us-central1.gcp.dev.paypalinc.com',
        "app_prefix": 'james_testing',
        "entity": "transaction",
        "user": 'james@simility.com'
    }
    url = params['url']
    app_prefix = params['app_prefix']
    entity = params['entity']
    endpoint = f'{url}/server/rest/rule/list/{app_prefix}/{entity}'
    api = ReturnRuleInfoFromSimilityAPI(**params)
    return api, endpoint


@httpretty.activate
def test_request(_instantiate):
    simility_apis.set_password.PASSWORD = 'abc'
    api, endpoint = _instantiate
    httpretty.enable()
    httpretty.register_uri(httpretty.GET, endpoint, body='{"entries": []}')
    response = api.request()
    response_json = json.loads(response.text)
    assert list(response_json.keys()) == ['entries']
    httpretty.disable()


@httpretty.activate
def test_request_bad_connection(_instantiate):
    simility_apis.set_password.PASSWORD = 'abc'
    api, endpoint = _instantiate
    httpretty.enable()
    httpretty.register_uri(httpretty.GET, endpoint, status=401)
    with pytest.raises(Exception) as e:
        api.request(rule_config='{}')
        assert e == 'The API request threw a 401 error.'
    httpretty.disable()


def test_request_no_password_error(_instantiate):
    api = _instantiate
    with pytest.raises(AttributeError) as e:
        api.request(rule_config='{}')
        assert e == "module 'simility_apis.set_password' has no attribute 'PASSWORD'"
