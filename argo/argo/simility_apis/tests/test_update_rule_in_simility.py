import pytest
from simility_apis.update_rule_in_simility import UpdateRuleInSimilityAPI
import simility_apis.set_password
import httpretty
import json


@pytest.fixture
def _instantiate():
    params = {
        "url": 'http://sim-ds.us-central1.gcp.dev.paypalinc.com',
        "app_prefix": 'james_testing',
        "user": 'james@simility.com'
    }
    url = params['url']
    rule_config = '{"id":123, "name": "test_rule"}'
    rule_id = json.loads(rule_config)['id']
    endpoint = f'{url}/server/rest/rule/{rule_id}?runRule=false'
    api = UpdateRuleInSimilityAPI(**params)
    return api, endpoint, rule_config


@httpretty.activate
def test_request(_instantiate):
    simility_apis.set_password.PASSWORD = 'abc'
    api, endpoint, rule_config = _instantiate
    httpretty.enable()
    httpretty.register_uri(httpretty.PUT, endpoint)
    response = api.request(rule_config=rule_config)
    assert response == None
    httpretty.disable()


@httpretty.activate
def test_request_bad_connection(_instantiate):
    simility_apis.set_password.PASSWORD = 'abc'
    api, endpoint, rule_config = _instantiate
    httpretty.enable()
    httpretty.register_uri(httpretty.PUT, endpoint,
                           status=401)
    with pytest.raises(Exception) as e:
        api.request(rule_config=rule_config)
        assert e == 'The API request threw a 401 error.'
    httpretty.disable()


def test_request_no_password_error(_instantiate):
    api = _instantiate
    with pytest.raises(AttributeError) as e:
        api.request(rule_config='{}')
        assert e == "module 'simility_apis.set_password' has no attribute 'PASSWORD'"
