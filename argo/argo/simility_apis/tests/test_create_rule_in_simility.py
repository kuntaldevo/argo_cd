import pytest
from simility_apis.create_rule_in_simility import CreateRuleInSimilityAPI
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
    endpoint = f'{url}/server/rest/rule?runRule=false'
    api = CreateRuleInSimilityAPI(**params)
    return api, endpoint


@httpretty.activate
def test_request(_instantiate):
    simility_apis.set_password.PASSWORD = 'abc'
    api, endpoint = _instantiate
    httpretty.enable()
    httpretty.register_uri(httpretty.POST, endpoint)
    response = api.request(rule_config='{}')
    assert response == None
    httpretty.disable()


@httpretty.activate
def test_request_bad_connection(_instantiate):
    simility_apis.set_password.PASSWORD = 'abc'
    api, endpoint = _instantiate
    httpretty.enable()
    httpretty.register_uri(httpretty.POST, endpoint,
                           status=401)
    with pytest.raises(Exception) as e:
        api.request(rule_config='{}')
        assert e == 'The API request threw a 401 error.'
    httpretty.disable()


def test_request_no_password_error(_instantiate):
    api = _instantiate
    with pytest.raises(AttributeError) as e:
        api.request(rule_config='{}')
        assert e == "module 'simility_apis.set_password' has no attribute 'PASSWORD'"
