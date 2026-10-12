from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from routing_lab.token_broker import DEFAULT_ALLOWED, PATH, Broker, BrokerError, serve, write_env

SECRET = "s" * 40


class FakeAz:
    def __init__(self, ttl=3600, now=lambda: 1_000_000):
        self.calls, self.ttl, self.now = [], ttl, now

    def __call__(self, resource):
        self.calls.append(resource)
        return {"accessToken": f"token-for-{resource}-{len(self.calls)}", "expires_on": int(self.now()) + self.ttl}


def test_a_valid_request_gets_the_az_token_in_the_managed_identity_shape():
    az = FakeAz()
    body = Broker(SECRET, runner=az, clock=lambda: 1_000_000).token(SECRET, "https://ml.azure.com", "2019-08-01")
    assert body == {"access_token": "token-for-https://ml.azure.com-1", "expires_on": "1003600", "resource": "https://ml.azure.com",
                    "token_type": "Bearer"}
    assert az.calls == ["https://ml.azure.com"]


@pytest.mark.parametrize("presented", [None, "", "wrong", "s" * 39, "s" * 41])
def test_a_missing_or_wrong_secret_is_refused_before_az_is_asked(presented):
    az = FakeAz()
    with pytest.raises(BrokerError) as caught:
        Broker(SECRET, runner=az).token(presented, "https://ml.azure.com", "2019-08-01")
    assert caught.value.status == 401 and az.calls == []


@pytest.mark.parametrize("resource", [None, "", "https://management.azure.com", "https://ml.azure.com.evil.example", "http://ml.azure.com",
                                      "https://vault.azure.net", "https://ml.azure.com/../x", "https://ml.azure.com?x=1", "file:///etc/passwd"])
def test_only_allow_listed_resources_are_served(resource):
    az = FakeAz()
    with pytest.raises(BrokerError) as caught:
        Broker(SECRET, runner=az).token(SECRET, resource, "2019-08-01")
    assert caught.value.status == 403 and az.calls == []


def test_an_unknown_api_version_and_a_short_secret_are_refused():
    with pytest.raises(BrokerError) as caught:
        Broker(SECRET, runner=FakeAz()).token(SECRET, "https://ml.azure.com", "1999-01-01")
    assert caught.value.status == 400
    with pytest.raises(ValueError):
        Broker("short")


def test_a_trailing_slash_resource_matches_and_extra_resources_can_be_allowed():
    broker = Broker(SECRET, allowed=(*DEFAULT_ALLOWED, "https://example.azure.com/"), runner=FakeAz())
    assert broker.token(SECRET, "https://ml.azure.com/", "2019-08-01")["resource"] == "https://ml.azure.com/"
    assert broker.token(SECRET, "https://example.azure.com", "2019-08-01")["access_token"]


def test_tokens_are_cached_until_they_are_close_to_expiry():
    now = [1_000_000]
    az = FakeAz(ttl=3600, now=lambda: now[0])
    broker = Broker(SECRET, runner=az, clock=lambda: now[0])
    first = broker.token(SECRET, "https://ml.azure.com", "2019-08-01")["access_token"]
    now[0] += 3000
    assert broker.token(SECRET, "https://ml.azure.com", "2019-08-01")["access_token"] == first and len(az.calls) == 1
    now[0] += 400  # inside the five-minute margin
    assert broker.token(SECRET, "https://ml.azure.com", "2019-08-01")["access_token"] != first and len(az.calls) == 2
    broker.token(SECRET, "https://ai.azure.com", "2019-08-01")
    assert az.calls[-1] == "https://ai.azure.com"  # resources are cached separately


def test_a_bad_az_response_is_a_502_and_a_signed_out_cli_is_a_401():
    for bad in ({}, {"accessToken": 5, "expires_on": 1}, {"accessToken": "t"}):
        with pytest.raises(BrokerError) as caught:
            Broker(SECRET, runner=lambda _r, bad=bad: bad).token(SECRET, "https://ml.azure.com", "2019-08-01")
        assert caught.value.status == 502

    def signed_out(_resource):
        raise BrokerError(401, "The Azure CLI is not signed in on this machine: run `az login` here")

    with pytest.raises(BrokerError) as caught:
        Broker(SECRET, runner=signed_out).token(SECRET, "https://ml.azure.com", "2019-08-01")
    assert caught.value.status == 401


@pytest.fixture
def running():
    az = FakeAz(now=lambda: __import__("time").time())
    server = serve(Broker(SECRET, runner=az), 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.server_address[1], az
    server.shutdown()
    server.server_close()


def get(port, path, headers=None):
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read()), dict(response.headers)
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read()), dict(error.headers)


def test_the_http_server_enforces_the_secret_the_path_and_the_allow_list(running):
    port, az = running
    query = f"{PATH}?resource=https%3A%2F%2Fml.azure.com&api-version=2019-08-01"
    assert get(port, query)[0] == 401
    status, body, headers = get(port, query, {"X-IDENTITY-HEADER": SECRET})
    assert status == 200 and body["token_type"] == "Bearer" and headers["Cache-Control"] == "no-store"
    assert get(port, "/other", {"X-IDENTITY-HEADER": SECRET})[0] == 404
    assert get(port, f"{PATH}?resource=https%3A%2F%2Fvault.azure.net&api-version=2019-08-01", {"X-IDENTITY-HEADER": SECRET})[0] == 403
    assert get(port, PATH, {"X-IDENTITY-HEADER": SECRET})[0] == 400  # no api-version
    assert len(az.calls) == 1


def test_the_real_azure_managed_identity_credential_works_against_the_broker(running, monkeypatch):
    """The container uses exactly this class, so the protocol must be one the SDK accepts."""
    from azure.identity import ManagedIdentityCredential

    port, az = running
    for name in ("MSI_ENDPOINT", "MSI_SECRET", "IDENTITY_SERVER_THUMBPRINT", "IMDS_ENDPOINT", "AZURE_POD_IDENTITY_AUTHORITY_HOST", "AZURE_CLIENT_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("IDENTITY_ENDPOINT", f"http://127.0.0.1:{port}{PATH}")
    monkeypatch.setenv("IDENTITY_HEADER", SECRET)
    credential = ManagedIdentityCredential()
    token = credential.get_token("https://ml.azure.com/.default")
    assert token.token == "token-for-https://ml.azure.com-1" and token.expires_on > 0
    assert credential.get_token("https://ai.azure.com/.default").token == "token-for-https://ai.azure.com-2"
    assert az.calls == ["https://ml.azure.com", "https://ai.azure.com"]


def test_the_broker_refuses_to_leave_loopback():
    with pytest.raises(ValueError, match="loopback"):
        serve(Broker(SECRET, runner=FakeAz()), 0, "0.0.0.0")  # noqa: S104


def test_the_env_file_is_updated_without_touching_other_lines(tmp_path):
    env = tmp_path / ".env"
    env.write_text("# mine\nAZURE_TENANT_ID=abc\nIDENTITY_HEADER=old\n", encoding="utf-8")
    write_env(env, {"IDENTITY_HEADER": "new", "IDENTITY_ENDPOINT": "http://h:1/msi/token"})
    lines = env.read_text(encoding="utf-8").splitlines()
    assert lines[:2] == ["# mine", "AZURE_TENANT_ID=abc"] and "IDENTITY_HEADER=new" in lines and "IDENTITY_HEADER=old" not in lines
    assert "IDENTITY_ENDPOINT=http://h:1/msi/token" in lines
    write_env(tmp_path / "new.env", {"A": "1"})
    assert (tmp_path / "new.env").read_text(encoding="utf-8") == "A=1\n"

def test_write_env_uses_line_feeds_so_a_linux_shell_reads_clean_values(tmp_path):
    path = tmp_path / "x.env"
    write_env(path, {"IDENTITY_HEADER": "secret", "A": "1"})
    assert b"\r" not in path.read_bytes()
