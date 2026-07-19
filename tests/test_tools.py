"""Tests for tools.py.

All network access (Wikipedia, DuckDuckGo, HTTP fetches) is mocked, so these
tests are fast, deterministic, and safe to run offline / in CI.
"""
import pytest

import tools


# --------------------------------------------------------------------------- #
# multiply (pure)
# --------------------------------------------------------------------------- #
def test_multiply_via_invoke():
    assert tools.multiply.invoke({"a": 6, "b": 7}) == 42.0


def test_multiply_handles_floats_and_negatives():
    assert tools.multiply.func(-2.5, 4) == -10.0


# --------------------------------------------------------------------------- #
# _is_safe_url (SSRF guard)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://localhost:8080/admin",
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata
        "file:///etc/passwd",
        "ftp://example.com/resource",
        "http:///no-host",
    ],
)
def test_is_safe_url_blocks_dangerous_targets(url):
    safe, reason = tools._is_safe_url(url)
    assert safe is False
    assert reason  # a human-readable reason is always given


def test_is_safe_url_allows_public_host(monkeypatch):
    # Pretend the DNS lookup returns a routable public address.
    monkeypatch.setattr(
        tools.socket,
        "getaddrinfo",
        lambda host, port: [(2, 1, 6, "", ("93.184.216.34", 0))],
    )
    safe, reason = tools._is_safe_url("https://example.com/page")
    assert safe is True
    assert reason == ""


def test_is_safe_url_blocks_public_name_resolving_to_private(monkeypatch):
    # DNS rebinding style: public-looking name resolves to a private IP.
    monkeypatch.setattr(
        tools.socket,
        "getaddrinfo",
        lambda host, port: [(2, 1, 6, "", ("10.0.0.5", 0))],
    )
    safe, _ = tools._is_safe_url("https://sneaky.example/")
    assert safe is False


def test_is_safe_url_handles_unresolvable_host(monkeypatch):
    def _boom(host, port):
        raise tools.socket.gaierror("nope")

    monkeypatch.setattr(tools.socket, "getaddrinfo", _boom)
    safe, reason = tools._is_safe_url("https://does-not-exist.invalid/")
    assert safe is False
    assert "resolve" in reason


# --------------------------------------------------------------------------- #
# wikipedia_search
# --------------------------------------------------------------------------- #
def test_wikipedia_search_success(monkeypatch):
    monkeypatch.setattr(tools.wikipedia, "summary", lambda q, sentences, auto_suggest: "Python is a language.")
    result = tools.wikipedia_search.func("Python", sentences=1)
    assert result == "Python is a language."


def test_wikipedia_search_handles_error(monkeypatch):
    def _boom(*args, **kwargs):
        raise ValueError("disambiguation")

    monkeypatch.setattr(tools.wikipedia, "summary", _boom)
    result = tools.wikipedia_search.func("Ambiguous")
    assert result.startswith("Wikipedia search failed:")


# --------------------------------------------------------------------------- #
# duckduckgo_search
# --------------------------------------------------------------------------- #
class _FakeDDGS:
    """Context-manager stand-in for the ddgs.DDGS client."""

    def __init__(self, results):
        self._results = results

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def text(self, query, max_results):
        return self._results


def test_duckduckgo_search_maps_fields(monkeypatch):
    fake = _FakeDDGS([
        {"title": "T1", "href": "https://a.com", "body": "snippet one"},
        {"title": "T2", "href": "https://b.com", "body": "snippet two"},
    ])
    monkeypatch.setattr(tools, "DDGS", lambda: fake)

    results = tools.duckduckgo_search.func("query", max_results=2)
    assert results == [
        {"title": "T1", "url": "https://a.com", "snippet": "snippet one"},
        {"title": "T2", "url": "https://b.com", "snippet": "snippet two"},
    ]


def test_duckduckgo_search_handles_error(monkeypatch):
    def _boom():
        raise RuntimeError("network down")

    monkeypatch.setattr(tools, "DDGS", _boom)
    results = tools.duckduckgo_search.func("query")
    assert results[0]["title"] == "Search error"
    assert "network down" in results[0]["snippet"]


# --------------------------------------------------------------------------- #
# fetch_webpage (mocked HTTP)
# --------------------------------------------------------------------------- #
class _FakeResponse:
    def __init__(self, text="", is_redirect=False, location=None):
        self.text = text
        self.is_redirect = is_redirect
        self.headers = {"location": location} if location else {}

    def raise_for_status(self):
        return None


class _FakeClient:
    """Queue of responses returned in order by successive .get() calls."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.get_calls = []

    def __call__(self, *args, **kwargs):  # allow use as httpx.Client(...)
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url, headers=None):
        self.get_calls.append(url)
        return self._responses.pop(0)


def test_fetch_webpage_blocks_ssrf_without_network(monkeypatch):
    # The real guard should reject the URL before any request is *sent*.
    # (A client may be constructed, but .get must never be called.)
    class _NoGetClient(_FakeClient):
        def get(self, url, headers=None):
            raise AssertionError("no HTTP request should be sent for a blocked URL")

    client = _NoGetClient([])
    monkeypatch.setattr(tools.httpx, "Client", client)
    result = tools.fetch_webpage.func("http://169.254.169.254/latest/meta-data/")
    assert result.startswith("Refused to fetch page:")


def test_fetch_webpage_strips_html_and_trims(monkeypatch):
    monkeypatch.setattr(tools, "_is_safe_url", lambda url: (True, ""))
    html = "<html><script>evil()</script><style>x</style><p>Hello World</p></html>"
    client = _FakeClient([_FakeResponse(text=html)])
    monkeypatch.setattr(tools.httpx, "Client", client)

    result = tools.fetch_webpage.func("https://example.com", max_chars=100)
    assert "evil()" not in result and "Hello World" in result


def test_fetch_webpage_respects_max_chars(monkeypatch):
    monkeypatch.setattr(tools, "_is_safe_url", lambda url: (True, ""))
    client = _FakeClient([_FakeResponse(text="<p>" + "A" * 500 + "</p>")])
    monkeypatch.setattr(tools.httpx, "Client", client)

    result = tools.fetch_webpage.func("https://example.com", max_chars=50)
    assert len(result) == 50


def test_fetch_webpage_revalidates_redirect_target(monkeypatch):
    # First hop is a public URL that redirects to an internal address; the
    # second-hop revalidation must block it.
    def guard(url):
        if "internal" in url:
            return (False, "host resolves to a disallowed address")
        return (True, "")

    monkeypatch.setattr(tools, "_is_safe_url", guard)
    client = _FakeClient([
        _FakeResponse(is_redirect=True, location="http://internal.service/secret"),
    ])
    monkeypatch.setattr(tools.httpx, "Client", client)

    result = tools.fetch_webpage.func("https://public.example/start")
    assert result.startswith("Refused to fetch page:")
    assert client.get_calls == ["https://public.example/start"]  # stopped after 1 hop


def test_fetch_webpage_caps_redirect_chain(monkeypatch):
    monkeypatch.setattr(tools, "_is_safe_url", lambda url: (True, ""))
    # Always redirect -> should hit the max-redirects guard.
    endless = [_FakeResponse(is_redirect=True, location="https://example.com/next") for _ in range(20)]
    client = _FakeClient(endless)
    monkeypatch.setattr(tools.httpx, "Client", client)

    result = tools.fetch_webpage.func("https://example.com/start")
    assert "too many redirects" in result


# --------------------------------------------------------------------------- #
# TOOLS registry
# --------------------------------------------------------------------------- #
def test_tools_registry_exposes_all_tools():
    names = {t.name for t in tools.TOOLS}
    assert names == {"wikipedia_search", "multiply", "duckduckgo_search", "fetch_webpage"}
