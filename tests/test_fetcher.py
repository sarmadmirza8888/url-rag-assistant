"""Fetcher tests against a local mock server that imitates Wikipedia's User-Agent policy.

The real validate_url / check_public_dns would (correctly) block 127.0.0.1, so they are
patched out here ONLY for the test. Their own behaviour is covered in test_url_validation.
"""
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from url_rag import fetcher

HTML = (
    "<html><head><title>University of California</title></head><body><article>"
    "<h1>University of California</h1><p>"
    + "The University of California is a public research university system. " * 10
    + "</p></article></body></html>"
)
POLICY = {"accept": "contact"}   # contact | browser | never


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        ua = self.headers.get("User-Agent", "")
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/page")
            self.end_headers()
            return
        if self.path == "/missing":
            self.send_response(404)
            self.end_headers()
            return
        if self.path == "/json":
            body = b"{}"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if POLICY["accept"] == "contact":
            ok = "@" in ua or "http" in ua.lower()
        elif POLICY["accept"] == "browser":
            ok = ua.startswith("Mozilla/5.0") and "Chrome" in ua
        else:
            ok = False
        if not ok:
            self.send_response(403)
            self.end_headers()
            return

        body = HTML.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture(scope="module")
def base_url():
    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


@pytest.fixture(autouse=True)
def allow_localhost(monkeypatch):
    monkeypatch.setattr(fetcher, "validate_url", lambda u: u)
    monkeypatch.setattr(fetcher, "check_public_dns", lambda u: None)


def test_polite_user_agent_accepted(base_url):
    POLICY["accept"] = "contact"
    page = fetcher.fetch_and_extract(base_url + "/wiki")
    assert page["title"] == "University of California"
    assert "public research university" in page["text"]


def test_falls_back_to_browser_user_agent_on_403(base_url):
    POLICY["accept"] = "browser"          # the exact Wikipedia-403 scenario
    page = fetcher.fetch_and_extract(base_url + "/wiki")
    assert page["title"] == "University of California"


def test_clear_error_when_site_always_refuses(base_url):
    POLICY["accept"] = "never"
    with pytest.raises(RuntimeError, match="refused the request"):
        fetcher.fetch_and_extract(base_url + "/wiki")


def test_404_is_reported(base_url):
    with pytest.raises(RuntimeError, match="not found"):
        fetcher.fetch_and_extract(base_url + "/missing")


def test_non_html_is_rejected(base_url):
    with pytest.raises(RuntimeError, match="Expected HTML"):
        fetcher.fetch_and_extract(base_url + "/json")


def test_redirects_are_followed(base_url):
    POLICY["accept"] = "contact"
    page = fetcher.fetch_and_extract(base_url + "/redirect")
    assert page["url"].endswith("/page")


def test_real_dns_check_blocks_private_addresses(monkeypatch):
    monkeypatch.undo()   # restore the real check_public_dns
    with pytest.raises(ValueError):
        fetcher.check_public_dns("http://127.0.0.1/")
    with pytest.raises(ValueError):
        fetcher.check_public_dns("http://192.168.0.10/")
