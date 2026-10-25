import pytest

from url_rag.url_validation import validate_url


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com",
        "https://en.wikipedia.org/wiki/University_of_California",
        "http://example.com/path?q=1",
    ],
)
def test_accepts_public_urls(url):
    assert validate_url(url).startswith(("http://", "https://"))


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost",
        "http://127.0.0.1",
        "http://192.168.1.1",
        "http://[::1]/",
        "http://10.0.0.5",
        "file:///etc/passwd",
        "ftp://example.com",
        "http://example.com:8080",
        "https://user:pass@example.com",
        "http://intranet",          # no dot -> not a public hostname
        "",
        "   ",
        "https://exa\nmple.com",    # control character
    ],
)
def test_rejects_unsafe_urls(url):
    with pytest.raises(ValueError):
        validate_url(url)


def test_rejects_non_string():
    with pytest.raises(ValueError):
        validate_url(None)


def test_fragment_removed_and_path_defaults():
    assert validate_url("https://example.com#section") == "https://example.com/"
