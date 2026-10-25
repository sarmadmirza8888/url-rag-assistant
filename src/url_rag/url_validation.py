"""URL validation: only safe, public http(s) URLs are accepted (SSRF protection)."""

from urllib.parse import urlsplit, urlunsplit
import ipaddress
import socket


def validate_url(raw_url: str) -> str:
    """
    Validate and normalize a user-supplied HTTP(S) URL.

    Returns:
        A normalized URL string.

    Raises:
        ValueError: If the URL is invalid or unsafe.
    """
    if not isinstance(raw_url, str):
        raise ValueError("URL must be a string.")

    raw_url = raw_url.strip()

    if not raw_url:
        raise ValueError("Please enter a URL.")

    if any(ord(char) < 32 or ord(char) == 127 for char in raw_url):
        raise ValueError("URL contains control characters.")

    try:
        parsed = urlsplit(raw_url)
        scheme = parsed.scheme.lower()

        if scheme not in ("http", "https"):
            raise ValueError("Only HTTP and HTTPS URLs are allowed.")

        if not parsed.netloc or not parsed.hostname:
            raise ValueError("URL must contain a hostname.")

        if parsed.username is not None or parsed.password is not None:
            raise ValueError("URLs containing credentials are not allowed.")

        hostname = parsed.hostname.rstrip(".").lower()

        if not hostname:
            raise ValueError("Hostname cannot be empty.")

        # Restrict hostnames to valid IP literals or DNS names.
        try:
            ip = ipaddress.ip_address(hostname)
        except ValueError:
            ip = None

        if ip is not None:
            if not ip.is_global:
                raise ValueError("Private or non-public IP addresses are blocked.")
            normalized_host = ip.compressed
        else:
            # Convert international domain names to ASCII.
            normalized_host = hostname.encode("idna").decode("ascii")

            if normalized_host == "localhost" or normalized_host.endswith(".localhost"):
                raise ValueError("Localhost is blocked.")

            if "." not in normalized_host:
                raise ValueError("A fully qualified public hostname is required.")

        try:
            port = parsed.port
        except ValueError as exc:
            raise ValueError("Invalid port number.") from exc

        # Permit only standard web ports.
        expected_port = 443 if scheme == "https" else 80

        if port is not None and port != expected_port:
            raise ValueError("Only ports 80 (HTTP) and 443 (HTTPS) are allowed.")

        # Restore brackets for IPv6 URL syntax.
        if isinstance(ip, ipaddress.IPv6Address) if ip is not None else False:
            netloc = f"[{normalized_host}]"
        else:
            netloc = normalized_host

        # Remove fragments because they are not sent in HTTP requests.
        clean_url = urlunsplit((
            scheme,
            netloc,
            parsed.path or "/",
            parsed.query,
            ""
        ))

        return clean_url

    except UnicodeError as exc:
        raise ValueError("Invalid hostname encoding.") from exc
