"""Safe page fetching and readable-text extraction.

Security: every hostname (and every redirect target) is resolved and rejected if it
points to a private / non-public address. Downloads are size-limited and HTML-only.

User-Agent: sites such as Wikipedia return HTTP 403 for generic bot User-Agents.
We first send a descriptive UA with contact info (set CONTACT_EMAIL / PROJECT_URL),
then retry once with a browser-style UA.
"""
import ipaddress
import os
import socket
from urllib.parse import urljoin, urlsplit

import httpx
import trafilatura

from .url_validation import validate_url


def check_public_dns(url: str) -> None:
    """
    Reject a hostname if DNS resolves to any non-public IP address.
    Must be called before each HTTP request, including redirects.
    """
    parsed = urlsplit(url)
    hostname = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    try:
        # Literal IP addresses do not require DNS resolution.
        try:
            ip = ipaddress.ip_address(hostname)
            addresses = [ip]
        except ValueError:
            results = socket.getaddrinfo(
                hostname,
                port,
                type=socket.SOCK_STREAM
            )
            addresses = [
                ipaddress.ip_address(result[4][0])
                for result in results
            ]

    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Could not safely resolve hostname: {hostname}"
        ) from exc

    if not addresses:
        raise ValueError("Hostname did not resolve to an IP address.")

    # Reject the whole hostname if even one result is non-public.
    if any(not address.is_global for address in addresses):
        raise ValueError(
            f"Hostname resolves to a private or non-public IP: {hostname}"
        )


POLITE_UA = (
    "URLReaderRAG/1.0 "
    f"({os.getenv('PROJECT_URL', 'https://github.com/your-username/url-rag-assistant')}; "
    f"{os.getenv('CONTACT_EMAIL', 'contact-not-set@example.com')}) httpx"
)
BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def fetch_and_extract_once(
    validated_url: str,
    user_agent: str,
    timeout: float = 15.0,
    
    max_bytes: int = 5_000_000,
    max_redirects: int = 5
) -> dict:
    """
    Download an HTML page safely and extract its readable content.

    Input:
        validated_url: URL returned by validate_url()

    Output:
        {
            "title": str,
            "text": str,
            "url": str
        }

    Raises:
        ValueError: For unsafe URLs or unusable page content.
        RuntimeError: For network, HTTP, or extraction failures.
    """
    # Revalidate even if the caller is expected to validate first.
    current_url = validate_url(validated_url)

    headers = {
          "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9"
    }

    try:
        with httpx.Client(
            timeout=httpx.Timeout(timeout),
            follow_redirects=False,
            trust_env=False,
            headers=headers
        ) as client:

            for redirect_count in range(max_redirects + 1):
                # Check every destination, including redirect targets.
                check_public_dns(current_url)

                with client.stream("GET", current_url) as response:
                    if response.status_code in (301, 302, 303, 307, 308):
                        location = response.headers.get("location")

                        if not location:
                            raise RuntimeError(
                                "Redirect response has no Location header."
                            )

                        if redirect_count >= max_redirects:
                            raise RuntimeError("Too many redirects.")

                        # A relative redirect must be resolved against
                        # the current page before validating it.
                        next_url = urljoin(current_url, location)
                        current_url = validate_url(next_url)
                        continue

                    response.raise_for_status()

                    content_type = response.headers.get(
                        "content-type", ""
                    ).lower()

                    if not (
                        "text/html" in content_type
                        or "application/xhtml+xml" in content_type
                    ):
                        raise RuntimeError(
                            f"Expected HTML, received: "
                            f"{content_type or 'unknown content type'}"
                        )

                    content_length = response.headers.get("content-length")

                    if content_length:
                        try:
                            if int(content_length) > max_bytes:
                                raise RuntimeError(
                                    "Page exceeds the download size limit."
                                )
                        except ValueError:
                            raise RuntimeError(
                                "Invalid Content-Length header."
                            )

                    # Enforce a size limit even if Content-Length is absent.
                    chunks = []
                    total_bytes = 0

                    for chunk in response.iter_bytes():
                        total_bytes += len(chunk)

                        if total_bytes > max_bytes:
                            raise RuntimeError(
                                "Page exceeds the download size limit."
                            )

                        chunks.append(chunk)

                    html_bytes = b"".join(chunks)
                    encoding = response.encoding or "utf-8"
                    html = html_bytes.decode(encoding, errors="replace")

                    break
            else:
                raise RuntimeError("Unable to fetch the page.")

    except httpx.TimeoutException as exc:
        raise RuntimeError("Request timed out.") from exc

    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code

        if status == 404:
            raise RuntimeError("Page not found (404).") from exc

        raise RuntimeError(
            f"HTTP request failed with status {status}."
        ) from exc

    except httpx.RequestError as exc:
        raise RuntimeError(
            f"Network request failed: {type(exc).__name__}."
        ) from exc

    # Extract the main readable content, not the raw HTML.
    metadata = trafilatura.extract_metadata(html)

    title = (
        metadata.title.strip()
        if metadata and metadata.title
        else "Untitled page"
    )

    text = trafilatura.extract(
        html,
        output_format="txt",
        include_comments=False,
        include_tables=True,
        favor_precision=True
    )

    if not text or not text.strip():
        raise RuntimeError(
            "No readable text was extracted. "
            "The page may rely on JavaScript or contain little content."
        )

    return {
        "title": title,
        "text": text.strip(),
        "url": current_url
    }


def fetch_and_extract(validated_url: str, **kwargs) -> dict:
    last_error = None

    for ua in (POLITE_UA, BROWSER_UA):
        try:
            return fetch_and_extract_once(
                validated_url,
                user_agent=ua,
                **kwargs
            )
        except RuntimeError as exc:
            if "status 403" in str(exc):
                last_error = exc
                continue
            raise

    raise RuntimeError(
        "The website refused the request (HTTP 403) even after retrying "
        "with a browser-style User-Agent."
    ) from last_error
