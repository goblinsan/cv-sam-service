"""Shared utility helpers for cv-sam-service."""

import ipaddress
import socket
from typing import Optional
from urllib.parse import urlparse

import requests
from fastapi import HTTPException

# Maximum image payload accepted via URL fetch (10 MB).
MAX_IMAGE_BYTES: int = 10 * 1024 * 1024

_ALLOWED_SCHEMES = {"http", "https"}
_FETCH_TIMEOUT_S = 10


def _is_safe_host(hostname: str) -> bool:
    """Return True only when *hostname* resolves exclusively to public IP addresses.

    Blocks loopback, private, link-local, multicast, and reserved ranges to
    mitigate Server-Side Request Forgery (SSRF) attacks.
    """
    try:
        addr_info = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return False

    for info in addr_info:
        raw_ip = info[4][0]
        # Strip IPv6 zone ID if present (e.g. "fe80::1%eth0")
        raw_ip = raw_ip.split("%")[0]
        try:
            ip = ipaddress.ip_address(raw_ip)
        except ValueError:
            return False
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            return False

    return True


def fetch_image_bytes(image_url: str) -> bytes:
    """Fetch raw image bytes from *image_url*.

    Validates the URL scheme (http/https only), blocks requests to private /
    internal hosts (SSRF mitigation), enforces a 10 MB size cap, and raises
    :class:`fastapi.HTTPException` (422) on any failure so callers don't need
    extra error-handling logic.
    """
    parsed = urlparse(image_url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"image_url scheme must be 'http' or 'https', got: {parsed.scheme!r}"
            ),
        )

    hostname = parsed.hostname or ""
    if not hostname or not _is_safe_host(hostname):
        raise HTTPException(
            status_code=422,
            detail="image_url must point to a publicly reachable host",
        )

    try:
        response = requests.get(image_url, timeout=_FETCH_TIMEOUT_S, stream=True)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Failed to fetch image_url: {exc}",
        ) from exc

    # Guard against a lying Content-Length header by also checking while reading.
    content_length = response.headers.get("content-length")
    if content_length is not None and int(content_length) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=422,
            detail=f"Image at image_url exceeds the {MAX_IMAGE_BYTES // (1024 * 1024)} MB limit",
        )

    data = b""
    for chunk in response.iter_content(chunk_size=65536):
        data += chunk
        if len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(
                status_code=422,
                detail=f"Image at image_url exceeds the {MAX_IMAGE_BYTES // (1024 * 1024)} MB limit",
            )

    return data


def resolve_image_bytes(
    image_data: Optional[bytes],
    image_url: Optional[str],
) -> bytes:
    """Return image bytes from either a direct upload or a remote URL.

    Exactly one of *image_data* or *image_url* must be supplied.
    Raises :class:`fastapi.HTTPException` (422) when neither is provided.
    """
    if image_data is not None:
        return image_data
    if image_url is not None:
        return fetch_image_bytes(image_url)
    raise HTTPException(
        status_code=422,
        detail="Provide either an 'image' file upload or an 'image_url' query/form field",
    )
