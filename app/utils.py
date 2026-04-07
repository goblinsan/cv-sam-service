"""Shared utility helpers for cv-sam-service."""

from typing import Optional
from urllib.parse import urlparse

import requests
from fastapi import HTTPException

# Maximum image payload accepted via URL fetch (10 MB).
MAX_IMAGE_BYTES: int = 10 * 1024 * 1024

_ALLOWED_SCHEMES = {"http", "https"}
_FETCH_TIMEOUT_S = 10


def fetch_image_bytes(image_url: str) -> bytes:
    """Fetch raw image bytes from *image_url*.

    Validates the URL scheme (http/https only), enforces a 10 MB size cap,
    and raises :class:`fastapi.HTTPException` (422) on any failure so callers
    don't need extra error-handling logic.
    """
    parsed = urlparse(image_url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"image_url scheme must be 'http' or 'https', got: {parsed.scheme!r}"
            ),
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
