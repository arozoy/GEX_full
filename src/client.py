"""HTTP client for the Polygon.io (Massive) REST API.

Handles authentication, pagination (next_url), concurrency, and retries.
All methods return parsed JSON bodies; callers are responsible for shape.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, AsyncIterator
from urllib.parse import urljoin

import httpx
from dotenv import load_dotenv
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import (
    API_BASE_URL,
    HTTP_CONNECT_TIMEOUT_SEC,
    HTTP_MAX_RETRIES,
    HTTP_POOL_TIMEOUT_SEC,
    HTTP_READ_TIMEOUT_SEC,
    HTTP_WRITE_TIMEOUT_SEC,
    MAX_CONCURRENT_REQUESTS,
)

logger = logging.getLogger(__name__)

load_dotenv()


class ApiError(Exception):
    """Non-retryable API error (4xx other than 429)."""


class RateLimitError(Exception):
    """Retryable rate-limit error (HTTP 429)."""


class ServerError(Exception):
    """Retryable server error (HTTP 5xx)."""


def _get_api_key() -> str:
    key = os.environ.get("MASSIVE_API_KEY")
    if not key:
        raise RuntimeError(
            "MASSIVE_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return key


class PolygonClient:
    """Async client with bounded concurrency and pagination helpers.

    Usage:
        async with PolygonClient() as client:
            async for page in client.paginate("/v3/trades/O:GOOGL..."):
                ...
    """

    def __init__(
        self,
        max_concurrent: int = MAX_CONCURRENT_REQUESTS,
    ) -> None:
        self._api_key = _get_api_key()
        self._sem = asyncio.Semaphore(max_concurrent)
        # Generous connection pool so the semaphore (not the pool) is the
        # bottleneck. max_connections caps total sockets; keepalive caps
        # idle-reuse sockets.
        self._client = httpx.AsyncClient(
            base_url=API_BASE_URL,
            timeout=httpx.Timeout(
                connect=HTTP_CONNECT_TIMEOUT_SEC,
                read=HTTP_READ_TIMEOUT_SEC,
                write=HTTP_WRITE_TIMEOUT_SEC,
                pool=HTTP_POOL_TIMEOUT_SEC,
            ),
            limits=httpx.Limits(
                max_keepalive_connections=max_concurrent,
                max_connections=int(max_concurrent * 1.5),
            ),
            http2=False,
        )

    async def __aenter__(self) -> "PolygonClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self._client.aclose()

    async def get(
        self, path_or_url: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """GET one URL (path or absolute) and return parsed JSON.

        Retries on 429 and transient network errors with exponential backoff.
        """
        params = dict(params or {})
        params["apiKey"] = self._api_key

        # If path_or_url is absolute (e.g., a next_url), httpx will use it as-is.
        url = path_or_url

        async for attempt in AsyncRetrying(
            reraise=True,
            stop=stop_after_attempt(HTTP_MAX_RETRIES),
            wait=wait_exponential(multiplier=1, min=1, max=30),
            retry=retry_if_exception_type(
                (
                    RateLimitError,
                    ServerError,
                    httpx.TransportError,
                    httpx.ReadTimeout,
                )
            ),
        ):
            with attempt:
                async with self._sem:
                    response = await self._client.get(url, params=params)
                if response.status_code == 429:
                    raise RateLimitError(f"Rate limited: {response.text[:200]}")
                if 500 <= response.status_code < 600:
                    raise ServerError(
                        f"HTTP {response.status_code} on {url}: {response.text[:200]}"
                    )
                if response.status_code >= 400:
                    raise ApiError(
                        f"HTTP {response.status_code} on {url}: {response.text[:500]}"
                    )
                return response.json()
        raise RuntimeError("unreachable")

    async def paginate(
        self,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield every result row by following `next_url` cursors.

        Works for any Polygon v3 endpoint that returns {results: [...], next_url: ...}.
        """
        next_url: str | None = path
        current_params: dict[str, Any] | None = params

        while next_url:
            payload = await self.get(next_url, current_params)
            for row in payload.get("results") or []:
                yield row

            next_url = payload.get("next_url")
            # next_url is absolute and carries the cursor; drop the original params
            # so we don't double-send things like timestamp filters (the API
            # embeds those into the cursor itself).
            current_params = None
