# -*- coding: utf-8 -*-
# flake8: noqa: E501
# pylint: disable=line-too-long
"""Tavily search backend (keyless default, optional authenticated)."""

from __future__ import annotations

import logging
import threading

import httpx

from ....config.context import get_current_workspace_dir
from ....drivers.credentials.store import AsyncCredentialStore
from ....drivers.errors import CredentialNotFoundError
from .base import SearchProvider, _post

logger = logging.getLogger(__name__)

_CREDENTIAL_REF = "tool/web_search/tavily"
_FALLBACK_STATUS_CODES = frozenset({401, 429, 432})

# mtime-keyed cache for the per-agent Tavily key: the credential YAML is
# re-read (and re-decrypted) only when the file changes, mirroring the
# AnySearch key cache.
_key_cache: dict[str, tuple[float, str]] = {}
_key_cache_lock = threading.RLock()


async def _current_agent_tavily_key() -> str:
    """Read the optional Tavily API key from the current agent's
    credential store (``tool/web_search/tavily``).

    Empty string means "no key configured" — the provider then stays on
    the keyless default path.
    """
    workspace_dir = get_current_workspace_dir()
    if not workspace_dir:
        return ""
    path = workspace_dir / "credentials.yaml"
    try:
        current_mtime = path.stat().st_mtime
    except OSError:
        current_mtime = 0.0
    cache_key = str(path)
    with _key_cache_lock:
        cached = _key_cache.get(cache_key)
        if cached is not None and cached[0] == current_mtime:
            return cached[1]
    store = AsyncCredentialStore(path)
    try:
        record = await store.get(_CREDENTIAL_REF)
        key = str(record.secrets.get("api_key") or "")
    except CredentialNotFoundError:
        key = ""
    with _key_cache_lock:
        _key_cache[cache_key] = (current_mtime, key)
    return key


class TavilyProvider(SearchProvider):
    """Tavily search backend.

    Default path is keyless anonymous REST (``X-Tavily-Access-Mode:
    keyless``). When the agent has a Tavily API key in the credential
    store, authenticated Bearer requests are preferred; on HTTP 401 /
    429 / 432 the provider logs a warning and falls back to keyless.
    """

    name = "tavily"

    _SEARCH_URL = "https://api.tavily.com/search"

    async def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[dict]:
        payload = {
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
        }
        api_key = await _current_agent_tavily_key()
        if api_key:
            try:
                data = await _post(
                    self._SEARCH_URL,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}",
                    },
                    payload=payload,
                )
                return list(data.get("results") or [])
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status not in _FALLBACK_STATUS_CODES:
                    raise
                logger.warning(
                    "Tavily authenticated search failed with HTTP %s; "
                    "falling back to keyless mode",
                    status,
                )
        data = await _post(
            self._SEARCH_URL,
            headers={
                "Content-Type": "application/json",
                "X-Tavily-Access-Mode": "keyless",
            },
            payload=payload,
        )
        return list(data.get("results") or [])


__all__ = ["TavilyProvider", "_current_agent_tavily_key"]
