"""Small setlist.fm API client with rate limiting and retries.

Note: this key gets 403 on /artist/{mbid}/setlists, so history is pulled
through /search/setlists?artistMbid=..., which returns the same data.
"""
from __future__ import annotations

import hashlib
import os
import threading
import time

import requests
from dotenv import load_dotenv

load_dotenv()

BASE = "https://api.setlist.fm/rest/1.0"
MIN_INTERVAL = 0.6  # seconds between calls per key; a standard key allows ~2/sec

# Calls can use a visitor's own key (sent by their browser) or the server's key from .env.
# Keys are only held in memory while a call runs; the rate limiter tracks a hash of each key.
_locks_guard = threading.Lock()
_locks: dict[str, threading.Lock] = {}
_last_call: dict[str, float] = {}


class SetlistFMError(RuntimeError):
    pass


class MissingKey(SetlistFMError):
    """No key from the visitor and none configured on the server."""


class KeyRejected(SetlistFMError):
    """setlist.fm refused the key."""


class LimitReached(SetlistFMError):
    """The key is fine but has used up setlist.fm's request limit (usually the daily one)."""


def server_has_key() -> bool:
    return bool(os.environ.get("SETLISTFM_API_KEY"))


def _key_id(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _get(path: str, params: dict, key: str | None = None) -> dict:
    key = key or os.environ.get("SETLISTFM_API_KEY")
    if not key:
        raise MissingKey("No setlist.fm API key")
    kid = _key_id(key)
    with _locks_guard:
        lock = _locks.setdefault(kid, threading.Lock())
    headers = {"x-api-key": key, "Accept": "application/json", "User-Agent": "showbiz/0.1"}
    for attempt in range(5):
        with lock:
            wait = MIN_INTERVAL - (time.time() - _last_call.get(kid, 0.0))
            if wait > 0:
                time.sleep(wait)
            _last_call[kid] = time.time()
        r = requests.get(f"{BASE}{path}", params=params, headers=headers, timeout=20)
        if r.status_code == 404:
            return {}
        if r.status_code == 401:
            raise KeyRejected("setlist.fm rejected this API key")
        # setlist.fm sometimes answers 403/429 when calls come too quickly; back off and retry.
        if r.status_code in (403, 429, 500, 502, 503):
            time.sleep(1.5 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()
    if r.status_code == 403:
        raise KeyRejected("setlist.fm kept refusing this API key")
    if r.status_code == 429:
        raise LimitReached("This API key has used up its setlist.fm requests for now")
    raise SetlistFMError(f"setlist.fm kept refusing {path} (last status {r.status_code})")


def check_key(key: str) -> str:
    """'ok', 'limited' (a real key that has used up its requests for now) or 'invalid'.
    One cheap call, plus one retry in case the first refusal was only 'too fast'."""
    status = 0
    for attempt in range(2):
        r = requests.get(f"{BASE}/search/artists", params={"artistName": "muse", "p": 1},
                         headers={"x-api-key": key, "Accept": "application/json", "User-Agent": "showbiz/0.1"},
                         timeout=20)
        status = r.status_code
        if status == 200:
            return "ok"
        if status == 429 and "limit exceeded" in r.text.lower():
            return "limited"          # setlist.fm only says this for a key it recognises
        if status not in (403, 429):
            return "invalid"
        time.sleep(1.5)
    return "limited" if status == 429 else "invalid"


def search_artists(name: str, key: str | None = None) -> list[dict]:
    data = _get("/search/artists", {"artistName": name, "sort": "relevance", "p": 1}, key)
    return [
        {"mbid": a["mbid"], "name": a["name"], "disambiguation": a.get("disambiguation", "")}
        for a in data.get("artist", [])[:10]
    ]


def setlists_page(artist_mbid: str, page: int, key: str | None = None) -> dict:
    return _get("/search/setlists", {"artistMbid": artist_mbid, "p": page}, key)
