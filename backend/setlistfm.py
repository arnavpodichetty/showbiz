"""Small setlist.fm API client with rate limiting and retries.

Note: this key gets 403 on /artist/{mbid}/setlists, so history is pulled
through /search/setlists?artistMbid=..., which returns the same data.
"""
from __future__ import annotations

import os
import threading
import time

import requests
from dotenv import load_dotenv

load_dotenv()

BASE = "https://api.setlist.fm/rest/1.0"
MIN_INTERVAL = 0.6  # seconds between calls; the standard key allows ~2/sec

_lock = threading.Lock()
_last_call = 0.0


class SetlistFMError(RuntimeError):
    pass


def _get(path: str, params: dict) -> dict:
    global _last_call
    key = os.environ.get("SETLISTFM_API_KEY")
    if not key:
        raise SetlistFMError("SETLISTFM_API_KEY is not set (put it in .env)")
    headers = {"x-api-key": key, "Accept": "application/json", "User-Agent": "showbiz/0.1"}
    for attempt in range(5):
        with _lock:
            wait = MIN_INTERVAL - (time.time() - _last_call)
            if wait > 0:
                time.sleep(wait)
            _last_call = time.time()
        r = requests.get(f"{BASE}{path}", params=params, headers=headers, timeout=20)
        if r.status_code == 404:
            return {}
        # setlist.fm sometimes answers 403/429 when calls come too quickly; back off and retry.
        if r.status_code in (403, 429, 500, 502, 503):
            time.sleep(1.5 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()
    raise SetlistFMError(f"setlist.fm kept refusing {path} (last status {r.status_code})")


def search_artists(name: str) -> list[dict]:
    data = _get("/search/artists", {"artistName": name, "sort": "relevance", "p": 1})
    return [
        {"mbid": a["mbid"], "name": a["name"], "disambiguation": a.get("disambiguation", "")}
        for a in data.get("artist", [])[:10]
    ]


def setlists_page(artist_mbid: str, page: int) -> dict:
    return _get("/search/setlists", {"artistMbid": artist_mbid, "p": page})
