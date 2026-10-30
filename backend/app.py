"""FastAPI app. Run with:  .venv/bin/uvicorn backend.app:app --reload"""
from __future__ import annotations

import threading
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import re

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import setlistfm, store
from .features import FEATURES, History, Target, parse_all, parse_setlist
import json

import joblib
import pandas as pd
from catboost import CatBoostClassifier

from .model import CATBOOST_PARAMS, ArtistModel, score_prediction, train_global

MAX_PAGES = 50  # 20 setlists per page -> up to the 1,000 most recent shows
# Predict as soon as this many setlists are stored (and enough are real shows), while the
# older ones keep downloading. setlist.fm sends the newest first, and those matter most.
READY_SETLISTS = 100
READY_SHOWS = 30

app = FastAPI(title="Showbiz")
store.init()

_jobs: dict[str, dict] = {}            # artist mbid -> sync status
# One model is shared by every artist. When an artist's setlists change, predictions
# keep using the current model while a new one trains in the background.
_histories: dict[str, tuple] = {}   # mbid -> (n setlists, History, training table)
_shared: dict = {"key": None, "clf": None, "training": None}
_model_lock = threading.RLock()     # guards _histories and _shared
_train_lock = threading.Lock()      # one retrain at a time

# Prepared training data and trained models are saved here, so a restart doesn't redo them.
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"


# ---------------- syncing ----------------

def _check_ready(mbid: str, job: dict) -> None:
    """Mark the artist ready to predict once there are enough recent setlists."""
    if job.get("ready") or len(store.known_setlist_ids(mbid)) < READY_SETLISTS:
        return
    played, _ = parse_all(store.load_setlists(mbid))
    if len(played) >= READY_SHOWS:
        get_model(mbid)              # prepare their data now, so the first prediction is quick
        job["ready"] = True


def _sync(mbid: str, max_pages: int = MAX_PAGES, key: Optional[str] = None) -> None:
    """Download an artist's setlists. `key` is the visitor's setlist.fm key (or None to use
    the server's). It's passed as an argument and never stored in the job, which visitors see."""
    job = _jobs[mbid]
    try:
        known = store.known_setlist_ids(mbid)
        job["first_time"] = not known     # a brand-new artist (vs. checking for new shows)
        _check_ready(mbid, job)      # artists downloaded before are ready straight away
        first = setlistfm.setlists_page(mbid, 1, key)
        total = int(first.get("total", 0))
        pages = min(max_pages, -(-total // 20)) if total else 0
        job.update(total=total, pages=pages)
        page, data = 1, first
        while data and data.get("setlist"):
            batch = data["setlist"]
            new = store.save_setlists(mbid, batch)
            job.update(page=page, fetched=job.get("fetched", 0) + len(batch), new=job.get("new", 0) + new)
            _check_ready(mbid, job)
            # incremental: once a whole page is already known, older pages are too
            if known and new == 0 and page > 2:
                break
            page += 1
            if page > pages:
                break
            data = setlistfm.setlists_page(mbid, page, key)
        store.upsert_artist(mbid, job["name"], job.get("disambiguation", ""), datetime.utcnow().isoformat())
        job["status"] = "training"
        get_model(mbid)              # prepare this artist's data; retrain in the background
        job["status"] = "done"
    except setlistfm.KeyRejected:
        job.update(status="error", error="key_rejected")
    except setlistfm.LimitReached:
        job.update(status="error", error="limit")
    except Exception as e:  # surface errors to the UI
        job.update(status="error", error=str(e))


def start_sync(mbid: str, name: str, disambiguation: str = "", key: Optional[str] = None) -> dict:
    job = _jobs.get(mbid)
    if job and job["status"] in ("syncing", "training"):
        return job
    if not (key or setlistfm.server_has_key()):
        # no key to download with: fine if the artist is already downloaded (predicting needs no key)
        if store.setlist_counts().get(mbid):
            return {"status": "done", "name": name, "new": 0}
        raise setlistfm.MissingKey("No setlist.fm API key")
    store.upsert_artist(mbid, name, disambiguation)
    _jobs[mbid] = {"status": "syncing", "ready": False, "name": name, "disambiguation": disambiguation,
                   "page": 0, "fetched": 0}
    threading.Thread(target=_sync, args=(mbid, MAX_PAGES, key), daemon=True).start()
    return _jobs[mbid]


def default_tour(hist: History) -> str:
    """Most recent tour name with at least 5 shows (skips one-off promo gigs)."""
    from collections import Counter
    recent = [s.tour for s in hist.shows[-80:]]
    counts = Counter(recent)
    for t in reversed(recent):
        if t and counts[t] >= 5:
            return t
    return hist.shows[-1].tour


def _slim(table: pd.DataFrame) -> pd.DataFrame:
    """Keep only what training needs, in compact types (about 4x less memory)."""
    out = table[FEATURES].astype(np.float32)
    out["label"] = table["label"].astype(np.int8)
    out["show_i"] = table["show_i"].astype(np.int32)
    return out


def _history(mbid: str, n: int) -> tuple:
    cached = _histories.get(mbid)
    if cached and cached[0] == n:
        return cached
    played, _ = parse_all(store.load_setlists(mbid))
    h = History(played)
    path = CACHE_DIR / "tables" / f"{mbid}.pkl"
    table = None
    try:
        saved = joblib.load(path)
        if saved["n"] == n and saved["columns"] == FEATURES:
            table = saved["table"]
    except Exception:
        pass
    if table is None:
        table = _slim(h.training_table()) if len(played) > 15 else pd.DataFrame()
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"n": n, "columns": FEATURES, "table": table}, path)
    _histories[mbid] = (n, h, table)
    return _histories[mbid]


MODEL_PATH = CACHE_DIR / "models" / "catboost.joblib"

# The model that ships with Showbiz (made by backend/export_model.py, committed to git).
# A fresh copy uses it until its own database holds at least as many setlists, so one
# downloaded artist never replaces a model trained on 100.
BUNDLED_DIR = Path(__file__).resolve().parent.parent / "model"
BUNDLED_MODEL = BUNDLED_DIR / "showbiz.cbm"
BUNDLED_META = BUNDLED_DIR / "showbiz.json"
_bundled: dict = {"loaded": False, "clf": None, "n_setlists": 0}


def _load_bundled() -> None:
    if _bundled["loaded"]:
        return
    _bundled["loaded"] = True
    try:
        meta = json.loads(BUNDLED_META.read_text())
        if meta["features"] != FEATURES:
            return                     # made with different features; can't be used
        clf = CatBoostClassifier()
        clf.load_model(str(BUNDLED_MODEL))
        _bundled.update(clf=clf, n_setlists=int(meta["n_setlists"]))
    except Exception:
        pass
# changes whenever the features or the model settings change, so an old saved model is ignored
MODEL_SPEC = repr((FEATURES, CATBOOST_PARAMS))


def _load_saved_model() -> None:
    if _shared["clf"] is not None:
        return
    try:
        saved = joblib.load(MODEL_PATH)
        if saved["spec"] == MODEL_SPEC:
            _shared["clf"], _shared["key"] = saved["clf"], saved["key"]
    except Exception:
        pass


def _retrain(key: tuple, hists: dict, tables: dict) -> None:
    with _train_lock:
        # skip if the model is already current, or a newer request (e.g. more setlists
        # arrived during a download) has replaced this one
        if _shared["key"] == key or _shared["training"] != key:
            return
        try:
            clf, _ = train_global(hists, tables)
        except Exception:
            with _model_lock:
                _shared["training"] = None   # let the next request try again
            raise
        with _model_lock:
            _shared.update(clf=clf, key=key)
            if _shared["training"] == key:
                _shared["training"] = None
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"spec": MODEL_SPEC, "key": key, "clf": clf}, MODEL_PATH)


def get_model(mbid: str) -> ArtistModel:
    """The artist's history plus the shared model.

    If setlists changed since the model was trained, this returns the current model right
    away and retrains in the background. Only the very first training waits.
    """
    with _model_lock:
        counts = store.setlist_counts()
        if mbid not in counts:
            raise ValueError("Need at least 10 setlists, found 0")
        for m, n in counts.items():
            _history(m, n)
        key = tuple(sorted(counts.items()))
        sh = _shared
        _load_saved_model()
        _load_bundled()
        if _bundled["clf"] is not None and sum(counts.values()) < _bundled["n_setlists"]:
            # less local data than the shipped model saw: use the shipped model, don't retrain
            return ArtistModel(_histories[mbid][1], _bundled["clf"], None)
        if sh["key"] != key:
            hists = {m: _histories[m][1] for m in counts}
            tables = {m: _histories[m][2] for m in counts}
            if sh["training"] != key:
                sh["training"] = key
                if sh["clf"] is None:
                    _retrain(key, hists, tables)      # nothing to fall back on yet
                else:
                    threading.Thread(target=_retrain, args=(key, hists, tables), daemon=True).start()
        return ArtistModel(_histories[mbid][1], sh["clf"], None)


# ---------------- API ----------------
# Visitors send two headers:
#   X-Setlistfm-Key: their own setlist.fm API key (optional; saved only in their browser)
#   X-Showbiz-User:  a random id their browser made, so searches and saved predictions stay theirs

NEED_KEY = "Add your setlist.fm API key to look up artists. It's free and takes a minute to get."
BAD_KEY = "setlist.fm didn't accept your API key. Check it and try again."
LIMIT = "Your setlist.fm API key has used up today's requests. It'll work again tomorrow; artists already downloaded still work."


def _user(x_showbiz_user: Optional[str]) -> str:
    u = (x_showbiz_user or "").strip()
    return u if re.fullmatch(r"[A-Za-z0-9-]{8,64}", u) else "anonymous"


def _key(x_setlistfm_key: Optional[str]) -> Optional[str]:
    k = (x_setlistfm_key or "").strip()
    return k if 8 <= len(k) <= 128 else None


@app.get("/api/config")
def api_config():
    """Whether this server has its own setlist.fm key, so visitors don't need one."""
    return {"server_key": setlistfm.server_has_key()}


class KeyReq(BaseModel):
    key: str


@app.post("/api/key/check")
def api_key_check(req: KeyReq):
    k = _key(req.key)
    result = setlistfm.check_key(k) if k else "invalid"
    return {"valid": result != "invalid", "limited": result == "limited"}


@app.get("/api/artists/search")
def api_search(q: str, x_setlistfm_key: Optional[str] = Header(None)):
    try:
        return setlistfm.search_artists(q, _key(x_setlistfm_key))
    except setlistfm.MissingKey:
        raise HTTPException(401, NEED_KEY)
    except setlistfm.KeyRejected:
        raise HTTPException(401, BAD_KEY)
    except setlistfm.LimitReached:
        raise HTTPException(429, LIMIT)
    except setlistfm.SetlistFMError as e:
        raise HTTPException(502, str(e))


@app.get("/api/artists")
def api_artists(x_showbiz_user: Optional[str] = Header(None)):
    """The "Checked before" list: the last 5 artists this visitor picked."""
    return store.recent_searches(_user(x_showbiz_user), 5)


class SyncReq(BaseModel):
    mbid: str
    name: str
    disambiguation: str = ""


@app.post("/api/sync")
def api_sync(req: SyncReq, x_setlistfm_key: Optional[str] = Header(None),
             x_showbiz_user: Optional[str] = Header(None)):
    try:
        job = start_sync(req.mbid, req.name, req.disambiguation, _key(x_setlistfm_key))
    except setlistfm.MissingKey:
        raise HTTPException(401, NEED_KEY)
    store.mark_searched(_user(x_showbiz_user), req.mbid)
    return job


@app.get("/api/sync/{mbid}")
def api_sync_status(mbid: str):
    job = _jobs.get(mbid)
    if job:
        return job
    a = store.get_artist(mbid)
    if a and a["last_synced"]:
        return {"status": "done", "name": a["name"]}
    return {"status": "none"}


@app.get("/api/artist/{mbid}/overview")
def api_overview(mbid: str):
    a = store.get_artist(mbid)
    if not a:
        raise HTTPException(404, "Unknown artist; sync it first")
    try:
        m = get_model(mbid)
    except ValueError as e:
        raise HTTPException(400, str(e))
    played, upcoming = parse_all(store.load_setlists(mbid))
    recent = played[-8:][::-1]
    return {
        "artist": a,
        "n_shows": len(played),
        "first_show": played[0].date.isoformat() if played else None,
        "last_show": played[-1].date.isoformat() if played else None,
        "current_tour": default_tour(m.hist),
        "tours": list(dict.fromkeys(s.tour for s in played[::-1] if s.tour))[:12],
        "backtest": m.backtest,
        "recent": [
            {"date": s.date.isoformat(), "venue": s.venue, "city": s.city, "country": s.country,
             "tour": s.tour, "festival": s.is_festival, "n_songs": len(s.songs), "url": s.url}
            for s in recent
        ],
        "upcoming": [
            {"date": s.date.isoformat(), "venue": s.venue, "city": s.city, "country": s.country,
             "tour": s.tour, "festival": s.is_festival}
            for s in upcoming[:20]
        ],
    }


NEW_TOUR = "__new__"   # a tour with no shows yet
NO_TOUR = "__none__"   # a one-off show that isn't part of any tour


class PredictReq(BaseModel):
    mbid: str
    date: Optional[str] = None   # blank = a typical show right after their latest one
    city: str = ""
    country: str = ""
    tour: Optional[str] = None
    is_festival: bool = False
    log: bool = False


@app.post("/api/predict")
def api_predict(req: PredictReq, x_showbiz_user: Optional[str] = Header(None)):
    a = store.get_artist(req.mbid)
    if not a:
        raise HTTPException(404, "Unknown artist; sync it first")
    if req.log and not req.date:
        raise HTTPException(400, "Pick a date to save a prediction, so we know which show to grade it against.")
    m = get_model(req.mbid)
    last = m.hist.shows[-1].date
    general = not req.date
    d = date.fromordinal(last.toordinal() + 1) if general else date.fromisoformat(req.date)
    if req.tour is None:
        tour = default_tour(m.hist)
    elif req.tour == NO_TOUR:
        tour = ""
    else:
        tour = req.tour  # a known tour name, or NEW_TOUR which matches no past shows
    t = Target(d, req.city.strip(), req.country.strip().upper(), tour, req.is_festival)
    out = m.predict(t)
    out["target"]["general"] = general
    out["target"]["tour_mode"] = "none" if req.tour == NO_TOUR else "new" if req.tour == NEW_TOUR else "tour"
    if req.log:
        out["prediction_id"] = store.add_prediction({
            "artist_mbid": req.mbid, "artist_name": a["name"], "created_at": datetime.utcnow().isoformat(),
            "show_date": req.date, "city": t.city, "country": t.country, "is_festival": t.is_festival,
            "payload": out, "user_id": _user(x_showbiz_user),
        })
    return out


@app.get("/api/scoreboard")
def api_scoreboard(refresh: bool = False, x_setlistfm_key: Optional[str] = Header(None),
                   x_showbiz_user: Optional[str] = Header(None)):
    preds = store.list_predictions(_user(x_showbiz_user))
    today = date.today().isoformat()
    due = [p for p in preds if p["result"] is None and p["show_date"] <= today]
    key = _key(x_setlistfm_key)
    if refresh and (key or setlistfm.server_has_key()):
        try:
            for mbid in {p["artist_mbid"] for p in due}:
                # pull the newest couple of pages to pick up freshly posted setlists
                for page in (1, 2):
                    data = setlistfm.setlists_page(mbid, page, key)
                    if data.get("setlist"):
                        store.save_setlists(mbid, data["setlist"])
        except setlistfm.SetlistFMError:
            pass                                   # grade with what's already downloaded
    by_artist: dict[str, dict] = {}
    for p in due:
        if p["artist_mbid"] not in by_artist:
            shows = [parse_setlist(s) for s in store.load_setlists(p["artist_mbid"])]
            by_artist[p["artist_mbid"]] = {s.date.isoformat(): s for s in shows if len(s.songs) >= 3}
        show = by_artist[p["artist_mbid"]].get(p["show_date"])
        if show:
            result = score_prediction(p["payload"], show.songs)
            result["url"] = show.url
            store.set_prediction_result(p["id"], result)
            p["result"] = result

    scored = [p for p in preds if p["result"]]
    summary = None
    if scored:
        summary = {
            "n": len(scored),
            "model_brier": float(np.mean([p["result"]["model"]["brier"] for p in scored])),
            "baseline_brier": float(np.mean([p["result"]["baseline"]["brier"] for p in scored])),
            "model_hits": float(np.mean([p["result"]["model"]["hits_at_set_size"] for p in scored])),
            "baseline_hits": float(np.mean([p["result"]["baseline"]["hits_at_set_size"] for p in scored])),
        }
    rows = [
        {"id": p["id"], "artist": p["artist_name"], "show_date": p["show_date"], "city": p["city"],
         "country": p["country"], "is_festival": bool(p["is_festival"]), "created_at": p["created_at"],
         "predicted": [s["song"] for s in p["payload"]["setlist"]], "result": p["result"]}
        for p in preds
    ]
    return {"summary": summary, "predictions": rows}


# ---------------- frontend ----------------

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=FRONTEND), name="static")


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")
