"""FastAPI app. Run with:  .venv/bin/uvicorn backend.app:app --reload"""
from __future__ import annotations

import threading
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import setlistfm, store
from .features import History, Target, parse_all, parse_setlist
import pandas as pd

from .model import DEFAULT_MODEL, MODELS, ArtistModel, score_prediction, train_global

MAX_PAGES = 50  # 20 setlists per page -> up to the 1,000 most recent shows

app = FastAPI(title="Showbiz")
store.init()

_jobs: dict[str, dict] = {}            # artist mbid -> sync status
# Each model type is shared by every artist. It's retrained (only when someone asks for that
# type) after any artist's setlists change.
_histories: dict[str, tuple] = {}   # mbid -> (n setlists, History, training table)
_shared: dict = {k: {"key": None, "clf": None, "backtests": {}} for k in MODELS}
_model_lock = threading.Lock()


# ---------------- syncing ----------------

def _sync(mbid: str, max_pages: int = MAX_PAGES) -> None:
    job = _jobs[mbid]
    try:
        known = store.known_setlist_ids(mbid)
        first = setlistfm.setlists_page(mbid, 1)
        total = int(first.get("total", 0))
        pages = min(max_pages, -(-total // 20)) if total else 0
        job.update(total=total, pages=pages)
        page, data = 1, first
        while data and data.get("setlist"):
            batch = data["setlist"]
            new = store.save_setlists(mbid, batch)
            job.update(page=page, fetched=job.get("fetched", 0) + len(batch))
            # incremental: once a whole page is already known, older pages are too
            if known and new == 0 and page > 2:
                break
            page += 1
            if page > pages:
                break
            data = setlistfm.setlists_page(mbid, page)
        store.upsert_artist(mbid, job["name"], job.get("disambiguation", ""), datetime.utcnow().isoformat())
        job["status"] = "training"
        get_model(mbid)
        job["status"] = "done"
    except Exception as e:  # surface errors to the UI
        job.update(status="error", error=str(e))


def start_sync(mbid: str, name: str, disambiguation: str = "") -> dict:
    job = _jobs.get(mbid)
    if job and job["status"] in ("syncing", "training"):
        return job
    store.upsert_artist(mbid, name, disambiguation)
    _jobs[mbid] = {"status": "syncing", "name": name, "disambiguation": disambiguation, "page": 0, "fetched": 0}
    threading.Thread(target=_sync, args=(mbid,), daemon=True).start()
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


def _history(mbid: str, n: int) -> tuple:
    cached = _histories.get(mbid)
    if cached and cached[0] == n:
        return cached
    played, _ = parse_all(store.load_setlists(mbid))
    h = History(played)
    table = h.training_table() if len(played) > 15 else pd.DataFrame()
    _histories[mbid] = (n, h, table)
    return _histories[mbid]


def get_model(mbid: str, kind: str = DEFAULT_MODEL) -> ArtistModel:
    if kind not in MODELS:
        raise ValueError(f"Unknown model {kind!r}")
    with _model_lock:
        counts = store.setlist_counts()
        if mbid not in counts:
            raise ValueError("Need at least 10 setlists, found 0")
        for m, n in counts.items():
            _history(m, n)
        key = tuple(sorted(counts.items()))
        sh = _shared[kind]
        if sh["key"] != key:
            hists = {m: _histories[m][1] for m in counts}
            tables = {m: _histories[m][2] for m in counts}
            sh["clf"], sh["backtests"] = train_global(hists, tables, kind)
            sh["key"] = key
        return ArtistModel(_histories[mbid][1], sh["clf"], sh["backtests"].get(mbid))


# ---------------- API ----------------

@app.get("/api/artists/search")
def api_search(q: str):
    try:
        return setlistfm.search_artists(q)
    except setlistfm.SetlistFMError as e:
        raise HTTPException(502, str(e))


@app.get("/api/artists")
def api_artists():
    return store.list_artists()


class SyncReq(BaseModel):
    mbid: str
    name: str
    disambiguation: str = ""


@app.post("/api/sync")
def api_sync(req: SyncReq):
    return start_sync(req.mbid, req.name, req.disambiguation)


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
def api_overview(mbid: str, model: str = DEFAULT_MODEL):
    a = store.get_artist(mbid)
    if not a:
        raise HTTPException(404, "Unknown artist; sync it first")
    try:
        m = get_model(mbid, model)
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
    model: str = DEFAULT_MODEL   # "accurate" (CatBoost) or "fast" (LightGBM)


@app.post("/api/predict")
def api_predict(req: PredictReq):
    a = store.get_artist(req.mbid)
    if not a:
        raise HTTPException(404, "Unknown artist; sync it first")
    if req.log and not req.date:
        raise HTTPException(400, "Pick a date to save a prediction, so we know which show to grade it against.")
    if req.model not in MODELS:
        raise HTTPException(400, "Choose either the more accurate or the faster model.")
    m = get_model(req.mbid, req.model)
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
    out["model"] = req.model
    out["target"]["tour_mode"] = "none" if req.tour == NO_TOUR else "new" if req.tour == NEW_TOUR else "tour"
    if req.log:
        out["prediction_id"] = store.add_prediction({
            "artist_mbid": req.mbid, "artist_name": a["name"], "created_at": datetime.utcnow().isoformat(),
            "show_date": req.date, "city": t.city, "country": t.country, "is_festival": t.is_festival,
            "payload": out,
        })
    return out


@app.get("/api/scoreboard")
def api_scoreboard(refresh: bool = False):
    preds = store.list_predictions()
    today = date.today().isoformat()
    due = [p for p in preds if p["result"] is None and p["show_date"] <= today]
    if refresh:
        for mbid in {p["artist_mbid"] for p in due}:
            # pull the newest couple of pages to pick up freshly posted setlists
            for page in (1, 2):
                data = setlistfm.setlists_page(mbid, page)
                if data.get("setlist"):
                    store.save_setlists(mbid, data["setlist"])
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
