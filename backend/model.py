"""Train, backtest, and predict.

Two models, both gradient-boosted trees on one row per (song, show), trained on
every downloaded artist at once (pooling beat per-artist models; see PROCESS.md):
  - "accurate": CatBoost, tuned with Optuna. Best accuracy, retrains in ~6 s.
  - "fast":     LightGBM, tuned with Optuna. About 1% less accurate, retrains in ~1.5 s.
Baseline: the song's play rate this tour (or over the last 10 shows if the
tour has fewer than 3 shows so far).
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from .features import FEATURES, History, Target

TEST_FRACTION = 0.15
MIN_SHOWS = 10            # shows needed to predict an artist (pooling lets small artists borrow from others)
MIN_BACKTEST_SHOWS = 30   # shows needed to report a backtest for an artist

# Settings picked by Optuna on a separate tuning block across 17 artists (experiments/round3.py).
MODELS = {
    "accurate": ("catboost", dict(
        depth=9, learning_rate=0.0466, iterations=150, l2_leaf_reg=1.21, random_strength=1.06,
        bagging_temperature=0.039, border_count=254, grow_policy="Depthwise", min_data_in_leaf=36,
        verbose=0, random_seed=0, thread_count=-1, allow_writing_files=False)),
    "fast": ("lightgbm", dict(
        num_leaves=42, min_child_samples=10, n_estimators=150, learning_rate=0.0391, reg_lambda=0.00214,
        subsample=0.887, subsample_freq=1, colsample_bytree=0.695, verbose=-1, random_state=0, n_jobs=-1)),
}
DEFAULT_MODEL = "accurate"


def baseline_probs(df: pd.DataFrame) -> np.ndarray:
    use_tour = df["n_tour_prior"].values >= 3
    p = np.where(use_tour, df["rate_tour"].values, df["rate_last10"].values)
    return np.clip(p, 0.01, 0.99)


def _fit(train: pd.DataFrame, kind: str = DEFAULT_MODEL):
    lib, params = MODELS[kind]
    if lib == "catboost":
        from catboost import CatBoostClassifier
        clf = CatBoostClassifier(**params)
    else:
        import lightgbm as lgb
        clf = lgb.LGBMClassifier(**params)
    clf.fit(train[FEATURES].values.astype(np.float32), train["label"].values)
    return clf


def _proba(clf, df: pd.DataFrame) -> np.ndarray:
    return np.clip(clf.predict_proba(df[FEATURES].values.astype(np.float32))[:, 1], 0.005, 0.995)


def _show_metrics(df: pd.DataFrame, p: np.ndarray, actual_len: dict) -> dict:
    """Ranking metrics averaged per show: precision at the real set size, hits in top 15."""
    prec, hits15 = [], []
    for i, g in df.assign(p=p).groupby("show_i"):
        g = g.sort_values("p", ascending=False)
        k = actual_len[i]
        prec.append(g["label"].values[:k].mean())
        hits15.append(g["label"].values[:15].sum())
    return {"precision_at_set_size": float(np.mean(prec)), "hits_in_top15": float(np.mean(hits15))}


def _calibration(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.sum() >= 10:
            out.append({"predicted": float(p[m].mean()), "observed": float(y[m].mean()), "n": int(m.sum())})
    return out


def _metrics(y, p) -> dict:
    return {"brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p, labels=[0, 1]))}


def train_global(hists: dict, tables: dict, kind: str = DEFAULT_MODEL) -> tuple:
    """Train one model on every artist.

    Backtest: one fit on each artist's first 85% of shows, scored on each artist's
    last 15%, so no artist's test shows are seen in training. Then a final fit on everything.
    Returns (final classifier, {artist id: backtest dict}).
    """
    cuts = {k: int(len(h.shows) * (1 - TEST_FRACTION)) for k, h in hists.items()}
    backtests: dict = {}
    eligible = [k for k, h in hists.items() if len(h.shows) >= MIN_BACKTEST_SHOWS and len(tables[k])]
    if eligible:
        train = pd.concat([tables[k][tables[k].show_i < cuts[k]] for k in hists if len(tables[k])])
        clf = _fit(train, kind)
        for k in eligible:
            h, t = hists[k], tables[k]
            test = t[t.show_i >= cuts[k]]
            if test.empty:
                continue
            y = test["label"].values
            p_model, p_base = _proba(clf, test), baseline_probs(test)
            actual_len = {i: int(h.P[i].sum()) for i in range(len(h.shows))}
            backtests[k] = {
                "n_train_shows": int(t[t.show_i < cuts[k]].show_i.nunique()),
                "n_test_shows": int(test.show_i.nunique()),
                "test_from": h.shows[cuts[k]].date.isoformat(),
                "test_to": h.shows[-1].date.isoformat(),
                "model": {**_metrics(y, p_model), **_show_metrics(test, p_model, actual_len),
                          "calibration": _calibration(y, p_model)},
                "baseline": {**_metrics(y, p_base), **_show_metrics(test, p_base, actual_len),
                             "calibration": _calibration(y, p_base)},
            }
    final = _fit(pd.concat([t for t in tables.values() if len(t)]), kind)
    return final, backtests


class ArtistModel:
    """One artist's history plus the shared model, used for predictions."""

    def __init__(self, hist: History, clf, backtest: dict | None):
        if len(hist.shows) < MIN_SHOWS:
            raise ValueError(f"Need at least {MIN_SHOWS} setlists, found {len(hist.shows)}")
        self.hist = hist
        self.clf = clf
        self.backtest = backtest or {}

    # ---------- prediction ----------

    def predict(self, t: Target) -> dict:
        h = self.hist
        i = int(np.searchsorted(h.dates, t.date.toordinal()))  # only shows before the target date
        df, cand = h.features(i, t)
        p = _proba(self.clf, df)
        base = baseline_probs(df)
        stats = h.song_stats(cand, i)
        df = pd.concat([df.reset_index(drop=True), stats], axis=1)
        df["p"] = p
        df["p_baseline"] = base
        df = df.sort_values("p", ascending=False).reset_index(drop=True)

        set_len = self._expected_length(i, t)
        setlist = self._build_setlist(df, set_len)
        rotations = self._rotation_slots(i, cand, dict(zip(df.song, df.p)), t.tour)
        rare = df[df["rate_last100"] < 0.10]
        rare_p = float(1 - np.prod(1 - rare["p"].values)) if len(rare) else 0.0

        last = h.shows[i - 1]
        return {
            "target": {"date": t.date.isoformat(), "city": t.city, "country": t.country,
                       "tour": t.tour, "is_festival": t.is_festival},
            "based_on": {"n_shows": i, "last_show": {"date": last.date.isoformat(), "venue": last.venue,
                                                     "city": last.city, "url": last.url}},
            "expected_length": set_len,
            "setlist": setlist,
            "songs": [
                {
                    "song": r.song, "p": round(float(r.p), 4), "p_baseline": round(float(r.p_baseline), 4),
                    "rate_tour": round(float(r.rate_tour), 3), "rate_last10": round(float(r.rate_last10), 3),
                    "shows_since": int(round(np.expm1(r.log_shows_since))),
                    "played_in_run": bool(r.played_in_run), "encore_rate": round(float(r.encore_rate), 2),
                }
                for r in df.itertuples()
            ],
            "rotations": rotations,
            "rare_alert": {
                "p_any": round(rare_p, 3),
                "top": [{"song": r.song, "p": round(float(r.p), 3)} for r in rare.head(5).itertuples()],
            },
        }

    def _expected_length(self, i: int, t: Target) -> int:
        h = self.hist
        tour_rows = np.where((h.tours[:i] == t.tour) & (h.fest[:i] == t.is_festival))[0] if t.tour else []
        if len(tour_rows) >= 3:
            return int(np.median(h.set_len[tour_rows[-15:]]))
        rows = np.arange(max(0, i - 40), i)
        same_kind = rows[h.fest[rows] == t.is_festival]
        if len(same_kind) >= 3:
            return int(np.median(h.set_len[same_kind[-10:]]))
        med = float(np.median(h.set_len[rows[-10:]]))
        return int(round(med * (0.7 if t.is_festival else 1.0)))

    @staticmethod
    def _build_setlist(df: pd.DataFrame, n: int) -> list[dict]:
        chosen = df.head(n).copy()
        chosen["avg_pos"] = chosen["avg_pos"].fillna(0.5)
        # encore songs go last, everything else ordered by where it usually sits
        chosen["is_encore"] = chosen["encore_rate"] >= 0.5
        chosen = chosen.sort_values(["is_encore", "avg_pos"])
        return [
            {"song": r.song, "p": round(float(r.p), 3), "encore": bool(r.is_encore),
             "opener_rate": round(float(r.opener_rate), 2), "closer_rate": round(float(r.closer_rate), 2)}
            for r in chosen.itertuples()
        ]

    def _rotation_slots(self, i: int, cand: np.ndarray, probs: dict, tour: str = "",
                        lookback: int = 25) -> list[dict]:
        """Groups of songs that swap with each other: each played fairly often,
        but rarely in the same show. Uses this tour's shows when there are enough."""
        h = self.hist
        tour_rows = np.where(h.tours[:i] == tour)[0] if tour else np.array([], dtype=int)
        rows = tour_rows[-40:] if len(tour_rows) >= 8 else np.arange(max(0, i - lookback), i)
        P = h.P[rows][:, cand]
        if P.shape[0] < 8:
            return []
        rates = P.mean(0)
        mid = np.where((rates >= 0.1) & (rates <= 0.85))[0]
        parent = {a: a for a in mid}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for x in range(len(mid)):
            for y in range(x + 1, len(mid)):
                a, b = mid[x], mid[y]
                together = (P[:, a] * P[:, b]).mean()
                expected = rates[a] * rates[b]
                covered = ((P[:, a] + P[:, b]) > 0).mean()
                if together <= 0.4 * expected and covered >= 0.6:
                    parent[find(a)] = find(b)
        groups: dict = {}
        for a in mid:
            groups.setdefault(find(a), []).append(a)
        out = []
        for g in groups.values():
            if len(g) < 2:
                continue
            names = [h.songs[cand[a]] for a in g]
            songs = sorted(({"song": s, "p": round(float(probs.get(s, 0)), 3)} for s in names),
                           key=lambda d: -d["p"])
            out.append({"songs": songs})
        out.sort(key=lambda g: -sum(s["p"] for s in g["songs"]))
        return out[:6]


def score_prediction(payload: dict, actual_songs: list[str]) -> dict:
    """Grade a logged prediction against the real setlist."""
    actual = set(actual_songs)
    songs = payload["songs"]
    y = np.array([s["song"] in actual for s in songs], dtype=float)
    pm = np.array([s["p"] for s in songs])
    pb = np.array([s["p_baseline"] for s in songs])
    k = len(actual)
    top_m = {s["song"] for s in sorted(songs, key=lambda s: -s["p"])[:k]}
    top_b = {s["song"] for s in sorted(songs, key=lambda s: -s["p_baseline"])[:k]}
    predicted = {s["song"] for s in payload["setlist"]}
    return {
        "actual": actual_songs,
        "n_actual": k,
        "surprises": [s for s in actual_songs if s not in {x["song"] for x in songs}],
        "model": {"brier": float(np.mean((pm - y) ** 2)), "hits_at_set_size": len(top_m & actual) / max(k, 1)},
        "baseline": {"brier": float(np.mean((pb - y) ** 2)), "hits_at_set_size": len(top_b & actual) / max(k, 1)},
        "setlist_hits": len(predicted & actual),
        "setlist_size": len(predicted),
        "scored_on": date.today().isoformat(),
    }
