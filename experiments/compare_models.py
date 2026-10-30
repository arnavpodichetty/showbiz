"""Compare ML models for setlist prediction on the artists already downloaded.

Run:  DYLD_LIBRARY_PATH=.venv/lib/python3.9/site-packages/sklearn/.dylibs \
      .venv/bin/python -m experiments.compare_models

Evaluation (no peeking at the future):
  - For each artist, shows are in date order. Two test blocks: shows at 70-85% and 85-100%
    of their history. Each model is trained only on shows before the block it's tested on.
  - Any tuning uses an earlier block (55-70%), never the test blocks.
  - "per-artist" models train on that artist only; "pooled" models also train on every
    other artist's shows (as the live site could).
Score: Brier score (lower is better), averaged so every artist counts equally.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend import store  # noqa: E402
from backend.features import FEATURES, History, parse_all  # noqa: E402
from backend.model import baseline_probs  # noqa: E402

warnings.filterwarnings("ignore")
CACHE = Path(__file__).resolve().parent / "cache"
TUNE = (0.55, 0.70)
TESTS = [(0.70, 0.85), (0.85, 1.00)]
MIN_SHOWS = 80
EPS = 0.005


# ---------------- data ----------------

def load_tables() -> dict[str, pd.DataFrame]:
    CACHE.mkdir(exist_ok=True)
    out = {}
    for a in store.list_artists() + [x for x in _unsynced()]:
        f = CACHE / f"{a['mbid']}.pkl"
        if f.exists():
            df = pd.read_pickle(f)
        else:
            played, _ = parse_all(store.load_setlists(a["mbid"]))
            if len(played) < MIN_SHOWS:
                continue
            h = History(played)
            df = h.training_table()
            df["set_len"] = df["show_i"].map({i: int(h.P[i].sum()) for i in range(len(played))})
            df["n_shows"] = len(played)
            df.to_pickle(f)
        out[a["name"]] = df
    return out


def _unsynced():
    import sqlite3
    c = sqlite3.connect(store.DB_PATH)
    return [{"mbid": m, "name": n} for m, n in c.execute("SELECT mbid, name FROM artists WHERE last_synced IS NULL")]


def split(df: pd.DataFrame, lo: float, hi: float):
    n = int(df["n_shows"].iloc[0])
    a, b = int(n * lo), int(n * hi)
    return df[df.show_i < a], df[(df.show_i >= a) & (df.show_i < b)]


# ---------------- models ----------------

def _xy(df):
    return df[FEATURES].values.astype(np.float32), df["label"].values


def m_baseline(train, test, params=None):
    return baseline_probs(test)


def m_logreg(train, test, params=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X, y = _xy(train)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=(params or {}).get("C", 1.0), max_iter=2000))
    clf.fit(X, y)
    return clf.predict_proba(_xy(test)[0])[:, 1]


def m_hgb(train, test, params=None):
    from sklearn.ensemble import HistGradientBoostingClassifier
    p = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
             l2_regularization=1.0, early_stopping=True, validation_fraction=0.1, random_state=0)
    p.update(params or {})
    X, y = _xy(train)
    return HistGradientBoostingClassifier(**p).fit(X, y).predict_proba(_xy(test)[0])[:, 1]


def m_rf(train, test, params=None):
    from sklearn.ensemble import RandomForestClassifier
    X, y = _xy(train)
    clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=20, max_features=0.5, n_jobs=-1, random_state=0)
    return clf.fit(X, y).predict_proba(_xy(test)[0])[:, 1]


def m_lgbm(train, test, params=None):
    import lightgbm as lgb
    p = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=40,
             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, random_state=0)
    p.update(params or {})
    X, y = _xy(train)
    return lgb.LGBMClassifier(**p).fit(X, y).predict_proba(_xy(test)[0])[:, 1]


def m_xgb(train, test, params=None):
    import xgboost as xgb
    p = dict(n_estimators=500, learning_rate=0.03, max_depth=5, min_child_weight=5,
             subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", random_state=0)
    p.update(params or {})
    X, y = _xy(train)
    return xgb.XGBClassifier(**p).fit(X, y).predict_proba(_xy(test)[0])[:, 1]


def m_catboost(train, test, params=None):
    from catboost import CatBoostClassifier
    p = dict(iterations=600, learning_rate=0.05, depth=6, l2_leaf_reg=3, verbose=0, random_seed=0, thread_count=-1)
    p.update(params or {})
    X, y = _xy(train)
    return CatBoostClassifier(**p).fit(X, y).predict_proba(_xy(test)[0])[:, 1]


def m_mlp(train, test, params=None):
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X, y = _xy(train)
    clf = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3,
                                                        early_stopping=True, max_iter=200, random_state=0))
    return clf.fit(X, y).predict_proba(_xy(test)[0])[:, 1]


MODELS = {
    "baseline (tour play count)": m_baseline,
    "logistic regression": m_logreg,
    "random forest": m_rf,
    "neural net (MLP)": m_mlp,
    "sklearn HistGB (current)": m_hgb,
    "LightGBM": m_lgbm,
    "XGBoost": m_xgb,
    "CatBoost": m_catboost,
}


# ---------------- scoring ----------------

def score(test: pd.DataFrame, p: np.ndarray) -> dict:
    p = np.clip(p, EPS, 1 - EPS)
    y = test["label"].values
    prec = []
    for _, g in test.assign(p=p).groupby("show_i"):
        k = int(g["set_len"].iloc[0])
        prec.append(g.sort_values("p", ascending=False)["label"].values[:k].mean())
    return {"brier": brier_score_loss(y, p), "log_loss": log_loss(y, p, labels=[0, 1]), "precision": float(np.mean(prec))}


def evaluate(tables, fn, pooled: bool, params=None, folds=TESTS) -> pd.DataFrame:
    rows = []
    for name, df in tables.items():
        for lo, hi in folds:
            train, test = split(df, lo, hi)
            if pooled:
                others = pd.concat([t for n, t in tables.items() if n != name])
                train = pd.concat([train, others])
            p = fn(train, test, params)
            rows.append({"artist": name, "fold": f"{lo:.2f}-{hi:.2f}", **score(test, p)})
    return pd.DataFrame(rows)


def summarize(res: pd.DataFrame) -> dict:
    per_artist = res.groupby("artist")[["brier", "log_loss", "precision"]].mean()
    return per_artist.mean().to_dict()


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    tables = load_tables()
    print("Artists:", {k: int(v["n_shows"].iloc[0]) for k, v in tables.items()})
    print("Rows:", sum(len(v) for v in tables.values()))
    results, detail = [], {}
    for name, fn in MODELS.items():
        for pooled in (False, True):
            if name.startswith("baseline") and pooled:
                continue
            t = time.time()
            res = evaluate(tables, fn, pooled)
            s = summarize(res)
            label = f"{name} [{'pooled' if pooled else 'per-artist'}]"
            detail[label] = res
            results.append({"model": label, **s, "secs": round(time.time() - t)})
            print(f"{label:45s} brier {s['brier']:.4f}  logloss {s['log_loss']:.4f}  precision {s['precision']:.3f}  ({results[-1]['secs']}s)", flush=True)
    out = pd.DataFrame(results).sort_values("brier")
    out.to_csv(CACHE / "results_round1.csv", index=False)
    pd.concat({k: v for k, v in detail.items()}).to_csv(CACHE / "detail_round1.csv")
    print("\n", out.to_string(index=False))
