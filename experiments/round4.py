"""Round 4 (round D in PROCESS.md): new features and artist weighting.

Run:  .venv/bin/python -m experiments.round4

Step 1 (choose): on the 55-70% tuning block only, try the current 19 features, each new
group added on its own, all groups together, all groups minus each one, and two ways of
weighting artists. Done for both site models.
Step 2 (score): the chosen set vs the current features on the 70-85% and 85-100% test
blocks, per artist, with a bootstrap interval. Plus speed: building the data and retraining.
"""
from __future__ import annotations

import json
import time
import warnings

import numpy as np
import pandas as pd

from backend import store
from backend.features import ALL_FEATURES, BASE_FEATURES, NEW_FEATURE_GROUPS, History, parse_all
from backend.model import CATBOOST_PARAMS
from experiments.compare_models import CACHE, TESTS, TUNE, _unsynced, score, split

warnings.filterwarnings("ignore")
V2 = CACHE / "v2"
# the two models the site offered at the time; the site now uses only CatBoost
MODELS = {
    "accurate": ("catboost", CATBOOST_PARAMS),
    "fast": ("lightgbm", dict(
        num_leaves=42, min_child_samples=10, n_estimators=150, learning_rate=0.0391, reg_lambda=0.00214,
        subsample=0.887, subsample_freq=1, colsample_bytree=0.695, verbose=-1, random_state=0, n_jobs=-1)),
}
MIN_SHOWS = 80


def load_tables():
    V2.mkdir(parents=True, exist_ok=True)
    out, build_s = {}, 0.0
    for a in store.list_artists() + _unsynced():
        f = V2 / f"{a['mbid']}.pkl"
        if f.exists():
            df = pd.read_pickle(f)
        else:
            t = time.time()
            played, _ = parse_all(store.load_setlists(a["mbid"]))
            if len(played) < MIN_SHOWS:
                continue
            h = History(played)
            df = h.training_table()
            build_s += time.time() - t
            df["set_len"] = df["show_i"].map({i: int(h.P[i].sum()) for i in range(len(played))})
            df["n_shows"] = len(played)
            df.to_pickle(f)
        out[a["name"]] = df.assign(artist=a["name"])
    return out, build_s


def weights(train, how):
    if how is None:
        return None
    n = train.groupby("artist")["label"].transform("size").values.astype(float)
    w = 1 / n if how == "equal" else 1 / np.sqrt(n)
    return w / w.mean()


def fit(train, kind, cols, weight=None):
    lib, params = MODELS[kind]
    X, y = train[cols].values.astype(np.float32), train["label"].values
    w = weights(train, weight)
    if lib == "catboost":
        from catboost import CatBoostClassifier
        clf = CatBoostClassifier(**params).fit(X, y, sample_weight=w)
    else:
        import lightgbm as lgb
        clf = lgb.LGBMClassifier(**params).fit(X, y, sample_weight=w)
    return lambda df: clf.predict_proba(df[cols].values.astype(np.float32))[:, 1]


def tune_score(tables, kind, cols, weight=None):
    """One fit on every artist's shows before the tuning block, scored on the block."""
    parts = {n: split(df, *TUNE) for n, df in tables.items()}
    predict = fit(pd.concat([tr for tr, _ in parts.values()]), kind, cols, weight)
    return float(np.mean([score(te, predict(te))["brier"] for _, te in parts.values()]))


def evaluate(tables, kind, cols, weight=None):
    rows = []
    for name, df in tables.items():
        others = pd.concat([t for n, t in tables.items() if n != name])
        for lo, hi in TESTS:
            train, test = split(df, lo, hi)
            predict = fit(pd.concat([train, others]), kind, cols, weight)
            rows.append({"artist": name, **score(test, predict(test))})
    return pd.DataFrame(rows).groupby("artist").mean()


def configs():
    groups = NEW_FEATURE_GROUPS
    yield "current 19 features", BASE_FEATURES, None
    for g, cols in groups.items():
        yield f"current + {g}", BASE_FEATURES + cols, None
    yield "all new features", ALL_FEATURES, None
    for g, cols in groups.items():
        yield f"all minus {g}", [c for c in ALL_FEATURES if c not in cols], None
    yield "all + equal artist weight", ALL_FEATURES, "equal"
    yield "all + sqrt artist weight", ALL_FEATURES, "sqrt"


if __name__ == "__main__":
    tables, build_s = load_tables()
    print(f"{len(tables)} artists, {sum(len(v) for v in tables.values())} rows "
          f"(building new tables took {build_s:.1f}s)", flush=True)

    # ---------- step 1: choose on the tuning block ----------
    screen = {}
    for kind in MODELS:
        print(f"\nTuning block, {kind} model:", flush=True)
        res = {}
        for label, cols, w in configs():
            res[label] = tune_score(tables, kind, cols, w)
            print(f"  {label:32s} {res[label]:.5f}", flush=True)
        screen[kind] = res
    s = pd.DataFrame(screen)
    s.to_csv(CACHE / "round4_screen.csv")

    # keep a group if both models agree it helps: removing it from "all" makes things worse
    keep = [g for g in NEW_FEATURE_GROUPS
            if all(s.loc[f"all minus {g}", k] > s.loc["all new features", k] for k in MODELS)]
    chosen = BASE_FEATURES + [c for g in keep for c in NEW_FEATURE_GROUPS[g]]
    w_best = {k: min([None, "equal", "sqrt"], key=lambda w: s.loc[
        {None: "all new features", "equal": "all + equal artist weight", "sqrt": "all + sqrt artist weight"}[w], k])
        for k in MODELS}
    weight = w_best["accurate"] if w_best["accurate"] == w_best["fast"] else None
    print(f"\nKept groups: {keep}; artist weighting: {weight}", flush=True)
    (CACHE / "round4_choice.json").write_text(json.dumps({"groups": keep, "weight": weight, "features": chosen}, indent=2))

    # ---------- step 2: score on the test blocks ----------
    allrows = pd.concat(tables.values())
    results, per = [], {}
    for kind in MODELS:
        for label, cols, w in [("current features", BASE_FEATURES, None), ("new features", chosen, weight)]:
            t = time.time()
            r = evaluate(tables, kind, cols, w)
            t0 = time.time()
            fit(allrows, kind, cols, w)
            retrain = time.time() - t0
            name = f"{kind}: {label}"
            per[name] = r["brier"]
            results.append({"model": name, **r.mean().to_dict(), "retrain_s": round(retrain, 1)})
            print(f"{name:32s} brier {r['brier'].mean():.5f}  precision {r['precision'].mean():.3f}  "
                  f"retrain {retrain:.1f}s  ({round(time.time() - t)}s)", flush=True)

    out = pd.DataFrame(results)
    out.to_csv(CACHE / "results_round4.csv", index=False)
    print("\n", out.to_string(index=False))
    table = pd.DataFrame(per)
    table.to_csv(CACHE / "per_artist_round4.csv")
    print("\nBrier by artist\n", table.round(4).to_string())
    rng = np.random.default_rng(0)
    for kind in MODELS:
        d = table[f"{kind}: current features"] - table[f"{kind}: new features"]   # positive = new is better
        boots = [d.sample(frac=1, replace=True, random_state=int(rng.integers(1e9))).mean() for _ in range(5000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"{kind}: new features better on {int((d > 0).sum())}/{len(d)} artists; "
              f"gain {d.mean():+.5f} ({d.mean() / table[f'{kind}: current features'].mean():+.1%}), "
              f"95% interval {lo:+.5f} to {hi:+.5f}")
