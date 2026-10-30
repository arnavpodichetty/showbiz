"""Round 2: tune the strongest models, then try calibration and blends.

Tuning picks settings on the 55-70% block only; the winners are then scored on the
same two test blocks as round 1, so the numbers are directly comparable.
"""
from __future__ import annotations

import itertools
import sys
import time

import numpy as np
import pandas as pd

from experiments.compare_models import (
    CACHE, TESTS, TUNE, evaluate, load_tables, m_catboost, m_hgb, m_lgbm, m_logreg, m_xgb, score, split, summarize,
)

GRIDS = {
    "LightGBM": (m_lgbm, {
        "num_leaves": [7, 15, 31],
        "min_child_samples": [40, 200],
        "n_estimators": [300, 800],
        "learning_rate": [0.03],
        "reg_lambda": [1.0, 10.0],
    }),
    "sklearn HistGB": (m_hgb, {
        "max_leaf_nodes": [7, 15, 31],
        "min_samples_leaf": [40, 200],
        "learning_rate": [0.03, 0.08],
        "l2_regularization": [1.0, 10.0],
    }),
    "XGBoost": (m_xgb, {
        "max_depth": [3, 5],
        "min_child_weight": [5, 50],
        "n_estimators": [300, 800],
        "learning_rate": [0.03],
        "reg_lambda": [1.0, 10.0],
    }),
    "CatBoost": (m_catboost, {
        "depth": [4, 6, 8],
        "l2_leaf_reg": [3, 10],
        "iterations": [400, 900],
    }),
}


def grid(g):
    keys = list(g)
    for vals in itertools.product(*(g[k] for k in keys)):
        yield dict(zip(keys, vals))


def tune(tables, fn, g, pooled):
    if pooled:
        # one fit per setting: train on every artist's shows before the tuning block
        parts = {n: split(df, *TUNE) for n, df in tables.items()}
        train = pd.concat([tr for tr, _ in parts.values()])
        test = pd.concat([te.assign(_artist=n) for n, (_, te) in parts.items()])
    best = None
    for params in grid(g):
        if pooled:
            p = fn(train, test, params)
            per = [score(test[m], p[m.values]) for m in (test["_artist"] == n for n in tables)]
            s = {"brier": float(np.mean([x["brier"] for x in per]))}
        else:
            s = summarize(evaluate(tables, fn, pooled, params, folds=[TUNE]))
        if best is None or s["brier"] < best[1]:
            best = (params, s["brier"])
    return best


def blend_eval(tables, members, pooled, weights=None):
    """Average the probabilities of several (fn, params) models."""
    rows = []
    for name, df in tables.items():
        for lo, hi in TESTS:
            train, test = split(df, lo, hi)
            if pooled:
                train = pd.concat([train] + [t for n, t in tables.items() if n != name])
            ps = np.array([fn(train, test, params) for fn, params in members])
            w = np.ones(len(members)) / len(members) if weights is None else np.array(weights)
            rows.append({"artist": name, "fold": f"{lo:.2f}-{hi:.2f}", **score(test, w @ ps)})
    return pd.DataFrame(rows)


def calibrated_eval(tables, fn, params, pooled):
    """Isotonic calibration fitted on the last 15% of each training window (by time)."""
    from sklearn.isotonic import IsotonicRegression
    rows = []
    for name, df in tables.items():
        for lo, hi in TESTS:
            train, test = split(df, lo, hi)
            cut = int(train.show_i.max() * 0.85)
            fit, cal = train[train.show_i < cut], train[train.show_i >= cut]
            others = [t for n, t in tables.items() if n != name] if pooled else []
            fit_all = pd.concat([fit] + others)
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.005, y_max=0.995)
            iso.fit(fn(fit_all, cal, params), cal["label"].values)
            p = iso.predict(fn(pd.concat([train] + others), test, params))
            rows.append({"artist": name, "fold": f"{lo:.2f}-{hi:.2f}", **score(test, p)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pooled = "--per-artist" not in sys.argv
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or list(GRIDS)
    tables = load_tables()
    tuned, results = {}, []
    for name in names:
        fn, g = GRIDS[name]
        t = time.time()
        params, tune_brier = tune(tables, fn, g, pooled)
        s = summarize(evaluate(tables, fn, pooled, params))
        tuned[name] = (fn, params)
        results.append({"model": f"{name} tuned", **s, "params": params})
        print(f"{name:16s} tuned -> {params}  test brier {s['brier']:.4f} precision {s['precision']:.3f} ({round(time.time()-t)}s)", flush=True)

    ranked = sorted(results, key=lambda r: r["brier"])
    top = [r["model"].replace(" tuned", "") for r in ranked]

    # isotonic calibration on the best tuned model
    fn, params = tuned[top[0]]
    s = summarize(calibrated_eval(tables, fn, params, pooled))
    results.append({"model": f"{top[0]} tuned + isotonic calibration", **s})
    print(f"{top[0]} + calibration  test brier {s['brier']:.4f} precision {s['precision']:.3f}", flush=True)

    # blends of the top 2 and top 3, plus top 3 with logistic regression for diversity
    for k in (2, 3):
        if len(top) >= k:
            s = summarize(blend_eval(tables, [tuned[n] for n in top[:k]], pooled))
            results.append({"model": f"blend of {', '.join(top[:k])}", **s})
            print(f"blend top {k}  test brier {s['brier']:.4f} precision {s['precision']:.3f}", flush=True)
    s = summarize(blend_eval(tables, [tuned[top[0]], (m_logreg, None)], pooled, weights=[0.8, 0.2]))
    results.append({"model": f"0.8 {top[0]} + 0.2 logistic regression", **s})
    print(f"top + logreg  test brier {s['brier']:.4f} precision {s['precision']:.3f}", flush=True)

    out = pd.DataFrame(results).sort_values("brier")
    out.to_csv(CACHE / f"results_round2_{'pooled' if pooled else 'per_artist'}.csv", index=False)
    print("\n", out.drop(columns=["params"], errors="ignore").to_string(index=False))

    # artist-by-artist check: baseline, the old site model, and the two best tuned models
    from experiments.compare_models import m_baseline
    finalists = {
        "baseline": (m_baseline, False, None),
        "old site model (HistGB, per-artist)": (m_hgb, False, None),
        f"{top[0]} tuned, pooled": (*tuned[top[0]], ),
        f"{top[1]} tuned, pooled": (*tuned[top[1]], ),
    }
    per = {}
    for label, cfg in finalists.items():
        fn, params = cfg[0], cfg[-1]
        is_pooled = pooled if label.endswith("pooled") else False
        per[label] = evaluate(tables, fn, is_pooled, params).groupby("artist")["brier"].mean()
    table = pd.DataFrame(per)
    table["shows"] = {k: int(v["n_shows"].iloc[0]) for k, v in tables.items()}
    print("\nBrier by artist\n", table.round(4).to_string())
    table.to_csv(CACHE / "per_artist_final.csv")

    a, b = table[f"{top[0]} tuned, pooled"], table[f"{top[1]} tuned, pooled"]
    old = table["old site model (HistGB, per-artist)"]
    rng = np.random.default_rng(0)
    diffs = [(b - a).sample(frac=1, replace=True, random_state=int(rng.integers(1e9))).mean() for _ in range(5000)]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"\n{top[0]} beats {top[1]} on {int((a < b).sum())}/{len(a)} artists; "
          f"mean gap {float((b - a).mean()):.5f} (95% interval {lo:.5f} to {hi:.5f})")
    print(f"{top[0]} beats the old site model on {int((a < old).sum())}/{len(a)} artists")
