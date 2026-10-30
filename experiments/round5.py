"""Round 5 (round E in PROCESS.md): does training on more artists help?

Run:  .venv/bin/python -m experiments.round5

1. The original 17 artists, scored on the same test blocks as rounds B-D, with the model
   trained on (a) those 17 only and (b) every artist now downloaded.
2. The newly added artists, which played no part in choosing the features in round D:
   the site's features vs the original 19, and the baseline. A fresh check of round D.
   (One fit per test block on every artist's earlier shows, to keep 80+ artists affordable.)
All runs use the site's model: CatBoost, Optuna settings, every artist weighted equally.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from backend.features import BASE_FEATURES, FEATURES
from backend.model import baseline_probs
from experiments.compare_models import CACHE, TESTS, score, split
from experiments.round4 import fit, load_tables

warnings.filterwarnings("ignore")


def evaluate(targets: dict, pool: dict, cols, weight="equal") -> pd.DataFrame:
    """Score each target artist on its test blocks; train on its earlier shows plus every
    other artist in the pool."""
    rows = []
    for name, df in targets.items():
        others = [t for n, t in pool.items() if n != name]
        for lo, hi in TESTS:
            train, test = split(df, lo, hi)
            predict = fit(pd.concat([train] + others), "accurate", cols, weight)
            rows.append({"artist": name, **score(test, predict(test))})
    return pd.DataFrame(rows).groupby("artist").mean()


def evaluate_blocks(targets: dict, pool: dict, cols, weight="equal") -> pd.DataFrame:
    """Cheaper and stricter: for each test block, ONE fit on every pool artist's shows before
    that block, then score every target artist's block. No artist's test shows are trained on."""
    rows = []
    for lo, hi in TESTS:
        parts = {n: split(df, lo, hi) for n, df in pool.items()}
        predict = fit(pd.concat([tr for tr, _ in parts.values()]), "accurate", cols, weight)
        for name in targets:
            test = parts[name][1]
            rows.append({"artist": name, **score(test, predict(test))})
    return pd.DataFrame(rows).groupby("artist").mean()


def baseline(targets: dict) -> pd.DataFrame:
    rows = []
    for name, df in targets.items():
        for lo, hi in TESTS:
            _, test = split(df, lo, hi)
            rows.append({"artist": name, **score(test, baseline_probs(test))})
    return pd.DataFrame(rows).groupby("artist").mean()


def interval(d: pd.Series, seed=0):
    rng = np.random.default_rng(seed)
    boots = [d.sample(frac=1, replace=True, random_state=int(rng.integers(1e9))).mean() for _ in range(5000)]
    return np.percentile(boots, [2.5, 97.5])


if __name__ == "__main__":
    tables, _ = load_tables()
    original = list(pd.read_csv(CACHE / "per_artist_round4.csv", index_col=0).index)
    orig = {n: tables[n] for n in original if n in tables}
    new = {n: t for n, t in tables.items() if n not in orig}
    print(f"{len(orig)} original artists, {len(new)} new: {sorted(new)}")
    print(f"rows: original {sum(map(len, orig.values()))}, all {sum(map(len, tables.values()))}", flush=True)

    # ---- 1. original 17: trained on 17 vs on everyone ----
    t = time.time()
    a = evaluate(orig, orig, FEATURES)
    print(f"original 17, trained on 17 artists: brier {a.brier.mean():.5f}  precision {a.precision.mean():.3f}  ({time.time() - t:.0f}s)", flush=True)
    t = time.time()
    b = evaluate(orig, tables, FEATURES)
    print(f"original 17, trained on {len(tables)} artists: brier {b.brier.mean():.5f}  precision {b.precision.mean():.3f}  ({time.time() - t:.0f}s)", flush=True)
    d = a.brier - b.brier                                  # positive = more artists is better
    lo, hi = interval(d)
    print(f"  more artists better for {int((d > 0).sum())}/{len(d)}; gain {d.mean():+.5f} ({d.mean() / a.brier.mean():+.1%}), 95% interval {lo:+.5f} to {hi:+.5f}", flush=True)

    # ---- 2. new artists: fresh check of the round D features ----
    t = time.time()
    c_new = evaluate_blocks(new, tables, FEATURES)
    c_old = evaluate_blocks(new, tables, BASE_FEATURES, weight=None)
    c_base = baseline(new)
    print(f"new artists, site features (22 + equal weight): brier {c_new.brier.mean():.5f}  precision {c_new.precision.mean():.3f}")
    print(f"new artists, original 19 features, no weighting: brier {c_old.brier.mean():.5f}  precision {c_old.precision.mean():.3f}")
    print(f"new artists, baseline:                           brier {c_base.brier.mean():.5f}  precision {c_base.precision.mean():.3f}  ({time.time() - t:.0f}s)")
    d2 = c_old.brier - c_new.brier
    lo, hi = interval(d2)
    print(f"  round D features better for {int((d2 > 0).sum())}/{len(d2)}; gain {d2.mean():+.5f} ({d2.mean() / c_old.brier.mean():+.1%}), 95% interval {lo:+.5f} to {hi:+.5f}")
    print(f"  model beats baseline for {int((c_new.brier < c_base.brier).sum())}/{len(c_new)}")

    per = pd.concat({
        "trained on 17": a.brier, f"trained on {len(tables)}": b.brier,
        "site features": c_new.brier, "original 19": c_old.brier, "baseline": c_base.brier,
    }, axis=1)
    per["shows"] = {n: int(t["n_shows"].iloc[0]) for n, t in tables.items()}
    per.to_csv(CACHE / "per_artist_round5.csv")
    print("\nBrier by artist\n", per.round(4).to_string())
