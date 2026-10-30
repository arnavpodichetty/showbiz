"""Round 3: new model types plus Optuna tuning, scored for accuracy AND speed.

Run:  DYLD_LIBRARY_PATH=.venv/lib/python3.9/site-packages/sklearn/.dylibs \
      .venv/bin/python -m experiments.round3

Same rules as rounds 1-2: settings are picked on the 55-70% block, scores come from
the 70-85% and 85-100% blocks, every model is pooled (trained on all artists).

Speed budget: a full retrain on every artist must take under FIT_BUDGET seconds, so the
site stays responsive. Optuna throws out any setting that would break it.
"""
from __future__ import annotations

import json
import sys
import time
import warnings

import numpy as np
import optuna
import pandas as pd

from backend.features import BASE_FEATURES as FEATURES  # round 3 used the original 19
from experiments.compare_models import CACHE, TESTS, TUNE, _xy, load_tables, score, split

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

FIT_BUDGET = 15.0          # seconds for a full retrain on all rows
TUNE_SHARE = 0.55          # tuning fits use ~55% of the rows, so scale the budget
N_TRIALS = {"CatBoost": 40, "LightGBM": 60, "XGBoost": 60}

# features where "more" should only ever raise (+1) or lower (-1) a song's chance
MONOTONE = {"rate_last5": 1, "rate_last10": 1, "rate_last25": 1, "rate_last100": 1,
            "rate_tour": 1, "rate_country": 1, "played_in_run": -1}


# ---------------- models: fit(train, params) -> predict(df) ----------------

def fit_catboost(train, params, cat_artist=False):
    from catboost import CatBoostClassifier
    p = dict(depth=8, l2_leaf_reg=10, iterations=400, learning_rate=0.05, verbose=0,
             random_seed=0, thread_count=-1, allow_writing_files=False)
    p.update(params or {})
    if cat_artist:
        cols = FEATURES + ["artist"]
        clf = CatBoostClassifier(**p, cat_features=["artist"]).fit(train[cols], train["label"].values)
        return lambda df: clf.predict_proba(df[cols])[:, 1]
    X, y = _xy(train)
    clf = CatBoostClassifier(**p).fit(X, y)
    return lambda df: clf.predict_proba(_xy(df)[0])[:, 1]


def fit_lgbm(train, params, monotone=False):
    import lightgbm as lgb
    p = dict(n_estimators=300, learning_rate=0.03, num_leaves=31, min_child_samples=40, subsample=0.8,
             subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, random_state=0, n_jobs=-1)
    p.update(params or {})
    if monotone:
        p["monotone_constraints"] = [MONOTONE.get(f, 0) for f in FEATURES]
    X, y = _xy(train)
    clf = lgb.LGBMClassifier(**p).fit(X, y)
    return lambda df: clf.predict_proba(_xy(df)[0])[:, 1]


def fit_xgb(train, params):
    import xgboost as xgb
    p = dict(n_estimators=300, learning_rate=0.03, max_depth=5, min_child_weight=5, subsample=0.8,
             colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", random_state=0, n_jobs=-1)
    p.update(params or {})
    X, y = _xy(train)
    clf = xgb.XGBClassifier(**p).fit(X, y)
    return lambda df: clf.predict_proba(_xy(df)[0])[:, 1]


def fit_ranker(train, params):
    """LightGBM LambdaRank: learns the order of songs within each show, then a small
    logistic regression turns (score, rank in show) into a probability."""
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression
    p = dict(objective="lambdarank", n_estimators=300, learning_rate=0.05, num_leaves=31,
             min_child_samples=40, verbose=-1, random_state=0, n_jobs=-1, lambdarank_truncation_level=30)
    p.update(params or {})
    tr = train.sort_values(["artist", "show_i"])
    shows = tr["artist"] + "|" + tr["show_i"].astype(str)
    # hold out a random 15% of shows to fit the probability mapping honestly
    rng = np.random.default_rng(0)
    uniq = shows.unique()
    cal_ids = set(rng.choice(uniq, size=len(uniq) * 15 // 100, replace=False))
    is_cal = shows.isin(cal_ids).values
    fit_part, cal_part = tr[~is_cal], tr[is_cal]
    groups = (fit_part["artist"] + "|" + fit_part["show_i"].astype(str)).value_counts(sort=False)
    groups = groups.reindex((fit_part["artist"] + "|" + fit_part["show_i"].astype(str)).unique())
    rk = lgb.LGBMRanker(**p).fit(_xy(fit_part)[0], fit_part["label"].values, group=groups.values)

    def feats(df):
        s = rk.predict(_xy(df)[0])
        key = df["artist"] + "|" + df["show_i"].astype(str)
        pct = pd.Series(s, index=df.index).groupby(key.values).rank(pct=True).values
        return np.column_stack([s, pct])

    lr = LogisticRegression(max_iter=1000).fit(feats(cal_part), cal_part["label"].values)
    return lambda df: lr.predict_proba(feats(df))[:, 1]


def fit_spline_logreg(train, params):
    """Logistic regression on spline-expanded features: very fast, smooth curves per feature."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import SplineTransformer, StandardScaler
    X, y = _xy(train)
    clf = make_pipeline(SplineTransformer(n_knots=6, degree=3), StandardScaler(),
                        LogisticRegression(C=1.0, max_iter=500))
    clf.fit(X, y)
    return lambda df: clf.predict_proba(_xy(df)[0])[:, 1]


# ---------------- Optuna search spaces ----------------

def space_catboost(t):
    p = dict(depth=t.suggest_int("depth", 4, 10),
             learning_rate=t.suggest_float("learning_rate", 0.02, 0.2, log=True),
             iterations=t.suggest_int("iterations", 150, 900, step=50),
             l2_leaf_reg=t.suggest_float("l2_leaf_reg", 1, 30, log=True),
             random_strength=t.suggest_float("random_strength", 0.1, 10, log=True),
             bagging_temperature=t.suggest_float("bagging_temperature", 0, 2),
             border_count=t.suggest_categorical("border_count", [64, 128, 254]),
             grow_policy=t.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise"]))
    if p["grow_policy"] == "Depthwise":
        p["min_data_in_leaf"] = t.suggest_int("min_data_in_leaf", 1, 300, log=True)
    return p


def space_lgbm(t):
    return dict(num_leaves=t.suggest_int("num_leaves", 7, 255, log=True),
                min_child_samples=t.suggest_int("min_child_samples", 10, 500, log=True),
                n_estimators=t.suggest_int("n_estimators", 100, 1200, step=50),
                learning_rate=t.suggest_float("learning_rate", 0.01, 0.2, log=True),
                reg_lambda=t.suggest_float("reg_lambda", 1e-3, 30, log=True),
                subsample=t.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree=t.suggest_float("colsample_bytree", 0.4, 1.0))


def space_xgb(t):
    return dict(max_depth=t.suggest_int("max_depth", 3, 10),
                min_child_weight=t.suggest_float("min_child_weight", 1, 100, log=True),
                n_estimators=t.suggest_int("n_estimators", 100, 1200, step=50),
                learning_rate=t.suggest_float("learning_rate", 0.01, 0.2, log=True),
                reg_lambda=t.suggest_float("reg_lambda", 1e-3, 30, log=True),
                subsample=t.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree=t.suggest_float("colsample_bytree", 0.4, 1.0))


SEARCH = {"CatBoost": (fit_catboost, space_catboost),
          "LightGBM": (fit_lgbm, space_lgbm),
          "XGBoost": (fit_xgb, space_xgb)}


def optuna_tune(tables, name):
    fit, space = SEARCH[name]
    parts = {n: split(df, *TUNE) for n, df in tables.items()}
    train = pd.concat([tr for tr, _ in parts.values()])
    tests = {n: te for n, (_, te) in parts.items()}

    def objective(trial):
        params = space(trial)
        t = time.time()
        predict = fit(train, params)
        secs = time.time() - t
        if secs > FIT_BUDGET * TUNE_SHARE:
            raise optuna.TrialPruned()          # too slow for the site
        return float(np.mean([score(te, predict(te))["brier"] for te in tests.values()]))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=N_TRIALS[name])
    pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
    print(f"  {name}: best tune brier {study.best_value:.5f} after {len(study.trials)} trials "
          f"({pruned} too slow)  {study.best_params}", flush=True)
    return study.best_params


# ---------------- evaluation ----------------

def evaluate(tables, fit, params) -> pd.DataFrame:
    rows = []
    for name, df in tables.items():
        others = pd.concat([t for n, t in tables.items() if n != name])
        for lo, hi in TESTS:
            train, test = split(df, lo, hi)
            predict = fit(pd.concat([train, others]), params)
            rows.append({"artist": name, **score(test, predict(test))})
    return pd.DataFrame(rows).groupby("artist").mean()


def speed(tables, fit, params):
    """Full retrain on every row, then predict one typical show (about 100 candidate songs)."""
    allrows = pd.concat(tables.values())
    t = time.time()
    predict = fit(allrows, params)
    fit_s = time.time() - t
    one_show = allrows[allrows.show_i == allrows.show_i.max()].head(100)
    t = time.time()
    for _ in range(20):
        predict(one_show)
    return fit_s, (time.time() - t) / 20 * 1000


if __name__ == "__main__":
    tables = {n: df.assign(artist=n) for n, df in load_tables().items()}
    print(f"{len(tables)} artists, {sum(len(v) for v in tables.values())} rows", flush=True)

    params_file = CACHE / "round3_params.json"
    if "--reuse-params" in sys.argv and params_file.exists():
        tuned = json.loads(params_file.read_text())
        print("Using saved Optuna settings from", params_file.name, flush=True)
    else:
        tuned = {}
        print("Optuna tuning (on the 55-70% block):", flush=True)
        for name in SEARCH:
            t = time.time()
            tuned[name] = optuna_tune(tables, name)
            print(f"    ({round(time.time() - t)}s)", flush=True)
        params_file.write_text(json.dumps(tuned, indent=2))

    # Optuna names CatBoost's min_data_in_leaf etc. the same as the model does, so params pass straight in
    candidates = {
        "CatBoost depth 8 (site now)": (fit_catboost, {}),
        "CatBoost + Optuna": (fit_catboost, tuned["CatBoost"]),
        "CatBoost depth 8 + artist as a feature": (lambda tr, p: fit_catboost(tr, p, cat_artist=True), {}),
        "LightGBM + Optuna": (fit_lgbm, tuned["LightGBM"]),
        "LightGBM + Optuna, monotone": (lambda tr, p: fit_lgbm(tr, p, monotone=True), tuned["LightGBM"]),
        "XGBoost + Optuna": (fit_xgb, tuned["XGBoost"]),
        "LightGBM LambdaRank": (fit_ranker, {}),
        "Spline logistic regression": (fit_spline_logreg, {}),
    }
    only = [a for a in sys.argv[1:] if not a.startswith("--")]
    results, per_artist = [], {}
    for label, (fit, params) in candidates.items():
        if only and not any(o.lower() in label.lower() for o in only):
            continue
        t = time.time()
        res = evaluate(tables, fit, params)
        fit_s, pred_ms = speed(tables, fit, params)
        per_artist[label] = res["brier"]
        row = {"model": label, **res.mean().to_dict(), "retrain_s": round(fit_s, 1), "predict_ms": round(pred_ms, 1)}
        results.append(row)
        print(f"{label:42s} brier {row['brier']:.5f}  precision {row['precision']:.3f}  "
              f"retrain {fit_s:5.1f}s  predict {pred_ms:5.1f}ms  ({round(time.time() - t)}s)", flush=True)

    out = pd.DataFrame(results).sort_values("brier")
    out.to_csv(CACHE / "results_round3.csv", index=False)
    print("\n", out.to_string(index=False))
    table = pd.DataFrame(per_artist)
    table.to_csv(CACHE / "per_artist_round3.csv")
    print("\nBrier by artist\n", table.round(4).to_string())

    ref = "CatBoost depth 8 (site now)"
    if ref in table:
        rng = np.random.default_rng(0)
        for label in table.columns.drop(ref):
            d = table[ref] - table[label]              # positive = label is better
            boots = [d.sample(frac=1, replace=True, random_state=int(rng.integers(1e9))).mean() for _ in range(5000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            print(f"{label:42s} better on {int((d > 0).sum())}/{len(d)} artists; "
                  f"gain {d.mean():+.5f} (95% interval {lo:+.5f} to {hi:+.5f})")
