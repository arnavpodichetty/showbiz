"""Train on every artist in the local database and save the model that ships with Showbiz.

Run:  .venv/bin/python -m backend.export_model

Writes model/showbiz.cbm (CatBoost's own format, which is safe to load, unlike a pickle)
and model/showbiz.json (what it was trained on). Commit both. A fresh copy of the site
uses this model until its own database holds at least as many setlists.
The model contains learned patterns only: no setlists, song names or artist names.
"""
from __future__ import annotations

import json
import time
from datetime import date

from . import app as A
from . import store
from .features import FEATURES
from .model import CATBOOST_PARAMS, train_global


def main() -> None:
    t = time.time()
    counts = store.setlist_counts()
    for m, n in counts.items():
        A._history(m, n)
    usable = [m for m in counts if len(A._histories[m][2])]
    hists = {m: A._histories[m][1] for m in usable}
    tables = {m: A._histories[m][2] for m in usable}
    clf, _ = train_global(hists, tables)
    A.BUNDLED_DIR.mkdir(exist_ok=True)
    clf.save_model(str(A.BUNDLED_MODEL))
    meta = {
        "trained": date.today().isoformat(),
        "n_artists": len(usable),
        "n_setlists": int(sum(counts.values())),
        "n_rows": int(sum(len(t) for t in tables.values())),
        "features": FEATURES,
        "params": {k: v for k, v in CATBOOST_PARAMS.items() if k not in ("verbose", "thread_count", "allow_writing_files")},
    }
    A.BUNDLED_META.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Saved {A.BUNDLED_MODEL.name}: {meta['n_artists']} artists, {meta['n_setlists']} setlists, "
          f"{meta['n_rows']} rows ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()
