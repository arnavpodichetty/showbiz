"""Turn raw setlist.fm data into a (song, show) feature table.

Every feature for show i is computed only from shows before i, so the same
code is used for training, backtesting, and predicting a future show.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
import pandas as pd

FESTIVAL_WORDS = re.compile(
    r"festival|\bfest\b|rock am ring|rock im park|lollapalooza|glastonbury|coachella|"
    r"rock in rio|download|reading|leeds|bonnaroo|primavera|roskilde|pinkpop|werchter|"
    r"austin city limits|governors ball|outside lands|summer sonic|fuji rock|isle of wight|"
    r"bbk live|mad cool|nova rock|hellfest|sziget|open'er|lowlands|bottlerock|life is beautiful",
    re.I,
)
MULTI_NIGHT_DAYS = 4       # shows in the same city within this many days count as one run
RECENT_CANDIDATES = 60     # a song is a candidate if played in the last N shows
MIN_SONGS = 3              # ignore setlists shorter than this (usually incomplete)

FEATURES = [
    "rate_last5", "rate_last10", "rate_last25", "rate_last100",
    "rate_tour", "n_tour_prior", "log_shows_since", "played_prev",
    "same_run", "played_in_run", "is_festival", "rate_festival", "rate_headline",
    "rate_country", "n_country_prior", "log_days_since_debut", "log_gap_days",
    "tour_is_new", "no_tour",
]


@dataclass
class Show:
    id: str
    date: date
    venue: str
    city: str
    country: str
    tour: str
    is_festival: bool
    songs: list = field(default_factory=list)       # names, in order
    encore: set = field(default_factory=set)         # names played in an encore
    url: str = ""


def parse_setlist(s: dict) -> Show:
    venue = s.get("venue", {}) or {}
    city = venue.get("city", {}) or {}
    tour = (s.get("tour") or {}).get("name", "") or ""
    songs, encore = [], set()
    for st in (s.get("sets") or {}).get("set", []) or []:
        for song in st.get("song", []) or []:
            name = (song.get("name") or "").strip()
            if not name or song.get("tape"):
                continue
            if name in songs:  # reprises: count once
                continue
            songs.append(name)
            if st.get("encore"):
                encore.add(name)
    text = " ".join([venue.get("name", ""), tour, s.get("info", "") or ""])
    return Show(
        id=s["id"],
        date=datetime.strptime(s["eventDate"], "%d-%m-%Y").date(),
        venue=venue.get("name", ""),
        city=city.get("name", ""),
        country=(city.get("country") or {}).get("code", ""),
        tour=tour,
        is_festival=bool(FESTIVAL_WORDS.search(text)),
        songs=songs,
        encore=encore,
        url=s.get("url", ""),
    )


def parse_all(raw: list[dict]) -> tuple[list[Show], list[Show]]:
    """Returns (played shows with real setlists, sorted by date; upcoming/empty events)."""
    played, upcoming = [], []
    today = date.today()
    for s in raw:
        sh = parse_setlist(s)
        if len(sh.songs) >= MIN_SONGS:
            played.append(sh)
        elif sh.date >= today:
            upcoming.append(sh)
    played.sort(key=lambda x: (x.date, x.id))
    upcoming.sort(key=lambda x: x.date)
    return played, upcoming


@dataclass
class Target:
    """The show we want features for. For history it comes from a real show."""
    date: date
    city: str
    country: str
    tour: str
    is_festival: bool


class History:
    """Precomputed matrices over the played shows of one artist."""

    def __init__(self, shows: list[Show]):
        self.shows = shows
        self.songs = sorted({s for sh in shows for s in sh.songs})
        self.song_idx = {s: j for j, s in enumerate(self.songs)}
        n, m = len(shows), len(self.songs)
        self.P = np.zeros((n, m), dtype=np.float32)
        self.pos = np.full((n, m), np.nan, dtype=np.float32)  # normalized position 0..1
        self.enc = np.zeros((n, m), dtype=np.float32)
        for i, sh in enumerate(shows):
            L = max(len(sh.songs) - 1, 1)
            for k, s in enumerate(sh.songs):
                j = self.song_idx[s]
                self.P[i, j] = 1
                self.pos[i, j] = k / L
                if s in sh.encore:
                    self.enc[i, j] = 1
        self.dates = np.array([sh.date.toordinal() for sh in shows])
        self.tours = np.array([sh.tour for sh in shows])
        self.countries = np.array([sh.country for sh in shows])
        self.cities = np.array([sh.city.lower() for sh in shows])
        self.fest = np.array([sh.is_festival for sh in shows])
        self.set_len = np.array([len(sh.songs) for sh in shows])
        # first show index each song appeared in
        first = np.where(self.P.any(0), self.P.argmax(0), n)
        self.first_date = np.array([self.dates[f] if f < n else 0 for f in first])

    def target_of(self, i: int) -> Target:
        sh = self.shows[i]
        return Target(sh.date, sh.city, sh.country, sh.tour, sh.is_festival)

    def features(self, i: int, t: Target) -> tuple[pd.DataFrame, np.ndarray]:
        """Features for every candidate song at a target show, using shows [0, i).

        Returns (feature frame with a 'song' column, candidate column indices)."""
        P = self.P[:i]
        if i == 0:
            return pd.DataFrame(columns=["song"] + FEATURES), np.array([], dtype=int)
        cand = np.where(P[max(0, i - RECENT_CANDIDATES):].any(0))[0]
        Pc = P[:, cand]
        tord = t.date.toordinal()

        def rate(rows: np.ndarray) -> np.ndarray:
            return Pc[rows].mean(0) if rows.any() else np.zeros(len(cand))

        idx = np.arange(i)
        f: dict[str, np.ndarray] = {}
        for k in (5, 10, 25, 100):
            f[f"rate_last{k}"] = Pc[max(0, i - k):].mean(0)

        tour_rows = (self.tours[:i] == t.tour) & (t.tour != "")
        f["rate_tour"] = rate(tour_rows)
        f["n_tour_prior"] = np.full(len(cand), tour_rows.sum())
        f["tour_is_new"] = np.full(len(cand), float(tour_rows.sum() < 3))
        f["no_tour"] = np.full(len(cand), float(t.tour == ""))  # one-off show, not part of a tour

        played_any = Pc.any(0)
        last_idx = np.where(played_any, i - 1 - np.argmax(Pc[::-1], 0), -10_000)
        f["log_shows_since"] = np.log1p(np.minimum(i - 1 - last_idx, 300))
        f["played_prev"] = Pc[-1]

        # multi-night run: consecutive previous shows in the same city within a few days
        run_rows = np.zeros(i, dtype=bool)
        ref = tord
        for r in range(i - 1, -1, -1):
            if self.cities[r] == t.city.lower() and ref - self.dates[r] <= MULTI_NIGHT_DAYS and t.city:
                run_rows[r] = True
                ref = self.dates[r]
            else:
                break
        f["same_run"] = np.full(len(cand), float(run_rows.any()))
        f["played_in_run"] = Pc[run_rows].max(0) if run_rows.any() else np.zeros(len(cand))

        recent = self.dates[:i] >= tord - 3 * 365
        f["is_festival"] = np.full(len(cand), float(t.is_festival))
        f["rate_festival"] = rate(recent & self.fest[:i])
        f["rate_headline"] = rate(recent & ~self.fest[:i])
        country_rows = recent & (self.countries[:i] == t.country)
        f["rate_country"] = rate(country_rows)
        f["n_country_prior"] = np.full(len(cand), country_rows.sum())
        f["log_days_since_debut"] = np.log1p(np.clip(tord - self.first_date[cand], 0, 20 * 365))
        f["log_gap_days"] = np.full(len(cand), np.log1p(max(tord - self.dates[i - 1], 0)))

        df = pd.DataFrame(f)[FEATURES]
        df.insert(0, "song", [self.songs[j] for j in cand])
        return df, cand

    def song_stats(self, cand: np.ndarray, i: int, lookback: int = 25) -> pd.DataFrame:
        """Ordering info from the last `lookback` plays of each song (before show i)."""
        rows = []
        for j in cand:
            played = np.where(self.P[:i, j] > 0)[0][-lookback:]
            if len(played) == 0:
                rows.append((np.nan, 0.0, 0.0, 0.0))
                continue
            pos = self.pos[played, j]
            rows.append((
                float(np.nanmean(pos)),
                float(self.enc[played, j].mean()),
                float((pos == 0).mean()),
                float((pos == 1).mean()),
            ))
        return pd.DataFrame(rows, columns=["avg_pos", "encore_rate", "opener_rate", "closer_rate"])

    def training_table(self, start: int = 15) -> pd.DataFrame:
        frames = []
        for i in range(start, len(self.shows)):
            df, cand = self.features(i, self.target_of(i))
            if len(cand) == 0:
                continue
            df["label"] = self.P[i, cand]
            df["show_i"] = i
            frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
