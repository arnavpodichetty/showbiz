# How Showbiz was built

This file records how the app was made, from the first idea to the current model, including what was tried, what changed and why.

---

## 1. The idea

**Goal:** a website that predicts the setlist for a band's next concert.

Setlist predictors already exist, but they mostly give one "average setlist" for a whole tour. The plan was to beat them in three ways:

- **Predictions for a specific show.** Take into account multi-night runs (bands avoid repeats), festival vs headline sets, the country, and how long since each song was last played.
- **A probability for every song,** instead of a single guess, plus "rotation slots" (songs that swap night to night) and deep-cut alerts.
- **Public scoring.** Save predictions before a show, grade them against the real setlist afterwards, and compare with a simple "most-played songs" baseline.

---

## 2. Checking the data source (setlist.fm API)

- The API key worked for searching artists and setlists.
- `/artist/{id}/setlists` returned **403 Forbidden** for this key, but `/search/setlists?artistMbid=…` returns the same data. The app uses that.
- Calls made too quickly sometimes returned 403. The client waits 0.6 s between calls and retries with backoff.
- The key is stored in `.env`, which is never committed (see section 8).

---

## 3. First version

### Stack
- **Backend:** Python with FastAPI.
- **Storage:** SQLite (`data/setlists.db`). It caches every downloaded setlist and every saved prediction.
- **Frontend:** plain HTML, CSS and JavaScript, served by the backend.

### Data
- For each artist, the app downloads up to the 1,000 most recent setlists, 20 per request. Later updates only fetch new pages.
- Setlists with fewer than 3 songs are skipped (usually incomplete). Songs marked as tape are skipped, and a reprise counts once.
- Festivals are detected from the venue or tour name (for example "festival", "Glastonbury" or "Lollapalooza").

### Framing it as machine learning
- Each row is one (song, show) pair, labelled **1 if the song was played** and **0 if not**.
- Candidate songs for a show are any songs played in the band's previous 60 shows.
- **Every feature is calculated only from shows before that show,** so nothing from the future leaks in. The same code is used for training, testing and predicting.

### Features (19 in total)
| Feature | Meaning |
|---|---|
| `rate_last5/10/25/100` | Share of the last 5, 10, 25 or 100 shows where the song was played |
| `rate_tour` / `n_tour_prior` | Play rate on this tour so far, and how many shows the tour has had |
| `tour_is_new` | The tour has fewer than 3 shows so far |
| `no_tour` | The show is a one-off, not part of a tour (added later, section 6) |
| `log_shows_since` | Shows since the song was last played (log scale) |
| `played_prev` | Played at the previous show |
| `same_run` / `played_in_run` | The show is part of a multi-night run in the same city, and the song was already played earlier in that run |
| `is_festival`, `rate_festival`, `rate_headline` | Festival flag, and the song's play rates at festivals vs headline shows over the last 3 years |
| `rate_country` / `n_country_prior` | Play rate in the same country over the last 3 years |
| `log_days_since_debut` | How long ago the song first appeared (new songs are played a lot) |
| `log_gap_days` | Days since the band's previous show (a long gap often means a new tour) |

### Baseline
- The song's play rate on the current tour.
- If the tour has fewer than 3 shows so far, its play rate over the last 10 shows instead.
- Capped between 1% and 99%.

This is the simple guess any fan could make. The model has to beat it to be worth using.

### First model
- **Model:** scikit-learn's `HistGradientBoostingClassifier` (gradient-boosted trees, like LightGBM), trained separately for each artist.
- **Overfitting protection:** early stopping, L2 regularization, and a minimum of 40 rows per leaf.
- **Testing:** a time-based split. Train on each artist's first 85% of shows and test on the last 15%.
- **Scores:**
  - Brier score and log loss: how good the probabilities are.
  - Calibration curve: when it says 30%, does that happen 30% of the time?
  - Precision at the set size: of the N most likely songs, where N is the real set length, how many were played.
  - Hits in the top 15.

**First result (Muse, tested on 145 shows from 2019 to 2026):**

| | Model | Baseline |
|---|---|---|
| Brier score | **0.060** | 0.073 |
| Log loss | **0.202** | 0.250 |
| Top picks played | **91%** | 90% |

The model was about 17% better on Brier score, and its calibration was close to perfect.

### Turning probabilities into a setlist
- **Set length:** the median for this tour and show type (festival or headline).
- **Order:** the top N songs, sorted by their usual position in the set. Songs played in the encore at least half the time go after the encore line.
- **Rotation slots:** pairs of songs that are each played often this tour but rarely in the same show (a co-occurrence analysis), grouped together.
- **Deep-cut alert:** the chance of at least one song played in fewer than 10% of the last 100 shows, which is 1 − the chance that none of them are played.

### Scoreboard
- "Save this prediction" stores the full prediction.
- Once the real setlist is on setlist.fm, it's graded on Brier score and on how many predicted songs were played, for both the model and the baseline.

### Early fixes
- The default tour was a 7-song promo show, so sets came out too short. **Fix:** the default is now the most recent tour with at least 5 shows.
- Rotation detection was too strict and found nothing. **Fix:** looser thresholds, using this tour's shows when there are enough. It then found real swaps, such as Map of the Problematique / Stockholm Syndrome / Thought Contagion.
- My own test predictions were deleted from the scoreboard so it started clean.

---

## 4. Explaining the model

- **"This tour"** is a plain count: the share of this tour's shows where the song was played.
- **The baseline** is that count, capped between 1% and 99%. With fewer than 3 tour shows, it uses the last 10 shows instead.
- **The model** starts from the same counts and adds context (multi-night runs, festivals, time since the song was last played, the country, how new the song is). It learns from the band's past shows how much each one matters.
- The baseline's job is to show the model is worth using. "91% of picks were played" sounds impressive until you see that the plain count gets 90%.

---

## 5. Redesign: the stage-floor setlist

The first interface was a generic dark dashboard with cards. It was redesigned from a detailed brief.

**The idea.** The page centres on the most recognizable object in concert culture: the setlist sheet, written in marker and gaff-taped to the stage floor.
- How firmly each song is written shows its confidence, using the variable axes of the Shantell Sans font:
  - **85% or more:** heavy and steady.
  - **55–85%:** quicker and lighter.
  - **Under 55%:** light and scrawled.
- Swap songs are written beside the main one ("or Stockholm Syndrome").
- The encore sits under a hand-drawn line.

**Tokens**
| Name | Hex | From |
|---|---|---|
| Deck | `#3B3632` | Painted plywood stage deck |
| Paper | `#EEF0EC` | Copy paper under cool stage light |
| Ink | `#1B1A24` (plus lighter `#3A3846` and `#5C5A68`) | Black marker |
| Gaff | `#24221F` | Matte black gaffer tape |
| Congo Blue | `#33228F` | Lee 181 lighting gel, used as the light wash |
| Spike | `#F2C318` | Yellow spike tape, used only for focus and the selected item |

- **Fonts:** Shantell Sans for the sheet only; Overpass (based on US highway-sign lettering) for everything else.
- **Type scale:** Bringhurst's 13 / 16 / 21 / 24 / 36 / 48 px.
- **Contrast:** every text colour passes WCAG AA. The faintest ink is 5.9:1 against the paper.

**Layout.** On desktop, a quiet column of controls sits on the left and the sheet is on the right. On phones, the sheet comes straight after the artist name.

**Avoiding generic looks.**
- The first idea for the floor was near-black with a yellow accent. It became a specific warm-grey stage deck with a blue light wash.
- Cards, all-caps headings, monospace numbers and "A · B · C" lines were all removed.

**Motion.** One moment: the sheet is slapped down, then each song is written in, then the encore line is drawn. This is turned off for people who prefer reduced motion.

**Fixed after reviewing screenshots:**
- a grid bug that pushed the form down
- song numbers too small to read
- paper corners showing past the tape
- swap songs widening the page on phones

The plywood seam lines were removed because they looked like layout grid lines.

**Copy.** Written from a fan's point of view: "Who are you seeing?", "Predict setlist", "Past predictions". Error messages say how to fix the problem.

---

## 6. Changes from using it

- **Old tours (Cigarettes After Sex hadn't toured since March 2025).** The prediction was quietly "their last tour's set." The sheet now says so when the last show was more than 4 months earlier.
- **Tape overlapping the artist name** at mid-size window widths. Fixed with extra space above the sheet.
- **Date is optional.** Leave it blank for "a typical show," meaning the next show after their latest one.
- **Shows that aren't part of a tour.**
  - Added a "Not part of a tour" option, and a `no_tour` feature so the model learns from past one-off shows.
  - Found and fixed a bug: "A new tour" had been quietly falling back to the last tour.
- **City and country examples** changed from London/GB to Los Angeles/US. They were only example text and never used unless typed in.
- **Percentages are always shown** next to every song, instead of only on hover.
- **Removed** the line explaining the handwriting and the accuracy sentence, to keep the page quieter.

---

## 7. Name

Renamed from "Setlist Predictor" to **Showbiz**, which is also the title of Muse's first album. The name was changed in the page title, the wordmark, the README, the server's internal name and the name the app sends to setlist.fm with each request.

---

## 8. Housekeeping decisions

- **The API key is never committed.**
  - It's in `.env`, which is gitignored.
  - Site visitors never need a key, because the server makes every setlist.fm call. Only someone running their own copy needs one.
- **`data/setlists.db` stays local.** It holds personal searches and saved predictions and is gitignored. Each copy of the app builds its own.
- **DESIGN.md was removed** once the theme was finished. The design decisions are summarized in section 5.
- **The README** was cut down to the essentials, then given a short "What it does" section.

---

## 9. Model comparison, round A (7 artists)

**Setup**
- Artists: the 7 already downloaded (Muse, My Chemical Romance, The Strokes, Cigarettes After Sex, beabadoobee, Gorillaz, Interpol).
- **Test blocks:** the shows at 70–85% and 85–100% of each artist's history. Models only ever train on shows before the block they're tested on.
- **Tuning block:** settings are chosen on the 55–70% block, so the test blocks are never used to pick settings.
- **Two ways to train:**
  - *Per-artist:* one model per band.
  - *Pooled:* one model trained on every band.
- **Main score:** Brier score, averaged so each artist counts equally.

**Round 1, eight model types (best 6 plus reference points):**
| Model | Training | Brier |
|---|---|---|
| XGBoost | pooled | **0.0722** |
| CatBoost | pooled | 0.0725 |
| LightGBM | pooled | 0.0726 |
| sklearn HistGB | pooled | 0.0729 |
| Random forest | pooled | 0.0729 |
| Neural net (MLP) | pooled | 0.0739 |
| sklearn HistGB (the site's model at the time) | per-artist | 0.0806 |
| Baseline | — | 0.0902 |

**Round 2, tuned.** CatBoost (depth 4, l2_leaf_reg 10, 400 iterations) scored **0.0719**. LightGBM, XGBoost and HistGB were all at 0.0723. Blends gave no gain, and isotonic calibration made results worse (0.0736).

**Per artist:** pooled, tuned CatBoost beat the old model for all 7 artists. The gains ranged from 1% (beabadoobee) to 16% (My Chemical Romance). For Interpol (139 shows), the old model had done *worse* than the baseline.

**Main lesson:** training on all artists together mattered more than which model was used.

**Change made:** the site switched to one pooled CatBoost model.
- It retrains automatically when any artist's setlists change.
- The minimum for a prediction dropped from 30 to 10 setlists.

**Concern:** 7 artists, almost all rock and alternative, is a small and narrow sample. That led to round B.

---

## 10. Model comparison, round B (17 artists)

10 artists were added to cover different genres and popularity levels. Each was limited to its 500 most recent setlists to save API requests.

| Genre | Artist | Usable setlists |
|---|---|---|
| Pop, stadium | Taylor Swift | 420 |
| Heartland rock | Bruce Springsteen | 470 |
| Metal | Metallica | 486 |
| Latin / reggaeton | Bad Bunny | 265 |
| Jam band | Phish | 496 |
| Hip-hop | Tyler, The Creator | 381 |
| Country | Zach Bryan | 259 |
| K-pop | TWICE | 289 |
| Indie folk | Big Thief | 415 |
| Pop, newer artist | Chappell Roan | 224 |

That makes 17 artists and about 7,100 shows. The same two rounds were rerun with the same rules.

**Round 1, same eight model types:**
| Model | Training | Brier | Top picks played |
|---|---|---|---|
| CatBoost | pooled | **0.0636** | 81.7% |
| LightGBM | pooled | 0.0638 | 81.6% |
| Random forest | pooled | 0.0641 | 81.7% |
| XGBoost | pooled | 0.0641 | 81.5% |
| sklearn HistGB | pooled | 0.0642 | 81.6% |
| Neural net (MLP) | pooled | 0.0652 | 81.4% |
| sklearn HistGB (the original site model) | per-artist | 0.0681 | 81.2% |
| Baseline | — | 0.0848 | 76.5% |

Pooled training still beat per-artist training for every model type except logistic regression, which is too simple to handle very different bands in one model.

**Round 2, tuned:**
| Model | Settings picked on the tuning block | Brier |
|---|---|---|
| CatBoost | depth 8, l2_leaf_reg 10, 400 iterations | **0.0635** |
| LightGBM | 31 leaves, 300 trees | 0.0640 |
| XGBoost | depth 5, 300 trees | 0.0644 |
| sklearn HistGB | 15 leaves | 0.0648 |
| CatBoost + isotonic calibration | | 0.0653 (worse) |
| Blend of CatBoost and LightGBM | | 0.0635 (no real gain) |

**The settings changed.** With 7 artists, the best CatBoost had depth 4. With 17, more varied artists, deeper trees (depth 8) won. A direct comparison on the test blocks:

| CatBoost | Brier | Top picks played |
|---|---|---|
| depth 4 (the site's model after round A) | 0.0646 | 81.2% |
| **depth 8** | **0.0635** | **81.8%** |

Depth 8 was better for 13 of 17 artists and only adds about 2 seconds to training.

**Brier by artist (lower is better):**
| Artist | Baseline | Original site model | CatBoost tuned, pooled |
|---|---|---|---|
| Bad Bunny | 0.0280 | 0.0289 | **0.0223** |
| Big Thief | 0.1111 | **0.0969** | 0.0983 |
| Bruce Springsteen | 0.0848 | 0.0635 | **0.0604** |
| Chappell Roan | 0.1152 | 0.0814 | **0.0661** |
| Cigarettes After Sex | 0.0738 | 0.0730 | **0.0691** |
| Gorillaz | 0.0846 | 0.0583 | **0.0505** |
| Interpol | 0.0940 | 0.0980 | **0.0928** |
| Metallica | 0.1318 | **0.1077** | 0.1094 |
| Muse | 0.0699 | 0.0611 | **0.0550** |
| My Chemical Romance | 0.1200 | 0.1194 | **0.0981** |
| Phish | 0.0688 | 0.0571 | **0.0561** |
| TWICE | 0.0491 | **0.0265** | 0.0278 |
| Taylor Swift | 0.0257 | 0.0071 | **0.0066** |
| The Strokes | 0.1039 | 0.0919 | **0.0815** |
| Tyler, The Creator | 0.1091 | 0.0505 | **0.0440** |
| Zach Bryan | 0.0866 | **0.0731** | 0.0783 |
| beabadoobee | 0.0852 | **0.0625** | 0.0629 |

**What this shows**
- The pooled CatBoost beats the baseline for **all 17 artists** and beats the original site model for 12 of 17.
- Where the original per-artist model wins (Big Thief, Metallica, TWICE, Zach Bryan, beabadoobee), the margins are small. These bands have long histories with their own habits, so a model trained only on them can match a shared one.
- CatBoost against LightGBM is close: CatBoost wins for 12 of 17 artists, but the 95% bootstrap interval for the average gap (−0.0004 to +0.0014) includes zero. CatBoost stays because it is at least as good, needs no extra system libraries on macOS, and handled the new genres better.
- Bands that change their set a lot (Metallica, Big Thief, My Chemical Romance) are the hardest. Bands that play nearly the same show every night (Taylor Swift, TWICE, Bad Bunny) are almost perfectly predictable, so the baseline already does well.

**Change made:** the site's model moved from CatBoost depth 4 to **depth 8** (`CATBOOST_PARAMS` in `backend/model.py`). Everything else stayed the same.

---

## 11. Model comparison, round C (more models, plus speed)

**Why:** with 17 artists there was enough data to test more kinds of model, and to tune with Optuna instead of a small grid. Speed mattered too, so each model was timed.

**Where the time goes on the site**
| Step | Time | When |
|---|---|---|
| Downloading from setlist.fm | about 15–40 s for 500 setlists | The first time an artist is added; later updates only fetch new shows |
| Retraining the pooled model | about 18 s (two fits of about 7.5 s: one for the accuracy check, one for the live model) | Only after data changes |
| Predicting one show | a few milliseconds | Every prediction |

**Speed rule:** a full retrain on every artist has to finish in under **15 seconds**. During tuning, Optuna discarded any setting that would go over this.

**Dropped before testing:** EBM (Explainable Boosting Machine). A single fit on all the data was still running after 10 minutes.

**Models tested** (all pooled, with the same tuning and test blocks as before):
- CatBoost depth 8, the site's current model, as the reference
- CatBoost, LightGBM and XGBoost, each tuned with Optuna (40–60 trials, searching learning rate, tree size, regularization and sampling)
- CatBoost with the artist as an extra input, so it can learn each band's habits
- LightGBM with monotone constraints, so recent play rates can only raise a song's chance and "already played this run" can only lower it
- LightGBM LambdaRank, which learns the order of songs within a show; a small logistic regression then turns rank into a probability
- Spline logistic regression, a very fast simple model with a smooth curve for each feature

Code: `experiments/round3.py`. Optuna and interpret-core were installed for the experiments only; the site doesn't need them.

**Also dropped during the run:** CatBoost with the artist as an extra input. CatBoost handles text inputs such as an artist name with an extra, expensive step. Scoring it was still running after 30 minutes (each fit took well over the speed limit), so it was stopped. The other models were rescored with the Optuna settings already found (`--reuse-params`).

**Optuna tuning:** 40 trials for CatBoost (18 were too slow), 60 for LightGBM (13 too slow), 60 for XGBoost (5 too slow). Tuning took about 67 minutes in total.

**Results** (17 artists, test blocks; retrain = one fit on all 460k rows; predict = one show of about 100 songs)
| Model | Brier | Top picks played | Retrain | Predict |
|---|---|---|---|---|
| **CatBoost + Optuna** | **0.06337** | 81.8% | 5.7 s | 0.5 ms |
| CatBoost depth 8 (the site's model) | 0.06347 | 81.8% | 8.1 s | 0.3 ms |
| XGBoost + Optuna | 0.06374 | 81.7% | 2.4 s | 0.8 ms |
| LightGBM + Optuna | 0.06399 | 81.6% | **1.3 s** | 1.0 ms |
| LightGBM + Optuna, monotone | 0.06467 | 81.2% | 1.1 s | 0.8 ms |
| LightGBM LambdaRank | 0.06768 | 81.6% | 9.2 s | 2.7 ms |
| Spline logistic regression | 0.06848 | 79.3% | 3.9 s | 1.0 ms |

**Against the site's model, artist by artist** (95% bootstrap interval of the average gain; positive means better):
| Model | Better for | Average gain |
|---|---|---|
| CatBoost + Optuna | 9 of 17 | +0.0001 (−0.0003 to +0.0007) |
| LightGBM + Optuna | 6 of 17 | −0.0005 (−0.0015 to +0.0005) |
| XGBoost + Optuna | 5 of 17 | −0.0003 (−0.0013 to +0.0006) |
| LightGBM, monotone | 6 of 17 | −0.0012 (−0.0029 to +0.0005) |
| LambdaRank | 3 of 17 | −0.0042 (−0.0063 to −0.0022), clearly worse |
| Spline logistic regression | 1 of 17 | −0.0050 (−0.0083 to −0.0024), clearly worse |

**What this shows**
- **Tuning has run out of room.** Optuna CatBoost ties the site's model on accuracy (better for 9 of 17, gap well inside the noise), but it retrains about 30% faster because it uses 150 deeper trees instead of 400.
- **The gradient-boosted tree libraries are effectively tied.** CatBoost, XGBoost and LightGBM are all within 1% of each other, and none of the gaps is clearly beyond chance. The real trade-off is speed: LightGBM retrains about 4× faster than CatBoost for about 1% worse accuracy.
- **Monotone rules hurt.** "Played more recently means more likely" isn't always true, because songs rotate in and out of the set. Forcing it made Metallica and TWICE noticeably worse.
- **LambdaRank and splines are clearly worse.** Ranking alone gets the top picks about right but its probabilities are poor. The spline model is too simple to combine features the way trees do.
- **Prediction speed doesn't matter.** Every model predicts a show in under 3 ms. Only retraining time differs.
- Further gains are more likely to come from **new features** (for example a band's habits, such as how much it varies its set) than from new models.

**Decision:** instead of picking one, the site now offers both, with a switch in the show form:
| Option | Model | First prediction after new data (17 artists) | After that |
|---|---|---|---|
| More accurate (default) | CatBoost + Optuna | about 15 s (was about 18 s with depth 8) | about 10 ms |
| Faster | LightGBM + Optuna | about 4 s | about 10 ms |

- The first prediction includes two fits (one for the accuracy check, one for the live model) plus preparing every artist's data.
- The choice is remembered in the browser.
- Each model trains only when someone uses it, so choosing "Faster" never waits on CatBoost.
- LightGBM needs OpenMP on macOS, so `brew install libomp` was added to the README.

---

## 12. Round D: new features, artist weighting and a faster site

Round C showed that model choice and tuning had run out of room, so this round gave the models new information instead, and changed how the site retrains.

### New features
Eight ideas, each added as a group so it could be tested on its own (`NEW_FEATURE_GROUPS` in `backend/features.py`). Every one uses only shows before the target show, like the original 19.

| Group | Features | Idea |
|---|---|---|
| variety | `band_overlap`, `band_pool_ratio` | How much the band changes its set: average overlap between consecutive setlists over the last 10 shows, and distinct songs in the last 10 shows ÷ average set length |
| due | `log_typical_gap`, `gap_ratio` | The song's usual gap between plays, and shows since it was last played ÷ that gap |
| recent | `played_last2`, `played_last3` | Played in either of the last 2 or 3 shows |
| run | `run_night`, `run_plays` | Which night of a multi-night run this is, and how often the song was already played in it |
| city | `has_city_prior`, `played_last_city` | Whether the band has played this city before (outside this run), and whether the song was in that set |
| cover | `is_cover` | setlist.fm marks the song as a cover |
| album | `debut_pct` | How new the song is compared with the rest of the band's catalog |
| weighting | (training weights) | Every artist counts equally. Before, Muse's 963 shows counted about 4× more than Chappell Roan's 224. Two versions were tested: equal, and 1/√rows |

The original 19 features came out exactly the same as before, so the comparison is fair. Building every artist's data with the new features takes 7.6 s in total.

### Step 1: choosing, on the tuning block only
Each version was trained on all artists' shows before the 55–70% block and scored on that block, for both site models (Brier, lower is better):

| Version | More accurate (CatBoost) | Faster (LightGBM) |
|---|---|---|
| Current 19 features | 0.07200 | 0.07188 |
| + variety | **0.07111** | **0.07103** |
| + due | 0.07251 | 0.07200 |
| + recent | 0.07243 | 0.07198 |
| + run | 0.07213 | 0.07196 |
| + city | 0.07279 | 0.07204 |
| + cover | 0.07205 | 0.07194 |
| + album | 0.07377 | 0.07382 |
| All new features | 0.07100 | 0.07129 |
| All minus variety | 0.07290 | 0.07263 |
| All minus due | 0.07137 | 0.07086 |
| All minus recent | 0.07185 | 0.07091 |
| All minus run | 0.07109 | 0.07086 |
| All minus city | 0.07120 | 0.07088 |
| All minus cover | 0.07142 | 0.07204 |
| All minus album | 0.07095 | 0.07086 |
| All + equal artist weight | **0.07066** | 0.07100 |
| All + 1/√rows artist weight | 0.07101 | 0.07128 |

**Rule:** keep a group only if removing it from "all" makes *both* models worse. That kept **variety** and **cover**. Equal artist weighting beat no weighting for both models, so it was kept too.

**Why the others were dropped:** each either helped only one model or nothing at all. "Album" made things clearly worse, probably because it overlaps with the existing "days since debut" feature and lets the model overfit to particular bands. "Due", "recent", "run" and "city" seem to repeat what `log_shows_since`, `played_prev` and `played_in_run` already say.

### Step 2: scoring the choice on the test blocks
| Model | Features | Brier | Top picks played | Retrain |
|---|---|---|---|---|
| More accurate | current 19 | 0.06358 | 81.8% | 5.6 s |
| More accurate | **+ variety, cover, equal weighting** | **0.06202** | **82.1%** | 7.5 s |
| Faster | current 19 | 0.06399 | 81.5% | 1.9 s |
| Faster | **+ variety, cover, equal weighting** | **0.06247** | 81.9% | 2.0 s |

(The current CatBoost scores 0.06358 here instead of 0.06337 in round C because its settings were rounded slightly when they were copied into the site.)

| Model | Better for | Average gain | 95% bootstrap interval |
|---|---|---|---|
| More accurate | 14 of 17 artists | **+2.5%** | +0.0005 to +0.0028 |
| Faster | 12 of 17 artists | **+2.4%** | +0.0006 to +0.0026 |

**What this shows**
- This is the biggest gain since pooling artists (section 9), and much bigger than all the tuning in rounds B and C. The interval doesn't include zero, so it isn't chance.
- The Faster model with the new features (0.06247) now beats the More accurate model with the old ones (0.06358).
- **Biggest gains:** Interpol (−9%), Chappell Roan (−5%), Metallica (−5%) and beabadoobee (−4%). These are smaller or set-shuffling artists, who benefit most from the model knowing "this band varies a lot" and from no longer being outweighed by Muse.
- **Small losses:** Muse (+3%), Phish and Gorillaz (about +1%). This is the cost of equal weighting: the biggest artists get less say in training.

**Change made:** the site uses the original 19 features plus `band_overlap`, `band_pool_ratio` and `is_cover` (22 in total), and weights every artist equally in training. The earlier experiment scripts were pinned to the original 19 features so they still reproduce their results.

### Making the site faster
Retraining was the only slow part (section 11), so the site now avoids waiting for it:
- **Background retraining.** When setlists change, predictions keep using the current model, and a new one trains in the background, replacing it when ready. The current model is still accurate, because each song's features (play rates, last played and so on) are always worked out from the latest shows. Only the very first training, when no model exists yet, makes you wait.
- **Saved to disk:** each artist's prepared data (`data/cache/tables/`) and the trained models (`data/cache/models/`). A restart loads them instead of rebuilding. A saved model is ignored automatically if the features or the model settings change. These files are gitignored with the rest of `data/`.
- **No self-check on the site.** Every retrain used to train twice, once to grade itself and once for real. The grade wasn't shown anywhere, so the site now trains once. The experiments still do the grading.
- **Retrains run one at a time,** so the two models don't compete for the CPU.

| Case (17 artists) | Before | After |
|---|---|---|
| First prediction after new setlists, More accurate | 15.2 s | **0.01 s** (retrains in the background in about 8 s) |
| First prediction after new setlists, Faster | 3.9 s | **0.01 s** |
| First prediction after a server restart | about 15 s | **0.4 s** |
| Very first start, nothing saved | about 15 s | 13.6 s (only ever once) |
| Prediction with an up-to-date model | about 10 ms | about 10 ms |

### Back to one model
With background retraining, both models predicted instantly, so the Faster option no longer saved any waiting. It was only about 1% less accurate. **The switch was removed, and the site now uses only CatBoost**, the more accurate model.
- Removed: the Model switch (HTML, JS, CSS), the model choice in the API, and LightGBM from `requirements.txt`.
- Removed from the README: the macOS `brew install libomp` step. It's still installed on this computer, because the experiment scripts use LightGBM and XGBoost.
- Kept: `experiments/round4.py` keeps its own copy of the LightGBM settings, so round D can be rerun.
- Saved model: `data/cache/models/catboost.joblib`.

---

## 13. Showing a prediction before the download finishes

**Problem:** with retraining in the background, the only wait left was downloading. A new artist took up to about a minute (50 pages of 20 setlists, rate-limited). Even an artist downloaded before waited a few seconds while the site checked setlist.fm for new shows.

**Change:** setlist.fm sends the newest setlists first, and those matter most. So the site now predicts as soon as it has enough of them, and keeps downloading the rest in the background.
- **Ready rule:** at least 100 setlists stored, with at least 30 real shows among them (`READY_SETLISTS` and `READY_SHOWS` in `backend/app.py`). An artist downloaded before is ready immediately.
- **The page** shows the prediction as soon as the artist is ready, with a note: "Still reading older setlists (160 of 1000). The prediction will update when it's done." When the download finishes, it predicts again with everything, keeping whatever is in the form. For an artist downloaded before, it only re-predicts if new setlists actually arrived.
- **Retraining during a download:** every new page changes the data, so a newer retrain request now replaces an older one that hasn't started yet. At most one retrain runs while another waits.
- **Switching artists** mid-download stops the page from tracking the old one.

**Tested** on a separate copy of the server and data (port 8001), so the test artist didn't appear in the real search history:
| Case | Before | After |
|---|---|---|
| New artist (Radiohead, 1,000 setlists) | about 36 s (whole download) | **6.4 s** to the first prediction; updated automatically at 36 s |
| Artist already downloaded (Muse) | a few seconds (checking for new shows first) | **1.1 s** |

Using all of Radiohead's data changed the early prediction only slightly. Most songs stayed the same, and "Lucky" went from 54% to 71%.

**Spotted while testing, and fixed:** for bands with a large rotation pool, like Radiohead, the sheet listed far too many swap songs on one line (about 18), and repeated the same group under several songs. The swap detection links songs in chains, so heavy rotators end up in one huge group. Now (in `writeSheet` in `frontend/app.js`):
- each song on the sheet shows **at most 3** swaps, the most likely first
- a swap is never listed under two songs
- when a group covers several sheet songs, its swaps are shared out between them, starting with the least certain song (the one most likely to be swapped out)

Checked on Phish, Muse and Big Thief: no song has more than 3 swaps and none is listed twice. Big Thief now shows "Contact *or Snail, or True Love*" instead of a long list.

---

## 14. Round E: 100 artists

**Why:** pooling artists was the biggest early win, so the training set was widened from 17 artists to 100, picked to cover as many kinds of music and live show as possible.

**Who was added** (83 artists; Radiohead had already been downloaded through the site):
| Group | Artists |
|---|---|
| Pop | Billie Eilish, Dua Lipa, Harry Styles, Ed Sheeran, Sabrina Carpenter, Lady Gaga, The Weeknd, Lorde, Olivia Rodrigo |
| Classic and arena rock | The Rolling Stones, Paul McCartney, U2, Bob Dylan, Red Hot Chili Peppers, Foo Fighters, Green Day, Guns N' Roses, Depeche Mode, Pearl Jam, Coldplay, The Cure |
| Metal and industrial | Ghost, Tool, Slipknot, Sleep Token, Bring Me the Horizon, Iron Maiden, Rammstein |
| Punk, emo and alt | blink-182, Paramore, The Killers, Weezer, IDLES, Fontaines D.C. |
| Indie and art rock | Radiohead, Arctic Monkeys, Vampire Weekend, The National, Wilco, Phoebe Bridgers, Japanese Breakfast, Florence + the Machine, Lana Del Rey, Tame Impala, Mitski |
| Jam and bluegrass | Dead & Company, Goose, Widespread Panic, Billy Strings, King Gizzard & the Lizard Wizard |
| Country, folk and Americana | Luke Combs, Chris Stapleton, Jason Isbell and the 400 Unit, Tyler Childers, Kacey Musgraves, The Avett Brothers, Hozier, Morgan Wallen, Noah Kahan |
| Hip-hop and R&B | Drake, Travis Scott, J. Cole, SZA, Doja Cat, Kendrick Lamar |
| Latin | Shakira, Peso Pluma, Maná, Karol G |
| East Asia | Stray Kids, SEVENTEEN, BABYMETAL, ONE OK ROCK |
| Electronic | LCD Soundsystem, ODESZA |
| Jazz and instrumental | Khruangbin, Jacob Collier |
| Non-English Europe | Måneskin, Die Toten Hosen, Indochine |
| Africa | Burna Boy |
| Residencies and solo | Billy Joel, John Mayer |

The list mixes fixed shows (Iron Maiden, Rammstein, Coldplay) with heavy rotators (Bob Dylan, Goose, Billy Strings), residencies (Billy Joel), stadium acts and mid-size touring bands, and artists from the 1960s to the 2020s.

**Downloading**
- The first 15 were capped at their 500 most recent setlists, and the other 67 at 260, to stay inside setlist.fm's daily request limit (about 1,440 for a standard key). Recent shows matter most, so the cap costs little.
- The download script looks each name up and checks it matches exactly. It skips artists already done and stops cleanly if setlist.fm starts refusing requests, so it can be resumed. It never needed to.
- All 82 downloads worked. Arctic Monkeys was already partly downloaded (300 setlists, up to their last tour in 2023), so it was left as it was.
- **Total: 100 artists, 33,860 setlists,** about 1.9 million training rows (up from 459,000). 98 artists have enough shows to be tested (at least 80 usable). Peso Pluma and Burna Boy have fewer, but still go into the site's training data.

**Test** (`experiments/round5.py`, the site's model: CatBoost with 22 features and equal artist weighting)

*1. Does training on more artists help the original 17?* Same test blocks as rounds B to D:
| Trained on | Brier | Top picks played |
|---|---|---|
| The 17 original artists | 0.06194 | 82.1% |
| All 98 testable artists | **0.06134** | **82.3%** |

About 1% better, and better for 14 of 17 artists, but the 95% interval (−0.0006 to +0.0015) includes zero. The gain is small and might be chance. Most of the benefit of pooling had already come from the first 17. Interpol (139 shows) got worse (0.086 → 0.093); Chappell Roan, beabadoobee, Metallica and Tyler, The Creator gained the most.

*2. Do the round D features hold up on artists that weren't used to choose them?* The 81 new testable artists. For this check, one model is trained per test block on every artist's earlier shows, so no artist's test shows are ever trained on:
| Version | Brier | Top picks played |
|---|---|---|
| Site features (22, equal artist weight) | **0.05796** | **82.3%** |
| Original 19 features, no weighting | 0.05912 | 81.9% |
| Baseline (tour play count) | 0.08731 | 76.8% |

- Round D's features are **2.0% better** on these new artists, for 58 of 81, with a 95% interval of +0.0007 to +0.0016. That's a real gain, which confirms round D wasn't a fluke of its 17 artists.
- **The model beats the baseline for all 81 new artists,** by 34% on average.
- **Biggest gains from the new features:** fixed-show acts, where "how much the band varies" matters most. Rammstein −33%, Iron Maiden −29%, Sleep Token −12%.
- **Biggest losses:** a few pop acts. Lady Gaga +14%, Tame Impala +8%, Billie Eilish +7%.
- **Easiest artists:** Iron Maiden (Brier 0.003), Dua Lipa, Rammstein, The Weeknd. **Hardest:** Jason Isbell (0.141), LCD Soundsystem, Radiohead, Red Hot Chili Peppers.

**Site changes for the bigger dataset**
- The site's saved training data now keeps only the columns the model uses, in compact number types, about 3× smaller (an artist's table went from 4.6 MB to 1.5 MB). The server uses about 470 MB with 100 artists.
- With 100 artists, a background retrain takes about 26 s. The first start after this change took 32 s while every artist's saved data was converted to the smaller format; that only happens once. Predictions still take about 30 ms.

**Decision:** keep all 100. The accuracy gain for the original 17 is small, but the site now has good predictions ready for 100 artists across many genres, and no part of the test got worse on average.

---

## 15. Shipping the trained model

**Goal:** everyone who runs Showbiz starts with the 100-artist model, instead of a model trained only on the few artists they happen to search.

**How**
- `backend/export_model.py` trains on every artist in the local database and saves `model/showbiz.cbm`, plus `model/showbiz.json` (date, number of artists, setlists and rows, the feature list and the settings). Both are committed to git. The model file is 2.2 MB.
- It uses CatBoost's own file format, not Python's pickle. Loading a pickle can run arbitrary code, which isn't safe for a file people download from a repo; the CatBoost format is just data.
- The model contains learned patterns only: no setlists, and no song or artist names. The training used number arrays without column names.

**The retraining rule:** on a fresh copy, the first artist someone downloads would normally trigger a retrain on that one artist, replacing a model trained on 100 with a much worse one. So a copy uses the shipped model, and doesn't retrain, **until its own database has at least as many setlists as the shipped model was trained on** (33,860). After that, it retrains on its own data as before. The shipped model is also ignored if the feature list in the code changes (it would no longer fit).

**Tested** with a fresh copy on port 8001 with an empty database: it downloaded Peso Pluma (113 setlists in 5.6 s), predicted in 0.04 s using the shipped model, and created no local model. Its predictions matched the main 100-artist copy exactly.

To refresh the shipped model after adding artists: `.venv/bin/python -m backend.export_model`, then commit `model/`.

---

## 16. "Checked before" shows only your own searches

**Problem:** the "Checked before" list under the search box showed every artist in the database, so after round E it listed all 100, including the 82 added only for training.

**Change**
- A new `searched_at` column in the `artists` table is set only when someone picks an artist on the site (`POST /api/sync`). Artists downloaded by the training scripts don't have one.
- "Checked before" (`/api/artists`) now shows the **5 most recently picked** artists, newest first. The full artist list (`store.list_artists`) is unchanged, so training still uses all 100.
- The list started empty, because there was no record of which existing artists had been picked on the site. It fills up as artists are picked.

Tested on a copy of the database: after picking 6 artists, the list showed the last 5, newest first.

---

## 17. Where things stand

**Model:** one CatBoost model, tuned with Optuna (depth 9, 150 trees, Depthwise growth), trained on every downloaded artist together (100 artists, 33,860 setlists) with every artist weighted equally. Brier 0.0613 on the original 17 artists' test blocks (baseline 0.0848), and 0.0580 on the 81 newer ones (baseline 0.0873).
- **Features:** 22, the original 19 plus band variety (2) and covers (1).
- **Speed:** predictions take about 10 ms. When setlists change, the site keeps predicting with the current model while the new one trains in the background (about 26 s with 100 artists). Prepared data and models are saved in `data/cache/`, so a restart is ready in under half a second. A new artist shows a prediction after its newest 100 setlists (about 6 s), and updates when the rest have downloaded.

**Files**
| Path | What it does |
|---|---|
| `backend/setlistfm.py` | setlist.fm API client (rate limiting and retries) |
| `backend/store.py` | SQLite storage for setlists and saved predictions |
| `backend/features.py` | Turns setlists into features (22 used, 32 computed), using only earlier shows |
| `backend/model.py` | The pooled CatBoost model, backtesting, building the setlist, rotation slots, deep cuts, scoring |
| `backend/app.py` | FastAPI server, API endpoints, background retraining, disk cache, shipped-model fallback |
| `backend/export_model.py` | Trains on the local database and saves the model that ships with the site |
| `model/` | The shipped model (`showbiz.cbm`) and what it was trained on (`showbiz.json`) |
| `frontend/` | The stage-floor interface (HTML, CSS, JS) |
| `experiments/compare_models.py` | Round 1: compares model types |
| `experiments/tune_models.py` | Round 2: tuning, blends, calibration, per-artist table |
| `experiments/round3.py` | Round 3: Optuna tuning, new model types, speed timing |
| `experiments/round4.py` | Round D: new feature groups and artist weighting |
| `experiments/round5.py` | Round E: training on 17 vs 100 artists, and a fresh check of round D |
| `experiments/cache/` | Cached training tables, logs and result CSVs (gitignored) |

**Ideas not yet tried**
- *Learning from everyone's searches:* host one shared copy, so every search adds to the same dataset. This needs throttled retraining, per-person privacy for searches and saved predictions, and care with the single API key's daily limit.
- *More band-level features,* since band variety was the biggest win (for example how often the band changes its opener or encore, or how long its tours are).
- *Retuning the models with the new features.* Their settings were picked with the original 19 features.
- *Retrying the dropped groups in other forms,* for example "due" only for songs with a regular rhythm.
- *Hierarchical Bayesian models,* which share information across artists in a more principled way and give uncertainty ranges.
- *Survival models* for "how many shows until this song comes back."
- *Song embeddings or matrix factorization* to learn which songs belong together.
- *Whole-setlist models* (CRF or DPP) that predict the set as a group, so swap songs aren't both picked.
- *Sequence models* (Markov chains) to predict the running order, not just which songs are played.
