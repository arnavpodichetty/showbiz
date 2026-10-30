# Showbiz

Predicts a band's next setlist from their past shows on setlist.fm.

## What it does

- **Search any artist.** It downloads up to their 1,000 most recent setlists and learns how they build a set.
- **Shows the predicted setlist** as a marker-written sheet taped to the stage, in order, with the encore below a line. Songs they almost always play are written firmly; gambles are light and scrawled. Each song shows its chance of being played.
- **Shows swaps.** Songs that take turns in the same spot are written beside each other ("or Stockholm Syndrome").
- **Fits the prediction to the show.** Optionally set the date, city, country, tour (or a new tour, or no tour), and whether it's a festival set. Leave the date blank for a typical show.
- **Flags deep cuts:** the chance they play at least one rarely played song.
- **Grades itself.** Save a prediction before the show, and once the real setlist is posted it's scored and compared with a plain count of what the band has been playing.

## The model

Showbiz ships with a model already trained on 100 artists across many genres (`model/showbiz.cbm`), so predictions are good from the first search. It holds learned patterns only, no setlists. Your copy keeps using it until your own database has more setlists than it was trained on, then it starts retraining on your data. To rebuild it from your database: `.venv/bin/python -m backend.export_model`.

## Run it

1. Get a free API key from [setlist.fm](https://api.setlist.fm/docs/1.0/index.html).
2. Set up and start the app:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
echo "SETLISTFM_API_KEY=your-key" > .env
.venv/bin/uvicorn backend.app:app
```

3. Open http://localhost:8000.

## Publishing it

Visitors can use their own free setlist.fm API key: the **API key** link in the header saves it in their browser only, and the server uses it for their requests without storing it.

- On the public server, don't set `SETLISTFM_API_KEY`, so visitors use their own keys rather than yours. Artists already downloaded work without a key.
- Serve it over HTTPS, since keys are sent with each request.
- Each browser gets an anonymous ID, so searches and saved predictions stay private to each visitor. Downloaded setlists and the model are shared, so everyone's searches improve it.

