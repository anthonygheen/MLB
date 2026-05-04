# MLB K Prop Modeling Pipeline

Pitch-level MLB pitcher strikeout prop model built on the BallDontLie GOAT tier API.
Predicts starter K totals using Statcast-grade stuff metrics, rolling performance,
opposing lineup K rates, and park factors. Compares projections against sportsbook lines
and Kalshi prediction markets to find edges.

**Live dashboard:** https://anthonygheen.github.io/MLB/

---

## How It Works

**The edge:** Books price K props primarily off recent K totals and ERA. This model prices
them off *stuff quality* — velocity, spin rate, induced vertical break, and whiff rate per
pitch type, z-scored against league average. A pitcher can have a bad box score with elite
underlying metrics. That divergence is where the edge lives.

**Target:** Strikeouts recorded by the starting pitcher in a game.

**Model architecture — two stages:**

1. **GBM regressor** predicts μ (expected Ks) from ~40 features. Tuned via grid search and
   evaluated with time-series cross-validation (train on past, test on future only).

2. **Negative Binomial calibration** — NegBinom(μ, r=41.4250) converts the point estimate
   into a full probability distribution. From that distribution, P(K > line) is read
   directly. The dispersion parameter r was fitted via MLE on the 2026 holdout season
   (April 2026, 243 starts).

**From prediction to bet sizing:**

```
GBM → μ  →  NegBinom(μ, r)  →  P(K > line)  →  EV  →  Half-Kelly fraction
```

- **EV** = P(win) × decimal_payout − P(lose) × 1.0 per $1 risked
- **Half-Kelly** = (P × payout − (1 − P)) / payout / 2, capped at 5% of bankroll
- Bets flagged when EV ≥ 3% per $100 risked (configurable via `--min-ev`)
- Breakeven accuracy at −110 juice: 52.4%

**Results:** 1.74 MAE, 65.5% over/under accuracy (time-series CV).

---

## Project Structure

```
MLB/
├── ingestion/
│   ├── schema.py           # DuckDB schema — run once to initialize
│   ├── bdl_client.py       # BallDontLie API wrapper (rate limiting, pagination)
│   ├── ingest.py           # Historical + daily game/PA ingestion
│   ├── ingest_props.py     # Daily pitcher K prop line ingestion (sportsbooks)
│   ├── ingest_kalshi.py    # Daily Kalshi prediction market ingestion
│   └── sync_players.py     # Syncs active player names/teams into DB
├── features/
│   ├── pitcher_features.py # Stuff grades, rolling metrics, park factors
│   └── lineup_features.py  # Opposing lineup K rate by pitcher hand
├── models/
│   ├── k_prop_model.py     # GBM training, CV, grid search, evaluation
│   ├── negbinom_model.py   # NegBinom calibration layer (MLE fit, P(K>line))
│   └── saved/              # Saved model pkl files
├── scripts/
│   ├── predict_today.py    # Pre-game predictions (props-first, confirmed starters)
│   ├── log_results.py      # Score yesterday's props against actual Ks
│   └── generate_data.py    # Export JSON files for GitHub Pages dashboard
├── docs/
│   ├── index.html          # GitHub Pages dashboard (vanilla JS, dark theme)
│   └── data/               # JSON data files updated each prediction run
│       ├── predictions.json
│       ├── sharp_markets.json
│       ├── stuff_grades.json
│       ├── accuracy.json
│       └── park_factors.json
├── data/                   # DuckDB database (gitignored)
├── run_daily.py            # Single-command daily workflow runner
├── .env                    # API key and paths (gitignored)
├── .env.example
└── requirements.txt
```

---

## Setup

### 1. Clone and create virtual environment

```bash
git clone https://github.com/anthonygheen/MLB.git
cd MLB

python -m venv .venv

# Windows
.venv\Scripts\Activate.ps1

# Mac/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — add BDL_API_KEY and set DB_PATH to an absolute path
```

`.env` contents:
```
BDL_API_KEY=your_key_here
DB_PATH=C:\projects\MLB\data\mlb_modeling.db
RAW_DATA_PATH=data/raw
PARQUET_PATH=data/parquet
```

### 3. Initialize database and sync players

```bash
python ingestion/schema.py
python ingestion/sync_players.py
```

### 4. Backfill historical data

```bash
python ingestion/ingest.py --season 2021
python ingestion/ingest.py --season 2022
python ingestion/ingest.py --season 2023
python ingestion/ingest.py --season 2024
python ingestion/ingest.py --season 2025
```

Each season takes 45–90 minutes. Fully resumable — rerun the same command if interrupted.

### 5. Train the model

```bash
# Cross-validation only (validate before committing)
python models/k_prop_model.py --cv-only

# Train with grid search and save (~15-20 min, recommended)
python models/k_prop_model.py --tune

# Train with default params and save (fast)
python models/k_prop_model.py
```

---

## Daily Workflow

Run once per day after yesterday's games are final (typically morning ET):

```bash
python run_daily.py
```

This executes six steps in order:

| Step | Script | Purpose |
|------|--------|---------|
| 1 | `ingestion/ingest.py` | Pull yesterday's completed games + plate appearances |
| 2 | `ingestion/ingest_props.py` | Pull today's K prop lines from sportsbooks |
| 3 | `ingestion/ingest_kalshi.py` | Pull today's Kalshi threshold markets (non-fatal) |
| 4 | `scripts/log_results.py` | Score yesterday's props against actual Ks |
| 5 | `scripts/predict_today.py` | Generate today's predictions |
| 6 | `scripts/generate_data.py` | Write dashboard JSON |

**Flags:**
```bash
python run_daily.py --date 2026-05-02   # run as if it were this date
python run_daily.py --min-ev 5.0        # raise EV threshold for predictions
python run_daily.py --skip-ingest       # skip game ingestion (already done)
```

After running, push the updated data to deploy to the dashboard:

```bash
git add docs/data/
git commit -m "Predictions YYYY-MM-DD"
git push
```

---

## Feature Groups

| Group | Features | Source |
|---|---|---|
| Rolling form | K rate, SwStr%, Zone%, BB rate, raw Ks — last 3/5/10 starts | plate_appearances |
| Stuff grades | Velo, spin, IVB, whiff rate z-scored vs league avg by pitch type | pitches |
| Pitch mix | % FF/SL/CH/CU/SI last 3 starts | pitches |
| Lineup | Opposing lineup K rate vs pitcher hand, weighted by lineup position | plate_appearances |
| Context | Park K factor, days rest, pitcher hand, month | games |

---

## Key Tables

| Table | Description |
|---|---|
| `games` | One row per game |
| `plate_appearances` | One row per PA — batter, pitcher, situation, result |
| `pitches` | One row per pitch — full Statcast fields |
| `players` | Player names, positions, current teams |
| `player_props` | Book lines and Kalshi markets (book column distinguishes source) |
| `model_predictions` | Model outputs + actual results for accuracy tracking |

---

## Dashboard Guide

The live dashboard at https://anthonygheen.github.io/MLB/ has five tabs.

### Predictions

The main tab. Shows today's model predictions for every pitcher with a prop line posted.

**Top Plays strip** — the three highest-confidence bets by Kelly fraction, shown at the top
of the page. Each card shows the pitcher, line, recommended side, Kelly %, and dollar amount
based on your bankroll.

**Bankroll input** — enter your bankroll in the top-right corner. Dollar amounts throughout
the dashboard scale automatically.

**Prediction cards** — one card per pitcher, showing:
- Model's μ (expected Ks) and the book's line
- Over/Under odds from sportsbooks
- Model's P(over) and P(under)
- EV per $100 risked on each side
- Kelly fraction and dollar amount for each side

**Flagged bets** — cards with a colored banner at the top have EV ≥ threshold. The banner
shows the recommended side, Kelly %, and dollar amount.

**Filters:**
- EDGE — shows only flagged bets, sorted by Kelly fraction descending
- OVER / UNDER — filter by direction
- Search — filter by pitcher name

### Sharp Markets

Kalshi prediction market analysis. Kalshi works differently from sportsbooks — instead of
a single Over/Under line, it offers binary contracts at every integer threshold (e.g., 3+,
4+, 5+ Ks), each with its own implied probability.

**Per-pitcher tables** show every liquid Kalshi threshold for that pitcher. For each threshold:
- **YES / NO** — buying YES means you win if the pitcher hits that K total; NO means you win
  if they fall short
- **Kalshi price** — the market's implied probability (e.g., 0.62 = 62% chance)
- **Model P** — the NegBinom model's probability for the same outcome
- **Edge** — model P minus Kalshi implied P (positive = model sees value)
- **EV** — expected value per $1 risked
- **Kelly %** — half-Kelly bet sizing
- Rows highlighted green have positive EV; greyed rows at 55% opacity have negative EV

**Flagged** markets have EV above threshold and are sorted to the top.

**Source filter** — buttons to filter by market source (Kalshi; extensible to ProphetX etc.).

### Stuff Grades

Current pitcher arsenal grades: velocity, spin rate, induced vertical break, and whiff rate
per pitch type, each z-scored vs. league average for that pitch type. A grade above zero
means the pitcher is above average on that metric. Useful for identifying pitchers whose
underlying stuff has improved or declined before the box score catches up.

### Accuracy

Historical model performance: predicted vs. actual Ks, over/under hit rate, MAE, and
calibration by confidence tier. Use this tab to assess whether model edges are holding over
time. The breakeven accuracy at −110 juice is 52.4% — anything above that on a sufficient
sample is profitable.

### Park Factors

K rate multipliers by ballpark, derived from historical plate appearance data. A park factor
above 1.0 means the park suppresses contact and inflates strikeouts relative to league
average. The model applies these factors automatically; this tab exposes them for reference.

---

## Maintenance

```bash
# Sync player names/teams weekly (catches trades and callups)
python ingestion/sync_players.py

# Retrain model monthly or after significant data accumulation
python models/k_prop_model.py --tune
```

---

## Roadmap

- [ ] ProphetX / sharp book integration — add as additional source in Sharp Markets tab
- [ ] Live deviation monitor — in-game stuff vs baseline, update projection before line moves
- [ ] Batter props model — hits/total bases using exit velocity and barrel rate
- [ ] RNN model — PA-level sequence model for K probability per plate appearance
- [ ] Webhook integration — BDL ALL-ACCESS real-time K tracking
