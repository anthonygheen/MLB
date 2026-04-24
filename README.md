# MLB K Prop Modeling Pipeline

Pitch-level MLB pitcher strikeout prop model built on the BallDontLie GOAT tier API.
Predicts starter K totals using Statcast-grade stuff metrics, rolling performance,
opposing lineup K rates, and park factors. Compares projections against book lines to find edges.

**Live dashboard:** https://anthonygheen.github.io/MLB/

---

## How It Works

**The edge:** Books price K props primarily off recent K totals and ERA. This model prices them off *stuff quality* — velocity, spin rate, induced vertical break, and whiff rate per pitch type, z-scored against league average. A pitcher can have a bad box score with elite underlying metrics. That divergence is where the edge lives.

**Target:** Strikeouts recorded by the starting pitcher in a game.

**Model:** Gradient Boosting regressor with grid search tuning. Evaluated with time-series cross-validation only — always train on past, test on future.

**Results:** 1.74 MAE, 64.4% over/under accuracy on 2025 holdout (5,089 starts).

---

## Project Structure

```
mlb-modeling/
├── ingestion/
│   ├── schema.py           # DuckDB schema — run once to initialize
│   ├── bdl_client.py       # BallDontLie API wrapper (rate limiting, pagination)
│   ├── ingest.py           # Historical + daily game/PA ingestion
│   ├── ingest_props.py     # Daily pitcher K prop line ingestion
│   └── sync_players.py     # Syncs active player names/teams into DB
├── features/
│   ├── pitcher_features.py # Stuff grades, rolling metrics, park factors
│   └── lineup_features.py  # Opposing lineup K rate by pitcher hand
├── models/
│   ├── k_prop_model.py     # Training, CV, grid search, evaluation
│   ├── negbinom_model.py   # Negative Binomial comparison model
│   └── saved/              # Saved model pkl files
├── evaluate/
│   └── edge_finder.py      # Backtest and live edge detection vs book lines
├── scripts/
│   ├── predict_today.py    # Pre-game predictions using confirmed lineups
│   └── generate_data.py    # Exports JSON files for GitHub Pages dashboard
├── docs/
│   ├── index.html          # GitHub Pages dashboard (vanilla JS, dark theme)
│   └── data/               # JSON data files updated each prediction run
├── data/                   # DuckDB database (gitignored)
├── .github/workflows/
│   ├── daily_ingest.yml    # Automated daily game/PA pull at 8 AM ET
│   └── predict.yml         # Manual trigger: props + predictions + dashboard update
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

Each season takes 45-90 minutes. Fully resumable — rerun the same command if interrupted.

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

Run these each day around 2-3 PM ET after lineups are confirmed:

```bash
# 1. Pull yesterday's completed games (also runs automatically at 8 AM via GitHub Actions)
python ingestion/ingest.py

# 2. Pull today's K prop lines from BDL
python ingestion/ingest_props.py

# 3. Generate pre-game predictions (uses confirmed lineups from BDL lineups endpoint)
python scripts/predict_today.py

# 4. Export JSON and update the live dashboard
python scripts/generate_data.py

# 5. Push dashboard data
git add docs/data/
git commit -m "Predictions $(date +%Y-%m-%d)"
git push
```

Or trigger the full workflow manually from GitHub Actions → K Prop Predictions → Run workflow.

---

## GitHub Actions

| Workflow | Schedule | Trigger |
|---|---|---|
| Daily MLB Ingestion | 8 AM ET daily | Auto + manual |
| K Prop Predictions | Manual only | Actions tab → Run workflow |

Setup: Settings → Secrets → Actions → add `BDL_API_KEY`

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
| `player_props` | Book lines (pitcher_strikeouts market) |
| `model_predictions` | Model outputs + actual results for accuracy tracking |

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

- [ ] Live deviation monitor — in-game stuff vs baseline, update projection before line moves
- [ ] Accuracy tracker — log predictions vs actual results to model_predictions table
- [ ] Batter props model — hits/total bases using exit velocity and barrel rate
- [ ] RNN model — PA-level sequence model for K probability per plate appearance
- [ ] Webhook integration — BDL ALL-ACCESS real-time K tracking
