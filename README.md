# MLB K Prop Modeling Pipeline

Pitch-level MLB pitcher strikeout prop model built on BallDontLie GOAT tier API data.
Predicts starter K totals using Statcast-grade stuff metrics, rolling performance,
opposing lineup K rates, and park factors. Compares projections against book lines to find edges.

---

## How It Works

**The edge:** Books price K props primarily off recent K totals and ERA. This model prices them off *stuff quality* — velocity, spin rate, induced vertical break, and whiff rate per pitch type, z-scored against league average. A pitcher can have a bad box score with elite underlying metrics. That divergence is where the edge lives.

**Target:** Strikeouts recorded by the starting pitcher in a game.

**Models:** Gradient Boosting (primary) + Poisson GLM. Evaluated with time-series cross-validation only — always train on past, test on future.

**Results:** ~1.75 MAE, ~65% over/under accuracy in CV across 2021-2025 data.

---

## Project Structure

```
mlb-modeling/
├── ingestion/
│   ├── schema.py           # DuckDB schema — run once to initialize
│   ├── bdl_client.py       # BallDontLie API wrapper (rate limiting, pagination)
│   ├── ingest.py           # Historical + daily game/PA ingestion
│   └── ingest_props.py     # Daily pitcher K prop line ingestion
├── features/
│   ├── pitcher_features.py # Stuff grades, rolling metrics, park factors
│   └── lineup_features.py  # Opposing lineup K rate by pitcher hand
├── models/
│   ├── k_prop_model.py     # Training, CV, grid search, evaluation
│   └── saved/              # Saved model pkl files (gitignored)
├── evaluate/
│   └── edge_finder.py      # Backtest and live edge detection vs book lines
├── scripts/
│   └── predict_today.py    # Daily predictions table with over/under flags
├── data/                   # DuckDB database (gitignored)
├── .github/workflows/
│   └── daily_ingest.yml    # Automated daily pull via GitHub Actions
├── .env                    # API key and paths (gitignored)
├── .env.example
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone and create virtual environment

```bash
git clone https://github.com/YOUR_USERNAME/mlb-modeling.git
cd mlb-modeling

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
# Edit .env and add your BDL_API_KEY and set DB_PATH to an absolute path
```

`.env` contents:
```
BDL_API_KEY=your_key_here
DB_PATH=C:\projects\mlb-modeling\data\mlb_modeling.db
RAW_DATA_PATH=data/raw
PARQUET_PATH=data/parquet
```

### 3. Initialize database

```bash
python ingestion/schema.py
```

### 4. Backfill historical data

```bash
python ingestion/ingest.py --season 2021
python ingestion/ingest.py --season 2022
python ingestion/ingest.py --season 2023
python ingestion/ingest.py --season 2024
python ingestion/ingest.py --season 2025
```

### 5. Train the model

```bash
# Cross-validation only (fast)
python models/k_prop_model.py --cv-only

# Train with grid search and save (recommended, ~15-20 min)
python models/k_prop_model.py --tune

# Train with default params and save (fast)
python models/k_prop_model.py
```

---

## Daily Workflow

```bash
# 1. Pull yesterday's completed games
python ingestion/ingest.py

# 2. Pull today's prop lines
python ingestion/ingest_props.py

# 3. Generate predictions and flag edges
python scripts/predict_today.py

# Adjust edge threshold
python scripts/predict_today.py --threshold 0.5
```

---

## GitHub Actions

Game data pulls automatically at 8 AM ET daily.

1. Push repo to GitHub
2. Settings → Secrets → Actions → add `BDL_API_KEY`
3. Workflow runs daily, also manually triggerable from Actions tab

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

## Roadmap

- [ ] Live deviation monitor — in-game stuff vs baseline, update projection before line moves
- [ ] Props ingestion automation via GitHub Actions
- [ ] Batter props model — hits/total bases using exit velocity and barrel rate
- [ ] Backtest dashboard — Plotly Dash showing edge accuracy by threshold
- [ ] Webhook integration — BDL ALL-ACCESS real-time K tracking
