# MLB Modeling

Pitch-level MLB modeling pipeline built on the BallDontLie GOAT tier API.
Stores Statcast-grade plate appearance and pitch data in a local DuckDB database.

## Project Structure

```
mlb-modeling/
├── ingestion/
│   ├── schema.py        # DuckDB schema — run once to initialize DB
│   ├── bdl_client.py    # BallDontLie API wrapper
│   └── ingest.py        # Main ingestion script
├── models/              # Model training scripts (coming soon)
├── notebooks/           # EDA and feature engineering notebooks
├── scripts/             # One-off utility scripts
├── data/
│   ├── mlb_modeling.db  # DuckDB database (gitignored)
│   ├── raw/             # Raw JSON responses (gitignored)
│   └── parquet/         # Exported parquet files (gitignored)
├── .github/workflows/
│   └── daily_ingest.yml # Scheduled daily pull
├── .env.example
├── .gitignore
└── requirements.txt
```

## Setup

### 1. Clone and create virtual environment

```bash
git clone https://github.com/YOUR_USERNAME/mlb-modeling.git
cd mlb-modeling

python -m venv .venv

# Mac/Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your BDL_API_KEY
```

### 3. Initialize the database

```bash
cd ingestion
python schema.py
```

### 4. Run your first ingestion

```bash
# Yesterday's games (default)
python ingest.py

# Specific date
python ingest.py --date 2026-04-12

# Full season backfill (~2,430 games, takes a while)
python ingest.py --season 2025

# Date range
python ingest.py --start-date 2026-04-01 --end-date 2026-04-13
```

## GitHub Actions (automated daily pulls)

1. Push repo to GitHub
2. Go to **Settings → Secrets and variables → Actions**
3. Add secret: `BDL_API_KEY` = your key
4. Workflow runs daily at 8 AM ET automatically

You can also trigger manually from the Actions tab with an optional date override.

## Key Tables

| Table | Description |
|---|---|
| `games` | One row per game, scores, venue |
| `plate_appearances` | One row per PA, batter/pitcher/situation |
| `pitches` | One row per pitch — Statcast-grade |
| `player_splits` | Pre-aggregated L/R, home/away, arena splits |
| `betting_odds` | Market lines at time of recording |
| `player_props` | Player prop lines |
| `model_predictions` | Your model outputs + results tracking |

## Example Query

```python
import duckdb

con = duckdb.connect('data/mlb_modeling.db')

# Pitcher K rate by handedness matchup
df = con.execute("""
    SELECT
        pa.pitcher_id,
        pa.batter_side,
        COUNT(*) as total_pa,
        SUM(CASE WHEN pa.result = 'Strikeout' THEN 1 ELSE 0 END) as strikeouts,
        ROUND(SUM(CASE WHEN pa.result = 'Strikeout' THEN 1 ELSE 0 END) * 1.0
              / COUNT(*), 3) as k_rate
    FROM plate_appearances pa
    GROUP BY pa.pitcher_id, pa.batter_side
    ORDER BY k_rate DESC
""").df()
```
