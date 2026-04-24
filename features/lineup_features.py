"""
features/lineup_features.py
----------------------------
Builds opposing lineup K-rate features for each pitcher start.

For a given game, the "opposing lineup" is the set of batters the pitcher
will face. We compute their recent K rates, split by pitcher handedness,
weighted by lineup position (leadoff batters get more PAs than 9-hole).

Output: one row per (pitcher_id, game_id) with lineup K-rate features.
"""

import os
import duckdb
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv("DB_PATH", "../data/mlb_modeling.db")


# Lineup position PA weights — leadoff sees ~15% more PAs than 9-hole
LINEUP_WEIGHTS = {1: 1.15, 2: 1.10, 3: 1.08, 4: 1.05, 5: 1.02,
                  6: 0.98, 7: 0.95, 8: 0.92, 9: 0.88}


# ------------------------------------------------------------------
# Per-batter rolling K rate
# ------------------------------------------------------------------

BATTER_ROLLING_K_SQL = """
WITH batter_games AS (
    SELECT
        pa.batter_id,
        pa.game_id,
        pa.batter_side,
        pa.pitcher_hand,
        g.game_date::DATE AS game_date,
        g.season,
        COUNT(*) AS pa_count,
        SUM(CASE WHEN pa.result = 'Strikeout' THEN 1 ELSE 0 END) AS k_count
    FROM plate_appearances pa
    JOIN games g ON pa.game_id = g.game_id
    WHERE g.season_type = 'regular'
    GROUP BY pa.batter_id, pa.game_id, pa.batter_side, pa.pitcher_hand,
             g.game_date, g.season
)
SELECT *,
    k_count * 1.0 / NULLIF(pa_count, 0) AS game_k_rate
FROM batter_games
ORDER BY batter_id, game_date
"""


def get_batter_rolling_k_rates(con, window_days: int = 30) -> pd.DataFrame:
    """
    For each batter, compute rolling K rate over the last N days,
    split by opposing pitcher hand (L vs R).

    window_days=30 is the default — roughly 25-30 games.
    """
    df = con.execute(BATTER_ROLLING_K_SQL).df()
    df['game_date'] = pd.to_datetime(df['game_date'])
    df = df.sort_values(['batter_id', 'game_date'])

    results = []

    for batter_id, grp in df.groupby('batter_id'):
        grp = grp.copy().reset_index(drop=True)

        for i, row in grp.iterrows():
            cutoff = row['game_date']
            lookback = cutoff - pd.Timedelta(days=window_days)

            # All PA in window BEFORE this game (no leakage)
            window = grp[(grp['game_date'] >= lookback) & (grp['game_date'] < cutoff)]

            # Overall K rate
            total_pa = window['pa_count'].sum()
            total_k  = window['k_count'].sum()
            k_rate_overall = total_k / total_pa if total_pa > 0 else np.nan

            # vs RHP
            vs_rhp = window[window['pitcher_hand'] == 'R']
            rhp_pa = vs_rhp['pa_count'].sum()
            rhp_k  = vs_rhp['k_count'].sum()
            k_rate_vs_rhp = rhp_k / rhp_pa if rhp_pa > 0 else k_rate_overall

            # vs LHP
            vs_lhp = window[window['pitcher_hand'] == 'L']
            lhp_pa = vs_lhp['pa_count'].sum()
            lhp_k  = vs_lhp['k_count'].sum()
            k_rate_vs_lhp = lhp_k / lhp_pa if lhp_pa > 0 else k_rate_overall

            results.append({
                'batter_id':      batter_id,
                'game_id':        row['game_id'],
                'game_date':      row['game_date'],
                'batter_side':    row['batter_side'],
                'k_rate_overall': round(k_rate_overall, 4) if not np.isnan(k_rate_overall) else None,
                'k_rate_vs_rhp':  round(k_rate_vs_rhp, 4),
                'k_rate_vs_lhp':  round(k_rate_vs_lhp, 4),
                'recent_pa':      int(total_pa),
            })

    return pd.DataFrame(results)


# ------------------------------------------------------------------
# Lineup K rate for a given game
# ------------------------------------------------------------------

GAME_LINEUPS_SQL = """
-- Reconstruct batting order from actual PA sequence
-- Uses pa_number and inning order to infer lineup position
WITH ordered_pa AS (
    SELECT
        pa.game_id,
        pa.batter_id,
        pa.half_inning,
        pa.inning,
        pa.pa_number,
        g.home_team_id,
        g.away_team_id,
        -- Infer team side: top inning = away team batting
        CASE WHEN pa.half_inning = 'top' THEN g.away_team_id
             ELSE g.home_team_id END AS batting_team_id,
        ROW_NUMBER() OVER (
            PARTITION BY pa.game_id, pa.half_inning
            ORDER BY pa.pa_number
        ) AS appearance_order
    FROM plate_appearances pa
    JOIN games g ON pa.game_id = g.game_id
),
lineup_positions AS (
    -- First 9 unique batters in each half = the lineup
    SELECT DISTINCT
        game_id,
        batting_team_id,
        batter_id,
        MIN(appearance_order) AS first_appearance
    FROM ordered_pa
    GROUP BY game_id, batting_team_id, batter_id
),
ranked_lineups AS (
    SELECT *,
        ROW_NUMBER() OVER (
            PARTITION BY game_id, batting_team_id
            ORDER BY first_appearance
        ) AS lineup_position
    FROM lineup_positions
)
SELECT *
FROM ranked_lineups
WHERE lineup_position <= 9
"""


def get_game_lineups(con) -> pd.DataFrame:
    """Inferred lineup positions for each game from actual PA order."""
    return con.execute(GAME_LINEUPS_SQL).df()


def build_lineup_features(con, batter_k_rates: pd.DataFrame,
                          lineups: pd.DataFrame) -> pd.DataFrame:
    """
    For each game, compute the opposing lineup's weighted K rate
    against the starting pitcher's handedness.

    Returns: DataFrame with (pitcher_id, game_id, lineup_k_rate_weighted, ...)
    """
    # Get pitcher hand per game from plate_appearances
    pitcher_hand_sql = """
        SELECT DISTINCT
            pa.game_id,
            pa.pitcher_id,
            pa.pitcher_hand,
            -- Identify starter: pitcher with most PA in first 3 innings
            COUNT(*) AS pa_count
        FROM plate_appearances pa
        WHERE pa.inning <= 3
        GROUP BY pa.game_id, pa.pitcher_id, pa.pitcher_hand
        QUALIFY ROW_NUMBER() OVER (PARTITION BY game_id ORDER BY pa_count DESC) = 1
    """
    pitcher_games = con.execute(pitcher_hand_sql).df()

    results = []

    for _, pg in pitcher_games.iterrows():
        game_id    = pg['game_id']
        pitcher_id = pg['pitcher_id']
        p_hand     = pg['pitcher_hand']

        # Get opposing lineup for this game
        # Pitcher is on one team; batters are on the other
        game_lineup = lineups[lineups['game_id'] == game_id].copy()
        if game_lineup.empty:
            continue

        weighted_k_rates = []
        lineup_pa_depth  = []

        for _, batter_row in game_lineup.iterrows():
            batter_id = batter_row['batter_id']
            pos       = batter_row['lineup_position']
            weight    = LINEUP_WEIGHTS.get(pos, 0.95)

            # Get batter's K rate vs this pitcher's hand on this game date
            batter_hist = batter_k_rates[
                (batter_k_rates['batter_id'] == batter_id) &
                (batter_k_rates['game_id'] == game_id)
            ]

            if batter_hist.empty:
                continue

            bh = batter_hist.iloc[0]
            k_rate = bh['k_rate_vs_rhp'] if p_hand == 'R' else bh['k_rate_vs_lhp']

            if k_rate is not None and not np.isnan(k_rate):
                weighted_k_rates.append(k_rate * weight)
                lineup_pa_depth.append(bh['recent_pa'])

        if not weighted_k_rates:
            continue

        results.append({
            'pitcher_id':             pitcher_id,
            'game_id':                game_id,
            'lineup_k_rate_weighted': round(np.mean(weighted_k_rates), 4),
            'lineup_k_rate_raw':      round(np.mean([w / LINEUP_WEIGHTS.get(i+1, 0.95)
                                                      for i, w in enumerate(weighted_k_rates)]), 4),
            'lineup_avg_recent_pa':   round(np.mean(lineup_pa_depth), 1),
            'lineup_batters_with_data': len(weighted_k_rates),
        })

    return pd.DataFrame(results)


# ------------------------------------------------------------------
# Master function
# ------------------------------------------------------------------

def build_lineup_feature_matrix(db_path: str = None) -> pd.DataFrame:
    """
    Builds the full opposing lineup K-rate feature matrix.
    This is joined onto the pitcher feature matrix in the model builder.
    """
    path = db_path or DB_PATH
    con  = duckdb.connect(path, read_only=True)

    print("🔍 Computing batter rolling K rates (this takes a few minutes)...")
    batter_k = get_batter_rolling_k_rates(con)
    print(f"   {len(batter_k):,} batter-game records computed")

    print("📋 Inferring game lineups...")
    lineups = get_game_lineups(con)
    print(f"   {lineups['game_id'].nunique():,} games with lineup data")

    print("🧮 Building lineup K-rate features per game...")
    lineup_features = build_lineup_features(con, batter_k, lineups)
    print(f"   {len(lineup_features):,} pitcher-game lineup features built")

    con.close()
    return lineup_features


if __name__ == "__main__":
    df = build_lineup_feature_matrix()
    print(df.head())
    print(df.describe())
