"""
features/pitcher_features.py
-----------------------------
Builds per-start pitcher feature vectors from the pitches + plate_appearances tables.

For each (pitcher_id, game_id) where the pitcher started, we compute:
  - Rolling performance metrics (last 3/5/10 starts)
  - Stuff grades per pitch type (velocity, spin, movement z-scores)
  - Command metrics (zone%, chase%, swstr%)
  - Pitch mix (% of each pitch type last 3 starts)
  - Fatigue (days rest, pitch count last outing)
  - Park and handedness context

Output: DataFrame with one row per pitcher-start, ready for model training.
"""

import os
import duckdb
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv("DB_PATH", "../data/mlb_modeling.db")


def get_con():
    return duckdb.connect(DB_PATH, read_only=True)


# ------------------------------------------------------------------
# Step 1: Build the per-game pitcher performance table
# One row per (pitcher_id, game_id) — actual outcomes for training labels
# ------------------------------------------------------------------

PITCHER_GAME_RESULTS_SQL = """
WITH pa_stats AS (
    -- PA-level aggregations only — no pitch join here
    SELECT
        pa.pitcher_id,
        pa.game_id,
        g.game_date::DATE                                           AS game_date,
        g.season,
        g.venue,
        pa.pitcher_hand,
        COUNT(*)                                                    AS batters_faced,
        SUM(CASE WHEN pa.result = 'Strikeout' THEN 1 ELSE 0 END)   AS strikeouts,
        SUM(CASE WHEN pa.result IN ('Walk','Intent Walk') THEN 1 ELSE 0 END) AS walks,
        SUM(CASE WHEN pa.result = 'Home Run' THEN 1 ELSE 0 END)    AS hr_allowed,
        SUM(CASE WHEN pa.result IN (
            'Single','Double','Triple','Home Run'
        ) THEN 1 ELSE 0 END)                                        AS hits_allowed
    FROM plate_appearances pa
    JOIN games g ON pa.game_id = g.game_id
    WHERE g.season_type = 'regular'
    GROUP BY pa.pitcher_id, pa.game_id, g.game_date, g.season, g.venue, pa.pitcher_hand
    HAVING COUNT(*) >= 12
),
pitch_stats AS (
    -- Pitch-level aggregations only — joined separately
    SELECT
        pa.pitcher_id,
        pa.game_id,
        COUNT(*)                                                    AS total_pitches,
        MAX(p.pitcher_pitch_count)                                  AS pitch_count,
        SUM(CASE WHEN p.pitch_call = 'S' THEN 1 ELSE 0 END)        AS swinging_strikes,
        SUM(CASE WHEN p.plate_x BETWEEN -0.83 AND 0.83
                  AND p.plate_z BETWEEN p.strike_zone_bottom AND p.strike_zone_top
                  AND p.plate_x IS NOT NULL
             THEN 1 ELSE 0 END)                                     AS in_zone_pitches
    FROM pitches p
    JOIN plate_appearances pa ON p.pa_id = pa.pa_id
    JOIN games g ON pa.game_id = g.game_id
    WHERE g.season_type = 'regular'
    GROUP BY pa.pitcher_id, pa.game_id
)
SELECT
    pa.*,
    ps.total_pitches,
    ps.pitch_count,
    ps.swinging_strikes,
    ps.in_zone_pitches,
    ROUND(pa.strikeouts * 1.0 / NULLIF(pa.batters_faced, 0), 4)        AS k_rate,
    ROUND(pa.walks * 1.0 / NULLIF(pa.batters_faced, 0), 4)             AS bb_rate,
    ROUND(ps.swinging_strikes * 1.0 / NULLIF(ps.total_pitches, 0), 4)  AS swstr_pct,
    ROUND(ps.in_zone_pitches * 1.0 / NULLIF(ps.total_pitches, 0), 4)   AS zone_pct
FROM pa_stats pa
LEFT JOIN pitch_stats ps ON pa.pitcher_id = ps.pitcher_id AND pa.game_id = ps.game_id
ORDER BY pa.pitcher_id, pa.game_date
"""

def get_pitcher_game_results(con) -> pd.DataFrame:
    """Raw per-game stats for every qualifying start."""
    return con.execute(PITCHER_GAME_RESULTS_SQL).df()


# ------------------------------------------------------------------
# Step 2: Rolling window features
# For each start, look back N starts and aggregate
# ------------------------------------------------------------------

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each pitcher-start, compute rolling stats over last 3, 5, 10 starts.
    Uses shift(1) so we never leak same-game data.
    """
    df = df.sort_values(['pitcher_id', 'game_date']).copy()

    for window in [3, 5, 10]:
        grp = df.groupby('pitcher_id')

        for col in ['k_rate', 'bb_rate', 'swstr_pct', 'zone_pct', 'pitch_count']:
            df[f'{col}_last{window}'] = (
                grp[col]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            )

        # Rolling K total (useful for over/under calibration)
        df[f'strikeouts_last{window}'] = (
            grp['strikeouts']
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )

    # Days rest
    df['days_rest'] = (
        df.groupby('pitcher_id')['game_date']
        .transform(lambda x: x.diff().dt.days)
        .fillna(5)   # default 5 days for first start
        .clip(1, 15)
    )

    return df


# ------------------------------------------------------------------
# Step 3: Stuff grades
# Z-score velocity, spin rate, and induced vertical break
# per pitch type against that season's league average
# ------------------------------------------------------------------

PITCH_STUFF_SQL = """
WITH pitcher_pitch_avgs AS (
    -- Average stuff metrics per pitcher per pitch type per season
    SELECT
        p.pitcher_id,
        g.season,
        p.pitch_type_code,
        p.pitch_type,
        COUNT(*)                            AS pitch_count,
        AVG(p.release_speed)                AS avg_velo,
        AVG(p.spin_rate)                    AS avg_spin,
        AVG(p.induced_vertical_break)       AS avg_ivb,
        AVG(p.horizontal_break)             AS avg_hbreak,
        -- Whiff rate per pitch type
        SUM(CASE WHEN p.pitch_call = 'S' THEN 1 ELSE 0 END) * 1.0
            / NULLIF(COUNT(*), 0)           AS whiff_rate
    FROM pitches p
    JOIN plate_appearances pa ON p.pa_id = pa.pa_id
    JOIN games g ON pa.game_id = g.game_id
    WHERE p.pitch_type_code IS NOT NULL
      AND p.release_speed IS NOT NULL
      AND g.season_type = 'regular'
    GROUP BY p.pitcher_id, g.season, p.pitch_type_code, p.pitch_type
    HAVING COUNT(*) >= 50   -- minimum pitches to compute reliable averages
),
league_avgs AS (
    -- League average per pitch type per season for z-scoring
    SELECT
        season,
        pitch_type_code,
        AVG(avg_velo)   AS lg_avg_velo,
        STDDEV(avg_velo) AS lg_std_velo,
        AVG(avg_spin)   AS lg_avg_spin,
        STDDEV(avg_spin) AS lg_std_spin,
        AVG(avg_ivb)    AS lg_avg_ivb,
        STDDEV(avg_ivb) AS lg_std_ivb,
        AVG(whiff_rate) AS lg_avg_whiff,
        STDDEV(whiff_rate) AS lg_std_whiff
    FROM pitcher_pitch_avgs
    GROUP BY season, pitch_type_code
)
SELECT
    ppa.*,
    -- Z-scores vs league average
    ROUND((ppa.avg_velo - la.lg_avg_velo) / NULLIF(la.lg_std_velo, 0), 3)   AS velo_z,
    ROUND((ppa.avg_spin - la.lg_avg_spin) / NULLIF(la.lg_std_spin, 0), 3)   AS spin_z,
    ROUND((ppa.avg_ivb  - la.lg_avg_ivb)  / NULLIF(la.lg_std_ivb,  0), 3)   AS ivb_z,
    ROUND((ppa.whiff_rate - la.lg_avg_whiff) / NULLIF(la.lg_std_whiff, 0), 3) AS whiff_z
FROM pitcher_pitch_avgs ppa
JOIN league_avgs la
    ON ppa.season = la.season
    AND ppa.pitch_type_code = la.pitch_type_code
"""


def get_pitch_stuff_grades(con) -> pd.DataFrame:
    """Per-pitcher, per-pitch-type stuff grades with league z-scores."""
    return con.execute(PITCH_STUFF_SQL).df()


def pivot_stuff_grades(stuff_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot stuff grades so each pitcher-season has one row with columns like:
    ff_velo_z, ff_spin_z, sl_whiff_z, ch_ivb_z, etc.

    Only pivots the 5 most common pitch types to keep feature space manageable.
    """
    key_pitches = ['FF', 'SL', 'CH', 'CU', 'SI']  # 4-seam, slider, changeup, curve, sinker
    key_metrics = ['avg_velo', 'velo_z', 'spin_z', 'ivb_z', 'whiff_rate', 'whiff_z']

    filtered = stuff_df[stuff_df['pitch_type_code'].isin(key_pitches)].copy()

    pivot = filtered.pivot_table(
        index=['pitcher_id', 'season'],
        columns='pitch_type_code',
        values=key_metrics,
        aggfunc='first'
    )

    # Flatten column names: (avg_velo, FF) -> ff_avg_velo
    pivot.columns = [f"{pt.lower()}_{metric}" for metric, pt in pivot.columns]
    pivot = pivot.reset_index()

    return pivot


# ------------------------------------------------------------------
# Step 4: Pitch mix features
# % of each pitch type thrown, last 3 starts
# ------------------------------------------------------------------

PITCH_MIX_SQL = """
WITH game_pitch_mix AS (
    SELECT
        pa.pitcher_id,
        pa.game_id,
        g.game_date::DATE AS game_date,
        p.pitch_type_code,
        COUNT(*) AS pitch_count,
        COUNT(*) * 1.0 / SUM(COUNT(*)) OVER (
            PARTITION BY pa.pitcher_id, pa.game_id
        ) AS pct_of_pitches
    FROM pitches p
    JOIN plate_appearances pa ON p.pa_id = pa.pa_id
    JOIN games g ON pa.game_id = g.game_id
    WHERE p.pitch_type_code IN ('FF', 'SL', 'CH', 'CU', 'SI', 'FC')
      AND g.season_type = 'regular'
    GROUP BY pa.pitcher_id, pa.game_id, g.game_date, p.pitch_type_code
)
SELECT *
FROM game_pitch_mix
ORDER BY pitcher_id, game_date
"""


def get_rolling_pitch_mix(con, window: int = 3) -> pd.DataFrame:
    """
    Returns rolling pitch mix (% FF, SL, CH, etc.) over last N starts.
    """
    mix_df = con.execute(PITCH_MIX_SQL).df()

    # Pivot to wide format per game
    wide = mix_df.pivot_table(
        index=['pitcher_id', 'game_id', 'game_date'],
        columns='pitch_type_code',
        values='pct_of_pitches',
        aggfunc='first'
    ).reset_index().fillna(0)

    wide.columns = [
        c if c in ['pitcher_id', 'game_id', 'game_date']
        else f"mix_{c.lower()}"
        for c in wide.columns
    ]

    # Rolling average of each pitch type % over last N starts
    wide = wide.sort_values(['pitcher_id', 'game_date'])
    mix_cols = [c for c in wide.columns if c.startswith('mix_')]

    for col in mix_cols:
        wide[f'{col}_last{window}'] = (
            wide.groupby('pitcher_id')[col]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )

    return wide[['pitcher_id', 'game_id'] + [f'{c}_last{window}' for c in mix_cols]]


# ------------------------------------------------------------------
# Step 5: Park factor for strikeouts
# ------------------------------------------------------------------

PARK_K_FACTOR_SQL = """
-- Strikeout rate by venue, normalized to league average
WITH venue_stats AS (
    SELECT
        g.venue,
        g.season,
        COUNT(*) AS total_pa,
        SUM(CASE WHEN pa.result = 'Strikeout' THEN 1 ELSE 0 END) AS total_k,
        SUM(CASE WHEN pa.result = 'Strikeout' THEN 1 ELSE 0 END) * 1.0
            / NULLIF(COUNT(*), 0) AS venue_k_rate
    FROM plate_appearances pa
    JOIN games g ON pa.game_id = g.game_id
    WHERE g.season_type = 'regular'
    GROUP BY g.venue, g.season
    HAVING COUNT(*) >= 500
),
league_k_rate AS (
    SELECT season, AVG(venue_k_rate) AS lg_k_rate
    FROM venue_stats
    GROUP BY season
)
SELECT
    v.venue,
    v.season,
    v.total_pa,
    v.venue_k_rate,
    l.lg_k_rate,
    ROUND(v.venue_k_rate / NULLIF(l.lg_k_rate, 0), 4) AS park_k_factor
FROM venue_stats v
JOIN league_k_rate l ON v.season = l.season
"""


def get_park_k_factors(con) -> pd.DataFrame:
    return con.execute(PARK_K_FACTOR_SQL).df()


# ------------------------------------------------------------------
# Master function: assemble full pitcher feature matrix
# ------------------------------------------------------------------

def build_pitcher_feature_matrix(db_path: str = None) -> pd.DataFrame:
    """
    Assembles the full pitcher feature matrix for model training.
    One row per qualifying start, with all features and the target variable.

    Returns a DataFrame ready for train/test split.
    """
    path = db_path or DB_PATH
    con = duckdb.connect(path, read_only=True)

    print("📊 Building pitcher game results...")
    results = get_pitcher_game_results(con)
    print(f"   {len(results):,} qualifying starts found")

    print("📈 Computing rolling window features...")
    results = add_rolling_features(results)

    print("🎯 Computing stuff grades...")
    stuff_raw = get_pitch_stuff_grades(con)
    stuff_pivot = pivot_stuff_grades(stuff_raw)
    results = results.merge(stuff_pivot, on=['pitcher_id', 'season'], how='left')

    print("🎲 Computing rolling pitch mix...")
    mix = get_rolling_pitch_mix(con)
    results = results.merge(mix, on=['pitcher_id', 'game_id'], how='left')

    print("🏟️  Joining park factors...")
    parks = get_park_k_factors(con)
    results = results.merge(parks[['venue', 'season', 'park_k_factor']],
                            on=['venue', 'season'], how='left')
    results['park_k_factor'] = results['park_k_factor'].fillna(1.0)

    con.close()

    # Encode pitcher hand
    results['pitcher_hand_enc'] = (results['pitcher_hand'] == 'R').astype(int)

    # Month (pitchers fatigue late season, batters adjust)
    results['month'] = pd.to_datetime(results['game_date']).dt.month

    print(f"\n✅ Feature matrix complete: {len(results):,} rows, {len(results.columns)} columns")
    print(f"   Date range: {results['game_date'].min()} → {results['game_date'].max()}")
    print(f"   Unique pitchers: {results['pitcher_id'].nunique():,}")

    return results


if __name__ == "__main__":
    df = build_pitcher_feature_matrix()
    print("\nSample columns:")
    print([c for c in df.columns if c not in ['pitcher_id', 'game_id', 'venue']])
    print("\nTarget distribution:")
    print(df['strikeouts'].describe())
