"""
scripts/log_results.py
-----------------------
Runs post-game to compare prop lines against actual K totals
and logs outcomes into the model_predictions table.

Can be run daily after games finish (~midnight ET) to build up
the accuracy tracker. Also supports backfilling all historical props.

Usage:
    python scripts/log_results.py                    # yesterday's games
    python scripts/log_results.py --date 2026-04-24  # specific date
    python scripts/log_results.py --backfill          # all historical props
"""

import os
import sys
import argparse
import pickle
import duckdb
import numpy as np
import pandas as pd
from datetime import date, timedelta
from scipy.stats import nbinom
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.pitcher_features import (
    get_pitcher_game_results, add_rolling_features,
    get_pitch_stuff_grades, pivot_stuff_grades,
    get_park_k_factors
)

DB_PATH    = os.getenv("DB_PATH", "data/mlb_modeling.db")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "saved")


def load_latest_model():
    files = sorted([f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')
                    and 'negbinom' not in f])
    if not files:
        return None, None, None
    with open(os.path.join(MODELS_DIR, files[-1]), 'rb') as f:
        payload = pickle.load(f)
    r = payload.get('metadata', {}).get('negbinom_r')
    return payload['model'], payload['features'], r


def compute_p_over(mu: float, r: float, line: float) -> float:
    p = r / (r + mu)
    return float(1 - nbinom.cdf(int(line), r, p))


def get_actual_ks(con) -> pd.DataFrame:
    """
    Compute actual K totals for every completed starter appearance.
    A starter is the pitcher with the most PA in the first 3 innings.
    """
    sql = """
        WITH starter_ks AS (
            SELECT
                pa.game_id,
                pa.pitcher_id,
                COUNT(*) as batters_faced,
                SUM(CASE WHEN pa.result = 'Strikeout' THEN 1 ELSE 0 END) as actual_ks
            FROM plate_appearances pa
            JOIN games g ON pa.game_id = g.game_id
            WHERE g.status = 'STATUS_FINAL'
              AND g.season_type = 'regular'
            GROUP BY pa.game_id, pa.pitcher_id
            HAVING COUNT(*) >= 12
        )
        SELECT game_id, pitcher_id, actual_ks
        FROM starter_ks
    """
    return con.execute(sql).df()


def get_props_to_score(con, target_date: str = None,
                       backfill: bool = False) -> pd.DataFrame:
    """
    Pull prop lines that haven't been scored yet.
    Joins against games to filter to completed games only.
    """
    date_filter = ""
    if not backfill and target_date:
        date_filter = f"AND g.game_date::DATE = '{target_date}'::DATE"
    elif not backfill:
        yesterday = str(date.today() - timedelta(days=1))
        date_filter = f"AND g.game_date::DATE = '{yesterday}'::DATE"

    sql = f"""
        SELECT DISTINCT
            pp.prop_id,
            pp.game_id,
            pp.player_id AS pitcher_id,
            pp.book,
            pp.line,
            pp.over_odds,
            pp.under_odds,
            pp.recorded_at,
            g.game_date::DATE AS game_date,
            g.venue
        FROM player_props pp
        JOIN games g ON pp.game_id = g.game_id
        WHERE pp.market_type = 'pitcher_strikeouts'
          AND g.status = 'STATUS_FINAL'
          AND pp.line IS NOT NULL
          {date_filter}
          -- Skip already logged predictions
          AND NOT EXISTS (
              SELECT 1 FROM model_predictions mp
              WHERE mp.prediction_id = pp.prop_id
          )
        ORDER BY g.game_date, pp.game_id
    """
    return con.execute(sql).df()


def build_prediction(pitcher_id: int, venue: str,
                     results_df: pd.DataFrame,
                     stuff_pivot: pd.DataFrame,
                     parks: pd.DataFrame,
                     features: list,
                     model,
                     game_date: str,
                     negbinom_r: float = None,
                     line: float = None) -> tuple[float | None, float | None]:
    """Build feature vector and generate model prediction for a pitcher."""
    hist = results_df[results_df['pitcher_id'] == pitcher_id]
    if hist.empty or model is None:
        return None, None

    # Use data up to the game date to avoid leakage
    hist = hist[hist['game_date'] < pd.Timestamp(game_date)].copy()
    if hist.empty:
        return None, None

    latest = hist.sort_values('game_date').iloc[-1].copy()
    current_season = pd.Timestamp(game_date).year

    ps = stuff_pivot[(stuff_pivot['pitcher_id'] == pitcher_id) &
                     (stuff_pivot['season'] == current_season)]
    if ps.empty:
        ps = stuff_pivot[stuff_pivot['pitcher_id'] == pitcher_id].sort_values('season').tail(1)
    if not ps.empty:
        for col in ps.columns:
            if col not in ['pitcher_id', 'season']:
                latest[col] = ps.iloc[0][col]

    park_row = parks[(parks['venue'] == venue) &
                     (parks['season'] == current_season)]
    latest['park_k_factor']         = park_row['park_k_factor'].values[0] if not park_row.empty else 1.0
    latest['pitcher_hand_enc']       = 1 if latest.get('pitcher_hand') == 'R' else 0
    latest['month']                  = pd.Timestamp(game_date).month
    latest['lineup_k_rate_weighted'] = 0.228
    latest['lineup_k_rate_raw']      = 0.228

    feat_vec = {f: (0 if pd.isna(latest.get(f, 0)) else latest.get(f, 0))
                for f in features}
    X = np.array([feat_vec[f] for f in features]).reshape(1, -1)
    predicted = float(np.clip(model.predict(X)[0], 0, 20))

    p_over = (compute_p_over(predicted, negbinom_r, line)
              if negbinom_r is not None and line is not None else None)

    return predicted, p_over


def log_results(target_date: str = None, backfill: bool = False):
    model, features, negbinom_r = load_latest_model()
    con = duckdb.connect(DB_PATH)

    mode = "backfill" if backfill else (target_date or str(date.today() - timedelta(days=1)))
    print(f"\n📊 Logging prediction results — {mode}")

    # Pull what we need
    props   = get_props_to_score(con, target_date, backfill)
    actuals = get_actual_ks(con)

    if props.empty:
        print("   No unscored props found for completed games")
        con.close()
        return

    print(f"   {len(props)} prop lines to score")
    print(f"   {len(actuals)} actual K totals available")

    # Build feature data
    results_df  = get_pitcher_game_results(con)
    results_df  = add_rolling_features(results_df)
    stuff_raw   = get_pitch_stuff_grades(con)
    stuff_pivot = pivot_stuff_grades(stuff_raw)
    parks       = get_park_k_factors(con)

    # Merge actuals onto props
    merged = props.merge(
        actuals[['game_id', 'pitcher_id', 'actual_ks']],
        on=['game_id', 'pitcher_id'],
        how='inner'
    )

    if merged.empty:
        print("   No matches between props and actual K data")
        print("   (Game PAs may not be ingested yet — run ingest.py first)")
        con.close()
        return

    print(f"   {len(merged)} matched prop-result pairs found")

    logged = 0
    skipped = 0

    for _, row in merged.iterrows():
        pitcher_id   = int(row['pitcher_id'])
        game_date    = str(row['game_date'])
        venue        = row.get('venue', '—') or '—'
        line         = float(row['line'])
        actual_ks    = float(row['actual_ks'])

        # Get model prediction and probability
        predicted, p_over = build_prediction(
            pitcher_id, venue, results_df, stuff_pivot,
            parks, features, model, game_date,
            negbinom_r=negbinom_r, line=line
        )

        if predicted is None:
            skipped += 1
            continue

        edge        = round(predicted - line, 3)
        direction   = 'OVER' if predicted > line else 'UNDER'
        actual_over = actual_ks > line
        correct     = (direction == 'OVER' and actual_over) or \
                      (direction == 'UNDER' and not actual_over)

        # confidence = P(the direction we bet is correct)
        if p_over is not None:
            confidence = p_over if direction == 'OVER' else (1 - p_over)
        else:
            confidence = None

        try:
            con.execute("""
                INSERT OR REPLACE INTO model_predictions (
                    prediction_id, game_id, player_id,
                    model_name, market_type,
                    predicted_value, line, edge, confidence,
                    result, correct,
                    predicted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                row['prop_id'],
                int(row['game_id']),
                pitcher_id,
                'k_prop_gbm',
                'pitcher_strikeouts',
                round(predicted, 2),
                line,
                edge,
                round(confidence, 4) if confidence is not None else None,
                actual_ks,
                correct,
                game_date,
            ])
            logged += 1
        except Exception as e:
            print(f"   ⚠️  Insert error for pitcher {pitcher_id}: {e}")
            skipped += 1

    con.commit()
    con.close()

    print(f"\n   ✅ {logged} predictions logged")
    if skipped:
        print(f"   ⚠️  {skipped} skipped (no history or insert error)")

    if logged > 0:
        # Quick accuracy summary
        _print_summary(merged, logged)


def _print_summary(merged: pd.DataFrame, logged: int):
    """Print a quick accuracy summary for the logged batch."""
    print(f"\n   Quick accuracy check on this batch:")
    # We can't easily recompute correct here without re-running predictions
    # Just show the distribution of edges
    if 'line' in merged.columns and 'actual_ks' in merged.columns:
        over_rate = (merged['actual_ks'] > merged['line']).mean()
        print(f"   Actual over rate vs line: {over_rate:.1%}")
        print(f"   Avg line: {merged['line'].mean():.2f}")
        print(f"   Avg actual Ks: {merged['actual_ks'].mean():.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date',     type=str,  default=None,
                        help='Date to score YYYY-MM-DD (default: yesterday)')
    parser.add_argument('--backfill', action='store_true',
                        help='Score all historical props in DB')
    args = parser.parse_args()

    log_results(target_date=args.date, backfill=args.backfill)


if __name__ == "__main__":
    main()
