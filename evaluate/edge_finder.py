"""
evaluate/edge_finder.py
------------------------
Compares model predictions against book lines from the player_props table.
Flags games where our model disagrees with the market by more than a threshold.

Usage:
    python edge_finder.py                     # today's games
    python edge_finder.py --date 2026-04-12   # specific date
    python edge_finder.py --backtest          # backtest on all historical props
"""

import os
import sys
import argparse
import pickle
import duckdb
import numpy as np
import pandas as pd
from datetime import date, timedelta
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.pitcher_features import build_pitcher_feature_matrix
from features.lineup_features import build_lineup_feature_matrix

DB_PATH    = os.getenv("DB_PATH", "../data/mlb_modeling.db")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "saved")

# Minimum edge to flag a bet (in strikeouts)
EDGE_THRESHOLD = 0.75


def load_latest_model():
    files = sorted([f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')])
    if not files:
        raise FileNotFoundError("No saved models found. Run models/k_prop_model.py first.")
    path = os.path.join(MODELS_DIR, files[-1])
    with open(path, 'rb') as f:
        payload = pickle.load(f)
    print(f"✅ Loaded model: {files[-1]}")
    return payload['model'], payload['features']


def get_todays_props(con, target_date: str) -> pd.DataFrame:
    """Fetch K prop lines from the player_props table for a given date."""
    sql = f"""
        SELECT
            pp.game_id,
            pp.player_id AS pitcher_id,
            pp.market_type,
            pp.line,
            pp.over_odds,
            pp.under_odds,
            pp.book,
            pp.recorded_at,
            g.home_team_name,
            g.away_team_name,
            g.game_date
        FROM player_props pp
        JOIN games g ON pp.game_id = g.game_id
        WHERE pp.market_type = 'pitcher_strikeouts'
          AND g.game_date::DATE = '{target_date}'
        ORDER BY pp.game_id, pp.book
    """
    return con.execute(sql).df()


def build_prediction_features(pitcher_df: pd.DataFrame,
                               lineup_df: pd.DataFrame,
                               game_ids: list,
                               features: list) -> pd.DataFrame:
    """
    Filter feature matrices to only the games we need predictions for,
    and return X matrix aligned to model feature columns.
    """
    pitcher_filtered = pitcher_df[pitcher_df['game_id'].isin(game_ids)].copy()
    lineup_filtered  = lineup_df[lineup_df['game_id'].isin(game_ids)].copy()

    merged = pitcher_filtered.merge(lineup_filtered, on=['pitcher_id', 'game_id'], how='left')

    # Fill missing values same as training
    for col in features:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)
        else:
            merged[col] = 0

    return merged


def american_odds_to_implied_prob(odds: int) -> float:
    """Convert American odds to implied probability."""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


def calculate_kelly(edge: float, odds: int, kelly_fraction: float = 0.25) -> float:
    """
    Fractional Kelly criterion for bet sizing.
    edge: our probability of winning (0-1)
    odds: American odds
    kelly_fraction: 0.25 = quarter Kelly (conservative)
    """
    if odds > 0:
        b = odds / 100
    else:
        b = 100 / abs(odds)

    q = 1 - edge
    kelly = (b * edge - q) / b
    return max(0, kelly * kelly_fraction)


def find_edges(model, features: list, target_date: str,
               db_path: str = None) -> pd.DataFrame:
    """
    Main edge-finding function. Returns DataFrame of flagged bets
    with predicted K total, book line, edge, and Kelly sizing.
    """
    path = db_path or DB_PATH
    con  = duckdb.connect(path, read_only=True)

    # Get props for the date
    props = get_todays_props(con, target_date)
    if props.empty:
        print(f"No K props found for {target_date}")
        con.close()
        return pd.DataFrame()

    print(f"📋 Found {len(props)} prop lines for {target_date}")

    # Build features for these games
    game_ids = props['game_id'].unique().tolist()
    pitcher_df = build_pitcher_feature_matrix(path)
    lineup_df  = build_lineup_feature_matrix(path)

    pred_df = build_prediction_features(pitcher_df, lineup_df, game_ids, features)

    if pred_df.empty:
        print("No feature data available for these games yet.")
        con.close()
        return pd.DataFrame()

    # Generate predictions
    X = pred_df[features].values
    predictions = np.clip(model.predict(X), 0, 20)

    pred_df = pred_df[['pitcher_id', 'game_id']].copy()
    pred_df['predicted_ks'] = predictions

    # Join predictions onto props
    results = props.merge(pred_df, on=['pitcher_id', 'game_id'], how='inner')

    if results.empty:
        print("Could not match predictions to props (pitcher IDs may not align).")
        con.close()
        return pd.DataFrame()

    # Calculate edges
    results['edge']       = results['predicted_ks'] - results['line']
    results['abs_edge']   = results['edge'].abs()
    results['direction']  = results['edge'].apply(lambda e: 'OVER' if e > 0 else 'UNDER')

    # Implied probability from odds
    results['implied_prob_over']  = results['over_odds'].apply(
        lambda o: american_odds_to_implied_prob(o) if pd.notna(o) else 0.5
    )
    results['implied_prob_under'] = results['under_odds'].apply(
        lambda o: american_odds_to_implied_prob(o) if pd.notna(o) else 0.5
    )

    # Our probability estimate (rough conversion from K count edge to probability)
    # A 1-K edge at typical variance (~2.5 K std dev) → ~35% extra probability mass
    results['our_prob_over']  = results['edge'].apply(
        lambda e: min(0.95, max(0.05, 0.5 + e / 5.0))
    )
    results['our_prob_under'] = 1 - results['our_prob_over']

    # Kelly sizing
    results['kelly_over'] = results.apply(
        lambda r: calculate_kelly(r['our_prob_over'], r['over_odds'])
        if pd.notna(r['over_odds']) else 0, axis=1
    )
    results['kelly_under'] = results.apply(
        lambda r: calculate_kelly(r['our_prob_under'], r['under_odds'])
        if pd.notna(r['under_odds']) else 0, axis=1
    )

    # Flag edges above threshold
    flagged = results[results['abs_edge'] >= EDGE_THRESHOLD].copy()
    flagged = flagged.sort_values('abs_edge', ascending=False)

    con.close()
    return flagged


def print_edge_report(edges: pd.DataFrame):
    if edges.empty:
        print("No edges found above threshold.")
        return

    print("\n" + "=" * 70)
    print("⚾ K PROP EDGE REPORT")
    print("=" * 70)

    for _, row in edges.iterrows():
        direction = row['direction']
        odds_key  = 'over_odds' if direction == 'OVER' else 'under_odds'
        kelly_key = 'kelly_over' if direction == 'OVER' else 'kelly_under'
        odds      = row.get(odds_key, 'N/A')
        kelly     = row.get(kelly_key, 0)

        print(f"\n  {row['away_team_name']} @ {row['home_team_name']}")
        print(f"  Pitcher ID: {row['pitcher_id']} | Book: {row['book']}")
        print(f"  Line:      {row['line']} Ks")
        print(f"  Model:     {row['predicted_ks']:.1f} Ks")
        print(f"  Edge:      {row['edge']:+.2f} → BET {direction} ({odds})")
        print(f"  Kelly:     {kelly:.1%} of bankroll")
        print(f"  " + "-" * 40)


def backtest(model, features: list, db_path: str = None) -> pd.DataFrame:
    """
    Backtest the model against all historical props in the DB.
    Returns a record of each bet with outcome.
    """
    path = db_path or DB_PATH
    con  = duckdb.connect(path, read_only=True)

    props_sql = """
        SELECT
            pp.game_id, pp.player_id AS pitcher_id,
            pp.line, pp.over_odds, pp.under_odds, pp.book,
            g.game_date::DATE AS game_date, g.season
        FROM player_props pp
        JOIN games g ON pp.game_id = g.game_id
        WHERE pp.market_type = 'pitcher_strikeouts'
          AND g.season_type = 'regular'
        ORDER BY g.game_date
    """
    props = con.execute(props_sql).df()

    # Actual K totals
    actuals_sql = """
        SELECT
            pa.pitcher_id, pa.game_id,
            SUM(CASE WHEN pa.result = 'Strikeout' THEN 1 ELSE 0 END) AS actual_ks
        FROM plate_appearances pa
        GROUP BY pa.pitcher_id, pa.game_id
    """
    actuals = con.execute(actuals_sql).df()

    props = props.merge(actuals, on=['pitcher_id', 'game_id'], how='inner')

    # Build features and predict
    pitcher_df = build_pitcher_feature_matrix(path)
    lineup_df  = build_lineup_feature_matrix(path)
    pred_df    = build_prediction_features(
        pitcher_df, lineup_df,
        props['game_id'].unique().tolist(), features
    )
    pred_df['predicted_ks'] = np.clip(model.predict(pred_df[features].values), 0, 20)

    results = props.merge(pred_df[['pitcher_id', 'game_id', 'predicted_ks']],
                          on=['pitcher_id', 'game_id'], how='inner')

    results['edge']      = results['predicted_ks'] - results['line']
    results['direction'] = results['edge'].apply(lambda e: 'OVER' if e > 0 else 'UNDER')
    results['actual_over'] = results['actual_ks'] > results['line']
    results['bet_correct'] = (
        ((results['direction'] == 'OVER') & results['actual_over']) |
        ((results['direction'] == 'UNDER') & ~results['actual_over'])
    )

    # Only look at edges above threshold
    flagged = results[results['edge'].abs() >= EDGE_THRESHOLD].copy()

    print(f"\n📊 Backtest Results (edge >= {EDGE_THRESHOLD} Ks)")
    print(f"   Total flagged bets: {len(flagged):,}")
    print(f"   Correct: {flagged['bet_correct'].sum():,} ({flagged['bet_correct'].mean():.1%})")
    print(f"   By edge bucket:")
    for lo, hi in [(0.75, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 99)]:
        bucket = flagged[(flagged['edge'].abs() >= lo) & (flagged['edge'].abs() < hi)]
        if len(bucket) > 0:
            print(f"     {lo}-{hi}+ K edge: {len(bucket)} bets, "
                  f"{bucket['bet_correct'].mean():.1%} correct")

    con.close()
    return flagged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date',     type=str, help='Date YYYY-MM-DD (default: today)')
    parser.add_argument('--backtest', action='store_true')
    parser.add_argument('--db',       type=str, default=None)
    args = parser.parse_args()

    model, features = load_latest_model()

    if args.backtest:
        backtest(model, features, args.db)
        return

    target_date = args.date or str(date.today())
    edges = find_edges(model, features, target_date, args.db)
    print_edge_report(edges)


if __name__ == "__main__":
    main()
