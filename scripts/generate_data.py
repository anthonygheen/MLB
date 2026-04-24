"""
scripts/generate_data.py
-------------------------
Queries DuckDB and exports JSON data files for the GitHub Pages dashboard.
Run after predict_today.py to update the dashboard data.

Output files written to docs/data/:
  predictions.json   — today's predictions + edges
  stuff_grades.json  — pitcher stuff grade leaderboard (current season)
  accuracy.json      — historical edge accuracy tracker
  park_factors.json  — park K factors by venue

Usage:
    python scripts/generate_data.py
    python scripts/generate_data.py --date 2026-04-25
"""

import os
import sys
import json
import pickle
import argparse
import duckdb
import numpy as np
import pandas as pd
from datetime import date, timedelta
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.pitcher_features import (
    get_pitcher_game_results, add_rolling_features,
    get_pitch_stuff_grades, pivot_stuff_grades,
    get_rolling_pitch_mix, get_park_k_factors
)

DB_PATH    = os.getenv("DB_PATH", "data/mlb_modeling.db")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "saved")
OUT_DIR    = os.path.join(os.path.dirname(__file__), "..", "docs", "data")


def load_latest_model():
    files = sorted([f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')
                    and 'negbinom' not in f])
    if not files:
        return None, None
    with open(os.path.join(MODELS_DIR, files[-1]), 'rb') as f:
        payload = pickle.load(f)
    return payload['model'], payload['features']


def generate_predictions(con, target_date: str) -> list:
    """Pull today's predictions from player_props joined with pitcher data."""
    sql = f"""
        SELECT
            pp.game_id,
            pp.player_id AS pitcher_id,
            pp.line,
            pp.over_odds,
            pp.under_odds,
            pp.book,
            pp.market_type
        FROM player_props pp
        WHERE pp.market_type = 'pitcher_strikeouts'
          AND DATE_TRUNC('day', pp.recorded_at) = '{target_date}'::DATE
    """
    props = con.execute(sql).df()
    if props.empty:
        return []

    # Get player names
    pitcher_ids = props['pitcher_id'].unique().tolist()
    id_list = ','.join(str(p) for p in pitcher_ids)

    name_sql = f"""
        SELECT DISTINCT pa.pitcher_id,
               MAX(pa.pitcher_hand) as pitcher_hand
        FROM plate_appearances pa
        WHERE pa.pitcher_id IN ({id_list})
        GROUP BY pa.pitcher_id
    """
    hands = con.execute(name_sql).df().set_index('pitcher_id')['pitcher_hand'].to_dict()

    model, features = load_latest_model()

    results = []
    results_df = get_pitcher_game_results(con)
    results_df  = add_rolling_features(results_df)
    stuff_raw   = get_pitch_stuff_grades(con)
    stuff_pivot = pivot_stuff_grades(stuff_raw)
    parks       = get_park_k_factors(con)
    current_season = date.today().year

    for _, row in props.iterrows():
        pitcher_id = int(row['pitcher_id'])
        hist = results_df[results_df['pitcher_id'] == pitcher_id]

        if hist.empty or model is None:
            predicted_ks = None
            edge = None
        else:
            latest = hist.sort_values('game_date').iloc[-1].copy()

            pitcher_stuff = stuff_pivot[
                (stuff_pivot['pitcher_id'] == pitcher_id) &
                (stuff_pivot['season'] == current_season)
            ]
            if pitcher_stuff.empty:
                pitcher_stuff = stuff_pivot[stuff_pivot['pitcher_id'] == pitcher_id]\
                                    .sort_values('season').tail(1)

            if not pitcher_stuff.empty:
                for col in pitcher_stuff.columns:
                    if col not in ['pitcher_id', 'season']:
                        latest[col] = pitcher_stuff.iloc[0][col]

            latest['park_k_factor']    = 1.0
            latest['pitcher_hand_enc'] = 1 if latest.get('pitcher_hand') == 'R' else 0
            latest['month']            = pd.Timestamp(target_date).month
            latest['lineup_k_rate_weighted'] = 0.228
            latest['lineup_k_rate_raw']      = 0.228

            feat_vec = {f: (0 if pd.isna(latest.get(f, 0)) else latest.get(f, 0))
                        for f in features}
            X = np.array([feat_vec[f] for f in features]).reshape(1, -1)
            predicted_ks = float(np.clip(model.predict(X)[0], 0, 20))
            edge = round(predicted_ks - float(row['line']), 2) if row['line'] else None

        results.append({
            'pitcher_id':   pitcher_id,
            'pitcher_hand': hands.get(pitcher_id, '?'),
            'line':         float(row['line']) if pd.notna(row['line']) else None,
            'predicted_ks': round(predicted_ks, 1) if predicted_ks else None,
            'edge':         edge,
            'direction':    ('OVER' if edge and edge > 0 else 'UNDER') if edge else None,
            'over_odds':    int(row['over_odds']) if pd.notna(row['over_odds']) else None,
            'under_odds':   int(row['under_odds']) if pd.notna(row['under_odds']) else None,
            'book':         row['book'],
            'flagged':      abs(edge) >= 0.75 if edge else False,
        })

    return sorted(results, key=lambda x: abs(x['edge'] or 0), reverse=True)


def generate_stuff_grades(con) -> list:
    """Top 50 pitchers by fastball whiff rate this season."""
    current_season = date.today().year
    sql = f"""
        WITH league_avg AS (
            SELECT AVG(release_speed) as lg_avg_velo
            FROM pitches p
            JOIN plate_appearances pa ON p.pa_id = pa.pa_id
            JOIN games g ON pa.game_id = g.game_id
            WHERE g.season = {current_season}
              AND g.season_type = 'regular'
              AND p.pitch_type_code = 'FF'
              AND p.release_speed IS NOT NULL
        )
        SELECT
            pa.pitcher_id,
            COUNT(*) as pitch_count,
            AVG(p.release_speed) as avg_velo,
            AVG(p.spin_rate) as avg_spin,
            AVG(p.induced_vertical_break) as avg_ivb,
            SUM(CASE WHEN p.pitch_call = 'S' THEN 1 ELSE 0 END) * 1.0
                / COUNT(*) as whiff_rate,
            AVG(p.release_speed) - (SELECT lg_avg_velo FROM league_avg) as velo_vs_avg
        FROM pitches p
        JOIN plate_appearances pa ON p.pa_id = pa.pa_id
        JOIN games g ON pa.game_id = g.game_id
        WHERE g.season = {current_season}
          AND g.season_type = 'regular'
          AND p.pitch_type_code = 'FF'
          AND p.release_speed IS NOT NULL
          AND p.spin_rate IS NOT NULL
        GROUP BY pa.pitcher_id
        HAVING COUNT(*) >= 50
        ORDER BY whiff_rate DESC
        LIMIT 50
    """
    df = con.execute(sql).df()
    return df.round(3).fillna(0).to_dict('records')


def generate_accuracy(con) -> dict:
    """Historical edge accuracy from model_predictions table."""
    sql = """
        SELECT
            predicted_at::DATE as pred_date,
            model_name,
            market_type,
            COUNT(*) as total_bets,
            SUM(CASE WHEN correct THEN 1 ELSE 0 END) as correct_bets,
            AVG(edge) as avg_edge,
            AVG(CASE WHEN correct THEN 1.0 ELSE 0.0 END) as accuracy
        FROM model_predictions
        WHERE correct IS NOT NULL
        GROUP BY pred_date, model_name, market_type
        ORDER BY pred_date DESC
        LIMIT 90
    """
    try:
        df = con.execute(sql).df()
        if df.empty:
            return {'records': [], 'summary': {
                'total_bets': 0, 'accuracy': None,
                'message': 'No tracked predictions yet — accuracy builds over time'
            }}

        total   = int(df['total_bets'].sum())
        correct = int(df['correct_bets'].sum())
        accuracy = round(correct / total, 4) if total > 0 else None

        return {
            'records': df.round(4).fillna(0).to_dict('records'),
            'summary': {
                'total_bets': total,
                'correct_bets': correct,
                'accuracy': accuracy,
                'avg_edge': round(float(df['avg_edge'].mean()), 3),
            }
        }
    except Exception:
        return {'records': [], 'summary': {
            'total_bets': 0, 'accuracy': None,
            'message': 'No tracked predictions yet'
        }}


def generate_park_factors(con) -> list:
    """Park K factors for current and prior season."""
    current_season = date.today().year
    sql = f"""
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
              AND g.season >= {current_season - 1}
            GROUP BY g.venue, g.season
            HAVING COUNT(*) >= 200
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
            ROUND(v.venue_k_rate, 4) AS venue_k_rate,
            ROUND(v.venue_k_rate / NULLIF(l.lg_k_rate, 0), 4) AS park_k_factor
        FROM venue_stats v
        JOIN league_k_rate l ON v.season = l.season
        ORDER BY park_k_factor DESC
    """
    df = con.execute(sql).df()
    return df.fillna(0).to_dict('records')


def write_json(data, filename: str):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, filename)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"   ✅ {filename} ({len(data) if isinstance(data, list) else 'object'})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, default=None)
    parser.add_argument('--db',   type=str, default=None)
    args = parser.parse_args()

    target_date = args.date or str(date.today())
    db_path     = args.db or DB_PATH

    print(f"\n📊 Generating dashboard data for {target_date}...")
    con = duckdb.connect(db_path, read_only=True)

    print("\n  Predictions...")
    predictions = generate_predictions(con, target_date)
    write_json({
        'date': target_date,
        'generated_at': str(pd.Timestamp.now()),
        'predictions': predictions,
        'edges_flagged': sum(1 for p in predictions if p.get('flagged')),
    }, 'predictions.json')

    print("  Stuff grades...")
    stuff = generate_stuff_grades(con)
    write_json({'season': date.today().year, 'pitchers': stuff}, 'stuff_grades.json')

    print("  Accuracy tracker...")
    accuracy = generate_accuracy(con)
    write_json(accuracy, 'accuracy.json')

    print("  Park factors...")
    parks = generate_park_factors(con)
    write_json({'parks': parks}, 'park_factors.json')

    con.close()
    print(f"\n✅ Dashboard data written to {OUT_DIR}")
    print(f"   {len(predictions)} predictions, {len(stuff)} pitchers, {len(parks)} parks")


if __name__ == "__main__":
    main()
