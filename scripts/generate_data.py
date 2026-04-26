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
import requests
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
API_KEY    = os.getenv("BDL_API_KEY")
BASE_URL   = "https://api.balldontlie.io/mlb/v1"
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "saved")
OUT_DIR    = os.path.join(os.path.dirname(__file__), "..", "docs", "data")


def fetch_games_from_api(target_date: str) -> dict:
    """Fetch game info directly from BDL API as fallback when not in DB."""
    try:
        resp = requests.get(
            f"{BASE_URL}/games",
            headers={"Authorization": API_KEY},
            params={"dates[]": target_date, "season_type": "regular", "per_page": 100}
        )
        games = resp.json().get("data", [])
        return {g['id']: g for g in games}
    except Exception:
        return {}


def load_latest_model():
    files = sorted([f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')
                    and 'negbinom' not in f])
    if not files:
        return None, None
    with open(os.path.join(MODELS_DIR, files[-1]), 'rb') as f:
        payload = pickle.load(f)
    return payload['model'], payload['features']


def generate_predictions(con, target_date: str) -> list:
    """Pull today's predictions joined with pitcher names, team, and game info."""
    sql = f"""
        SELECT
            pp.game_id,
            pp.player_id AS pitcher_id,
            pp.line, pp.over_odds, pp.under_odds, pp.book,
            pp.recorded_at,
            pl.full_name  AS pitcher_name,
            pl.team_name  AS pitcher_team
        FROM player_props pp
        LEFT JOIN players pl ON pp.player_id = pl.player_id
        WHERE pp.market_type = 'pitcher_strikeouts'
          AND DATE_TRUNC('day', pp.recorded_at) = '{target_date}'::DATE
    """
    props = con.execute(sql).df()
    if props.empty:
        return []

    # Pitcher hands from PA history
    id_list = ','.join(str(p) for p in props['pitcher_id'].unique().tolist())
    hands = con.execute(f"""
        SELECT pa.pitcher_id, MAX(pa.pitcher_hand) as pitcher_hand
        FROM plate_appearances pa WHERE pa.pitcher_id IN ({id_list})
        GROUP BY pa.pitcher_id
    """).df().set_index('pitcher_id')['pitcher_hand'].to_dict()

    # Game info — try DB first, fall back to API
    gid_list = ','.join(str(g) for g in props['game_id'].unique().tolist())
    try:
        games_map = con.execute(f"""
            SELECT game_id, home_team_name, away_team_name, venue
            FROM games WHERE game_id IN ({gid_list})
        """).df().set_index('game_id').to_dict('index')
    except Exception:
        games_map = {}

    # Fill missing games from API
    missing_ids = [g for g in props['game_id'].unique() if g not in games_map]
    if missing_ids:
        api_games = fetch_games_from_api(target_date)
        for gid, g in api_games.items():
            if gid not in games_map:
                games_map[gid] = {
                    'home_team_name': g.get('home_team_name', '—'),
                    'away_team_name': g.get('away_team_name', '—'),
                    'venue':          g.get('venue', '—'),
                }

    model, features = load_latest_model()
    results_df  = get_pitcher_game_results(con)
    results_df  = add_rolling_features(results_df)
    stuff_raw   = get_pitch_stuff_grades(con)
    stuff_pivot = pivot_stuff_grades(stuff_raw)
    parks       = get_park_k_factors(con)
    current_season = date.today().year

    results = []
    for _, row in props.iterrows():
        pitcher_id   = int(row['pitcher_id'])
        pitcher_name = row.get('pitcher_name') or f'ID {pitcher_id}'
        pitcher_team = row.get('pitcher_team') or '—'
        game_id      = int(row['game_id'])

        # Game context
        g         = games_map.get(game_id, {})
        home_team = g.get('home_team_name', '—')
        away_team = g.get('away_team_name', '—')
        venue     = g.get('venue', '—')
        matchup   = f"{away_team} @ {home_team}" if home_team != '—' else f"Game {game_id}"

        # Line freshness
        recorded_at = str(row.get('recorded_at', ''))
        if recorded_at and 'T' in recorded_at:
            try:
                ts = pd.Timestamp(recorded_at)
                line_updated = ts.strftime('%-I:%M %p').lstrip('0') if hasattr(ts, 'strftime') else recorded_at
            except Exception:
                line_updated = recorded_at[:16]
        else:
            line_updated = '—'

        # Feature vector
        hist = results_df[results_df['pitcher_id'] == pitcher_id]
        predicted_ks = edge = None

        if not hist.empty and model is not None:
            latest = hist.sort_values('game_date').iloc[-1].copy()

            ps = stuff_pivot[(stuff_pivot['pitcher_id'] == pitcher_id) &
                             (stuff_pivot['season'] == current_season)]
            if ps.empty:
                ps = stuff_pivot[stuff_pivot['pitcher_id'] == pitcher_id].sort_values('season').tail(1)
            if not ps.empty:
                for col in ps.columns:
                    if col not in ['pitcher_id', 'season']:
                        latest[col] = ps.iloc[0][col]

            park_row = parks[(parks['venue'] == venue) & (parks['season'] == current_season)]
            latest['park_k_factor']          = park_row['park_k_factor'].values[0] if not park_row.empty else 1.0
            latest['pitcher_hand_enc']        = 1 if latest.get('pitcher_hand') == 'R' else 0
            latest['month']                   = pd.Timestamp(target_date).month
            latest['lineup_k_rate_weighted']  = 0.228
            latest['lineup_k_rate_raw']       = 0.228

            feat_vec = {f: (0 if pd.isna(latest.get(f, 0)) else latest.get(f, 0)) for f in features}
            X = np.array([feat_vec[f] for f in features]).reshape(1, -1)
            predicted_ks = float(np.clip(model.predict(X)[0], 0, 20))
            edge = round(predicted_ks - float(row['line']), 2) if pd.notna(row['line']) else None

        results.append({
            'pitcher_id':   pitcher_id,
            'pitcher_name': pitcher_name,
            'pitcher_team': pitcher_team,
            'pitcher_hand': hands.get(pitcher_id, '?'),
            'matchup':      matchup,
            'venue':        venue,
            'line':         float(row['line']) if pd.notna(row['line']) else None,
            'predicted_ks': round(predicted_ks, 1) if predicted_ks is not None else None,
            'edge':         edge,
            'direction':    ('OVER' if edge > 0 else 'UNDER') if edge else None,
            'over_odds':    int(row['over_odds']) if pd.notna(row['over_odds']) else None,
            'under_odds':   int(row['under_odds']) if pd.notna(row['under_odds']) else None,
            'book':         row['book'],
            'line_updated': line_updated,
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
            pl.full_name  AS pitcher_name,
            pl.team_name  AS pitcher_team,
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
        LEFT JOIN players pl ON pa.pitcher_id = pl.player_id
        WHERE g.season = {current_season}
          AND g.season_type = 'regular'
          AND p.pitch_type_code = 'FF'
          AND p.release_speed IS NOT NULL
          AND p.spin_rate IS NOT NULL
        GROUP BY pa.pitcher_id, pl.full_name, pl.team_name
        HAVING COUNT(*) >= 50
        ORDER BY whiff_rate DESC
        LIMIT 50
    """
    df = con.execute(sql).df()
    return df.round(3).fillna(0).to_dict('records')


def generate_accuracy(con) -> dict:
    """
    Historical edge accuracy from model_predictions table.
    Returns three views:
      - summary: overall stats
      - daily: rollup by date
      - bets: individual bet-level detail with pitcher name, line, predicted, actual
      - by_book: accuracy breakdown per sportsbook
    """
    empty_response = {
        'summary': {'total_bets': 0, 'accuracy': None,
                    'message': 'No tracked predictions yet — run log_results.py after games finish'},
        'daily': [], 'bets': [], 'by_book': []
    }

    try:
        # Check if table has any data
        count = con.execute("SELECT COUNT(*) FROM model_predictions WHERE correct IS NOT NULL").fetchone()[0]
        if count == 0:
            return empty_response
    except Exception:
        return empty_response

    try:
        # -- Daily rollup --
        daily_sql = """
            SELECT
                CAST(mp.predicted_at::DATE AS VARCHAR) AS pred_date,
                COUNT(*)                        AS total_bets,
                SUM(CASE WHEN mp.correct THEN 1 ELSE 0 END) AS correct_bets,
                ROUND(AVG(CASE WHEN mp.correct THEN 1.0 ELSE 0.0 END), 4) AS accuracy,
                ROUND(AVG(mp.edge), 3)          AS avg_edge,
                ROUND(AVG(mp.predicted_value), 2) AS avg_predicted,
                ROUND(AVG(mp.line), 2)          AS avg_line,
                ROUND(AVG(mp.result), 2)        AS avg_actual
            FROM model_predictions mp
            WHERE mp.correct IS NOT NULL
            GROUP BY pred_date
            ORDER BY pred_date DESC
            LIMIT 90
        """
        daily_df = con.execute(daily_sql).df()

        # -- Bet-level detail --
        bets_sql = """
            SELECT
                CAST(mp.predicted_at::DATE AS VARCHAR) AS game_date,
                mp.player_id                    AS pitcher_id,
                COALESCE(pl.full_name, 'ID ' || CAST(mp.player_id AS VARCHAR)) AS pitcher_name,
                COALESCE(pl.team_name, '—')     AS pitcher_team,
                COALESCE(pp.book, '—')          AS book,
                ROUND(mp.line, 1)               AS line,
                ROUND(mp.predicted_value, 1)    AS predicted,
                ROUND(mp.result, 1)             AS actual,
                ROUND(mp.edge, 2)               AS edge,
                CAST(mp.correct AS INTEGER)     AS correct,
                CAST(pp.over_odds AS INTEGER)   AS over_odds,
                CAST(pp.under_odds AS INTEGER)  AS under_odds,
                CASE WHEN mp.predicted_value > mp.line THEN 'OVER' ELSE 'UNDER' END AS direction
            FROM model_predictions mp
            LEFT JOIN players pl ON mp.player_id = pl.player_id
            LEFT JOIN player_props pp
                ON mp.player_id = pp.player_id
                AND mp.game_id  = pp.game_id
                AND pp.market_type = 'pitcher_strikeouts'
            WHERE mp.correct IS NOT NULL
            ORDER BY mp.predicted_at DESC, ABS(mp.edge) DESC
            LIMIT 500
        """
        bets_df = con.execute(bets_sql).df()
        # Convert correct back to bool-like for JS (0/1 instead of NA)
        bets_df['correct'] = bets_df['correct'].fillna(-1).astype(int)
        bets_df['over_odds']  = bets_df['over_odds'].where(bets_df['over_odds'].notna(), None)
        bets_df['under_odds'] = bets_df['under_odds'].where(bets_df['under_odds'].notna(), None)

        # -- By book breakdown --
        by_book_sql = """
            SELECT
                pp.book,
                COUNT(*)                        AS total_bets,
                SUM(CASE WHEN mp.correct THEN 1 ELSE 0 END) AS correct_bets,
                ROUND(AVG(CASE WHEN mp.correct THEN 1.0 ELSE 0.0 END), 4) AS accuracy,
                ROUND(AVG(mp.edge), 3)          AS avg_edge,
                ROUND(AVG(mp.predicted_value - mp.result), 3) AS avg_prediction_error
            FROM model_predictions mp
            JOIN player_props pp
                ON mp.player_id = pp.player_id
                AND mp.game_id  = pp.game_id
                AND pp.market_type = 'pitcher_strikeouts'
            WHERE mp.correct IS NOT NULL
              AND pp.book IS NOT NULL
            GROUP BY pp.book
            HAVING COUNT(*) >= 5
            ORDER BY accuracy DESC
        """
        by_book_df = con.execute(by_book_sql).df()

        # -- Overall summary --
        total   = int(daily_df['total_bets'].sum())
        correct = int(daily_df['correct_bets'].sum())

        return {
            'summary': {
                'total_bets':    total,
                'correct_bets':  correct,
                'accuracy':      round(correct / total, 4) if total > 0 else None,
                'avg_edge':      round(float(daily_df['avg_edge'].mean()), 3),
                'avg_actual':    round(float(daily_df['avg_actual'].mean()), 2),
                'avg_predicted': round(float(daily_df['avg_predicted'].mean()), 2),
                'avg_line':      round(float(daily_df['avg_line'].mean()), 2),
            },
            'daily':   daily_df.fillna(0).to_dict('records'),
            'bets':    bets_df.where(bets_df.notna(), None).to_dict('records'),
            'by_book': by_book_df.fillna(0).to_dict('records'),
        }

    except Exception as e:
        print(f"   ⚠️  Accuracy generation error: {e}")
        return empty_response



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
