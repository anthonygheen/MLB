"""
scripts/predict_today.py
-------------------------
Pre-game K prop prediction sheet using confirmed lineups from the BDL API.

Workflow:
  1. Pull today's scheduled games from API
  2. Pull confirmed lineups (is_probable_pitcher identifies starters)
  3. Build pitcher feature vector from historical data
  4. Build opposing lineup K rate from confirmed batting order
  5. Generate predicted K total
  6. Join against props lines if available
  7. Print edge table

Run 2-3 hours before first pitch after lineups are posted.

Usage:
    python scripts/predict_today.py
    python scripts/predict_today.py --date 2026-04-25
    python scripts/predict_today.py --threshold 0.5
"""

import os
import sys
import argparse
import pickle
import requests
import duckdb
import numpy as np
import pandas as pd
from datetime import date
from scipy.stats import nbinom
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

LINEUP_WEIGHTS = {1: 1.15, 2: 1.10, 3: 1.08, 4: 1.05, 5: 1.02,
                  6: 0.98, 7: 0.95, 8: 0.92, 9: 0.88}


def api_get(endpoint: str, params: dict = None) -> dict:
    resp = requests.get(
        f"{BASE_URL}/{endpoint}",
        headers={"Authorization": API_KEY},
        params=params or {}
    )
    resp.raise_for_status()
    return resp.json()


def get_games_for_date(target_date: str) -> list:
    data = api_get("games", {
        "dates[]": target_date,
        "season_type": "regular",
        "per_page": 100
    })
    return data.get("data", [])


def get_all_lineups(game_ids: list) -> list:
    """
    Fetch lineup data once for the slate.
    The BDL lineups endpoint returns all probable pitchers and batting orders
    league-wide regardless of game_id, so one call covers the full day.
    We try each game_id until we get a non-empty response.
    """
    for gid in game_ids:
        try:
            entries = api_get("lineups", {"game_id": gid, "per_page": 100}).get("data", [])
            if entries:
                return entries
        except Exception:
            continue
    return []


def build_lineup_maps(lineup_entries: list) -> tuple[dict, dict]:
    """
    Build team_id → starter and team_id → batting_order maps from lineup data.
    Returns (starters_map, orders_map).
    """
    starters: dict = {}
    orders:   dict = {}
    for entry in lineup_entries:
        team_id   = entry.get("team", {}).get("id")
        player    = entry.get("player", {})
        is_prob   = entry.get("is_probable_pitcher", False)
        bat_order = entry.get("batting_order")
        if team_id is None:
            continue
        if is_prob:
            starters[team_id] = player
        if bat_order is not None:
            orders.setdefault(team_id, []).append(
                (bat_order, player.get("id"), player)
            )
    return starters, orders


def load_latest_model():
    files = sorted([f for f in os.listdir(MODELS_DIR)
                    if f.endswith('.pkl') and 'negbinom' not in f])
    if not files:
        raise FileNotFoundError("No saved models found. Run: python models/k_prop_model.py")
    path = os.path.join(MODELS_DIR, files[-1])
    with open(path, 'rb') as f:
        payload = pickle.load(f)
    r = payload.get('metadata', {}).get('negbinom_r')
    if r:
        print(f"✅ Model: {files[-1]}  (NegBinom r={r:.4f})")
    else:
        print(f"✅ Model: {files[-1]}  (no NegBinom r — retrain to enable probabilities)")
    return payload['model'], payload['features'], r


def compute_p_over(mu: float, r: float, line: float) -> float:
    """P(K > line) using NegBinom(mu, r). Works for half and whole number lines."""
    p = r / (r + mu)
    return float(1 - nbinom.cdf(int(line), r, p))


def implied_prob(american_odds: int) -> float:
    """Convert American odds to implied probability (no vig removed)."""
    if american_odds < 0:
        return (-american_odds) / (-american_odds + 100)
    return 100 / (american_odds + 100)


def compute_ev(p_win: float, american_odds: int) -> float:
    """Expected value per $100 risked."""
    payout = 100 / (-american_odds) * 100 if american_odds < 0 else float(american_odds)
    return round(p_win * payout - (1 - p_win) * 100, 2)


def build_pitcher_features(con, pitcher_id: int, venue: str, features: list):
    """Build current feature vector for a pitcher using their historical data."""
    results = get_pitcher_game_results(con)
    pitcher_hist = results[results['pitcher_id'] == pitcher_id].copy()

    if pitcher_hist.empty:
        return None, None

    pitcher_hist = add_rolling_features(pitcher_hist)

    # Stuff grades — current season first, fall back to most recent
    stuff_raw   = get_pitch_stuff_grades(con)
    stuff_pivot = pivot_stuff_grades(stuff_raw)
    current_season = date.today().year
    pitcher_stuff = stuff_pivot[
        (stuff_pivot['pitcher_id'] == pitcher_id) &
        (stuff_pivot['season'] == current_season)
    ]
    if pitcher_stuff.empty:
        pitcher_stuff = stuff_pivot[stuff_pivot['pitcher_id'] == pitcher_id] \
                            .sort_values('season').tail(1)

    # Pitch mix
    mix = get_rolling_pitch_mix(con)
    pitcher_mix = mix[mix['pitcher_id'] == pitcher_id]

    # Park factor
    parks = get_park_k_factors(con)
    park_row = parks[(parks['venue'] == venue) & (parks['season'] == current_season)]
    park_k_factor = park_row['park_k_factor'].values[0] if not park_row.empty else 1.0

    # Start with most recent start as base
    latest = pitcher_hist.sort_values('game_date').iloc[-1].copy()

    if not pitcher_stuff.empty:
        for col in pitcher_stuff.columns:
            if col not in ['pitcher_id', 'season']:
                latest[col] = pitcher_stuff.iloc[0][col]

    if not pitcher_mix.empty:
        for col in pitcher_mix.columns:
            if col not in ['pitcher_id', 'game_id']:
                latest[col] = pitcher_mix.iloc[-1][col]

    latest['park_k_factor']    = park_k_factor
    latest['pitcher_hand_enc'] = 1 if latest.get('pitcher_hand') == 'R' else 0
    latest['month']            = pd.Timestamp(date.today()).month

    feature_vec = {f: (0 if pd.isna(latest.get(f, 0)) else latest.get(f, 0))
                   for f in features}

    return pd.Series(feature_vec), latest.get('pitcher_hand', '?')


def compute_lineup_k_rate(con, batting_order: list, pitcher_hand: str) -> float:
    """Weighted opposing lineup K rate vs pitcher handedness, last 30 days."""
    if not batting_order:
        return 0.228

    batter_ids = [b[1] for b in batting_order if b[1]]
    if not batter_ids:
        return 0.228

    sql = f"""
        SELECT
            pa.batter_id,
            COUNT(*) as total_pa,
            SUM(CASE WHEN pa.result = 'Strikeout' THEN 1 ELSE 0 END) as k_count
        FROM plate_appearances pa
        JOIN games g ON pa.game_id = g.game_id
        WHERE pa.batter_id IN ({','.join(str(b) for b in batter_ids)})
          AND pa.pitcher_hand = '{pitcher_hand}'
          AND g.game_date >= (CURRENT_DATE - INTERVAL '30 days')
          AND g.season_type = 'regular'
        GROUP BY pa.batter_id
        HAVING COUNT(*) >= 5
    """
    batter_rates = con.execute(sql).df()

    weighted = []
    for pos, batter_id, _ in batting_order:
        weight    = LINEUP_WEIGHTS.get(pos, 0.95)
        brow      = batter_rates[batter_rates['batter_id'] == batter_id]
        k_rate    = (brow.iloc[0]['k_count'] / brow.iloc[0]['total_pa']
                     if not brow.empty else 0.228)
        weighted.append(k_rate * weight)

    return round(np.mean(weighted), 4) if weighted else 0.228


def get_props_for_date(con, target_date: str) -> dict:
    """
    Pull K prop lines recorded today — no games join needed since
    tonight's games may not be in the games table yet.
    Returns dict keyed by pitcher_id with best available line (lowest juice book).
    """
    sql = f"""
        SELECT player_id AS pitcher_id, game_id,
               line, over_odds, under_odds, book
        FROM player_props
        WHERE market_type = 'pitcher_strikeouts'
          AND DATE_TRUNC('day', recorded_at) = '{target_date}'::DATE
        ORDER BY game_id, book
    """
    df = con.execute(sql).df()
    if df.empty:
        return {}
    # One row per pitcher — use first book available
    return {int(row['pitcher_id']): row.to_dict()
            for _, row in df.groupby('pitcher_id').first().reset_index().iterrows()}


def print_predictions(predictions: list, min_ev: float):
    print("\n" + "=" * 75)
    print(f"⚾  PRE-GAME K PROP PREDICTIONS  —  {date.today().strftime('%B %d, %Y')}")
    print("=" * 75)

    if not predictions:
        print("\n  No predictions available — lineups may not be posted yet.")
        print("  Try again 2-3 hours before first pitch.\n")
        return

    has_lines  = any(p.get('line') is not None for p in predictions)
    has_probs  = any(p.get('p_over') is not None for p in predictions)
    bets_found = 0

    for p in predictions:
        matchup  = f"{p['away_team']} @ {p['home_team']}"
        pitcher  = p.get('pitcher_name', f"ID:{p['pitcher_id']}")
        hand_str = 'RHP' if p.get('pitcher_hand') == 'R' else 'LHP'
        pred     = p.get('predicted_ks')
        time_str = p.get('game_time', '')

        print(f"\n  {matchup:<35} {time_str}")
        print(f"  Starter: {pitcher} ({hand_str})")
        print(f"  Opp lineup K rate: {p.get('lineup_k_rate', 0.228):.3f} vs {p.get('pitcher_hand','?')}HP")

        if pred is None:
            print(f"  Projected Ks: N/A (insufficient history)")
            print(f"  {'─' * 55}")
            continue

        print(f"  Projected Ks: {pred:.2f}")

        if has_lines and p.get('line') is not None:
            line     = p['line']
            book     = p.get('book', '?')
            o_odds   = p.get('over_odds')
            u_odds   = p.get('under_odds')
            p_over   = p.get('p_over')
            p_under  = 1 - p_over if p_over is not None else None

            odds_str = ""
            if o_odds and u_odds:
                odds_str = f"  Over {o_odds:+d} / Under {u_odds:+d}"
            print(f"  Book line:    {line:.1f}  [{book}]{odds_str}")

            if p_over is not None:
                # Probability-based display
                imp_over  = implied_prob(o_odds) if o_odds else None
                imp_under = implied_prob(u_odds) if u_odds else None
                ev_over   = compute_ev(p_over,  o_odds) if o_odds else None
                ev_under  = compute_ev(p_under, u_odds) if u_odds else None

                print(f"  P(Over):      {p_over:.1%}  |  P(Under): {p_under:.1%}")

                if imp_over and imp_under:
                    edge_over  = p_over  - imp_over
                    edge_under = p_under - imp_under
                    print(f"  Implied:      Over {imp_over:.1%} / Under {imp_under:.1%}  "
                          f"(edge: Over {edge_over:+.1%} / Under {edge_under:+.1%})")
                    print(f"  EV ($100):    Over {ev_over:+.2f} / Under {ev_under:+.2f}")

                    # Flag best side if above min_ev threshold
                    best_ev   = max(ev_over, ev_under)
                    best_side = 'OVER ↑' if ev_over >= ev_under else 'UNDER ↓'
                    best_odds = o_odds if ev_over >= ev_under else u_odds
                    if best_ev >= min_ev:
                        print(f"  {'─' * 55}")
                        print(f"  🎯 BET {best_side}  {best_odds:+d}  (EV: +${best_ev:.2f} per $100)")
                        bets_found += 1
                    else:
                        print(f"  → No edge  (best EV: ${best_ev:+.2f}, need ${min_ev:.0f})")
                else:
                    # Have probabilities but no odds — show P only
                    direction = 'OVER ↑' if p_over > 0.524 else 'UNDER ↓'
                    print(f"  → {direction}  (no odds in DB — run ingest_props.py)")
            else:
                # No NegBinom r — fall back to raw edge display
                edge      = pred - line
                direction = 'OVER ↑' if edge > 0 else 'UNDER ↓'
                odds      = o_odds if edge > 0 else u_odds
                o_str     = f" ({odds:+d})" if odds else ""
                print(f"  Edge:         {edge:+.2f} Ks  →  {direction}{o_str}")
        else:
            print(f"  Book line:    not available")

        print(f"  {'─' * 55}")

    if has_lines and has_probs:
        print(f"\n  Bets flagged (EV ≥ ${min_ev:.0f} per $100): {bets_found}\n")
    elif has_lines:
        print(f"\n  ℹ️  Retrain model to enable probability/EV output\n")
    else:
        print(f"\n  ℹ️  Run ingestion/ingest_props.py to pull book lines\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date',   type=str,  default=None)
    parser.add_argument('--min-ev', type=float, default=3.0,
                        help='Minimum EV per $100 to flag a bet (default: 3.0)')
    parser.add_argument('--db',     type=str,  default=None)
    args = parser.parse_args()

    target_date = args.date or str(date.today())
    db_path     = args.db or DB_PATH

    print(f"\n🔍 K Prop Predictions — {target_date}")
    model, features, negbinom_r = load_latest_model()
    con = duckdb.connect(db_path, read_only=True)

    # Pull today's games from API
    print("   Fetching today's schedule...")
    games = get_games_for_date(target_date)
    scheduled = [g for g in games if g.get('status') in
                 ('STATUS_SCHEDULED', 'scheduled', 'STATUS_IN_PROGRESS')]

    if not scheduled:
        print(f"   No scheduled games found for {target_date}")
        con.close()
        return

    print(f"   {len(scheduled)} games on the slate")

    # Pull props if available
    props_map = get_props_for_date(con, target_date)
    prop_count = con.execute(
        f"SELECT COUNT(DISTINCT player_id) FROM player_props "
        f"WHERE DATE_TRUNC('day', recorded_at) = '{target_date}'::DATE"
    ).fetchone()[0]
    if prop_count > 0:
        print(f"   {prop_count} pitchers with prop lines in DB")
    else:
        print("   No prop lines in DB — run ingestion/ingest_props.py first")

    # Fetch all lineups once — BDL returns league-wide data for any game_id
    game_ids = [g['id'] for g in scheduled]
    print("   Fetching lineups...")
    lineup_entries         = get_all_lineups(game_ids)
    starters_map, orders_map = build_lineup_maps(lineup_entries)
    print(f"   {len(starters_map)} probable starters confirmed")

    predictions = []

    for game in scheduled:
        game_id   = game['id']
        home_team = game.get('home_team_name', '?')
        away_team = game.get('away_team_name', '?')
        venue     = game.get('venue', '')

        try:
            ts = pd.Timestamp(game.get('date', ''))
            # API returns UTC; convert to ET (UTC-4 during EDT)
            ts_et     = ts - pd.Timedelta(hours=4)
            game_time = ts_et.strftime('%I:%M %p ET').lstrip('0')
        except Exception:
            game_time = ''

        home_team_id = game.get('home_team', {}).get('id')
        away_team_id = game.get('away_team', {}).get('id')

        for side in ['home', 'away']:
            team_id  = home_team_id if side == 'home' else away_team_id
            opp_id   = away_team_id if side == 'home' else home_team_id
            starter  = starters_map.get(team_id)
            opp_order = sorted(orders_map.get(opp_id, []), key=lambda x: x[0])

            if not starter:
                continue

            pitcher_id   = starter.get('id')
            pitcher_name = starter.get('full_name', f'ID:{pitcher_id}')

            feat_series, pitcher_hand = build_pitcher_features(
                con, pitcher_id, venue, features
            )

            if feat_series is None:
                print(f"   ⚠️  No history for {pitcher_name} — skipping")
                continue

            lineup_k_rate = compute_lineup_k_rate(con, opp_order, pitcher_hand)

            if 'lineup_k_rate_weighted' in features:
                feat_series['lineup_k_rate_weighted'] = lineup_k_rate
            if 'lineup_k_rate_raw' in features:
                feat_series['lineup_k_rate_raw'] = lineup_k_rate

            X = feat_series[features].values.reshape(1, -1)
            predicted_ks = float(np.clip(model.predict(X)[0], 0, 20))

            prop   = props_map.get(pitcher_id, {})
            line   = prop.get('line')
            p_over = (compute_p_over(predicted_ks, negbinom_r, line)
                      if negbinom_r and line is not None else None)

            predictions.append({
                'game_id':       game_id,
                'pitcher_id':    pitcher_id,
                'pitcher_name':  pitcher_name,
                'pitcher_hand':  pitcher_hand,
                'home_team':     home_team,
                'away_team':     away_team,
                'venue':         venue,
                'game_time':     game_time,
                'predicted_ks':  predicted_ks,
                'lineup_k_rate': lineup_k_rate,
                'line':          line,
                'p_over':        p_over,
                'over_odds':     prop.get('over_odds'),
                'under_odds':    prop.get('under_odds'),
                'book':          prop.get('book'),
            })

    con.close()
    print_predictions(predictions, args.min_ev)


if __name__ == "__main__":
    main()
