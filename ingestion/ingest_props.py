"""
ingestion/ingest_props.py
--------------------------
Pulls pitcher strikeout prop lines from the BallDontLie API
and lands them in the player_props table.

BDL props schema:
  - prop_type: 'pitcher_strikeouts' (the field we filter on)
  - vendor: sportsbook name
  - line_value: string (e.g. '5.5')
  - market.type: 'over' or 'under' (separate records per side)
  - market.odds: American odds integer

Over and under come as separate records — we pivot them into
one row per (game_id, player_id, vendor) with over_odds and under_odds.

Run this daily after lineups confirm (~2-3 hrs pre-game).

Usage:
    python ingestion/ingest_props.py                  # today
    python ingestion/ingest_props.py --date 2026-04-25
    python ingestion/ingest_props.py --debug          # show all prop_types seen
"""

import os
import sys
import argparse
import requests
from datetime import date
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingestion.bdl_client import BDLClient
from ingestion.schema import get_connection, init_db

DB_PATH  = os.getenv("DB_PATH", "data/mlb_modeling.db")
API_KEY  = os.getenv("BDL_API_KEY")
BASE_URL = "https://api.balldontlie.io/mlb/v1"

# K prop type strings to match
K_PROP_TYPES = {'pitcher_strikeouts', 'strikeouts', 'pitcher_ks'}


def get_game_ids_for_date(target_date: str) -> list:
    """Always pull game IDs from API — don't rely on DB."""
    resp = requests.get(
        f"{BASE_URL}/games",
        headers={"Authorization": API_KEY},
        params={"dates[]": target_date, "season_type": "regular", "per_page": 100}
    )
    resp.raise_for_status()
    games = resp.json().get("data", [])
    # Only scheduled or in-progress — skip completed games
    active = [g['id'] for g in games
              if g.get('status') not in ('STATUS_FINAL', 'STATUS_POSTPONED')]
    return active, games


def ingest_props_for_date(target_date: str, debug: bool = False):
    client = BDLClient()
    init_db()
    con = get_connection()

    print(f"\n📋 Pulling K props for {target_date}...")

    game_ids, all_games = get_game_ids_for_date(target_date)

    if not game_ids:
        print(f"   No active games found for {target_date} — all may be final or postponed")
        con.close()
        return

    print(f"   Found {len(game_ids)} active games")

    total_inserted = 0
    prop_types_seen = set()

    for game_id in game_ids:
        try:
            props = list(client.get_player_props(game_id=game_id))
        except Exception as e:
            print(f"   ⚠️  Error fetching props for game {game_id}: {e}")
            continue

        if not props:
            continue

        # Track prop types for debug output
        for p in props:
            prop_types_seen.add(p.get('prop_type', 'unknown'))

        # Filter to K props only
        k_props = [p for p in props
                   if (p.get('prop_type') or '').lower() in K_PROP_TYPES
                   or any(k in (p.get('prop_type') or '').lower()
                          for k in ['strikeout', 'pitcher_k'])]

        if debug and not k_props:
            print(f"   ℹ️  Game {game_id}: {len(props)} props, "
                  f"none are K props. Types seen: "
                  f"{set(p.get('prop_type') for p in props)}")

        # Pivot over/under into one row per (player, vendor, line)
        # Key: (player_id, vendor, line_value)
        pivoted = {}
        for prop in k_props:
            player_id  = prop.get('player_id')
            vendor     = prop.get('vendor', 'unknown')
            line_value = prop.get('line_value')
            market     = prop.get('market', {})
            mkt_type   = (market.get('type') or '').lower()
            mkt_odds   = market.get('odds')

            key = (player_id, vendor, line_value)
            if key not in pivoted:
                pivoted[key] = {
                    'game_id':    game_id,
                    'player_id':  player_id,
                    'vendor':     vendor,
                    'prop_type':  prop.get('prop_type'),
                    'line_value': line_value,
                    'over_odds':  None,
                    'under_odds': None,
                }

            if mkt_type == 'over':
                pivoted[key]['over_odds'] = mkt_odds
            elif mkt_type == 'under':
                pivoted[key]['under_odds'] = mkt_odds

        # Insert pivoted rows
        for key, row in pivoted.items():
            player_id, vendor, line_value = key
            prop_id = f"{game_id}_{player_id}_{vendor}"

            try:
                con.execute("""
                    INSERT OR REPLACE INTO player_props (
                        prop_id, game_id, player_id, book,
                        market_type, line, over_odds, under_odds,
                        recorded_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, current_timestamp)
                """, [
                    prop_id,
                    game_id,
                    player_id,
                    vendor,
                    row['prop_type'],
                    float(line_value) if line_value else None,
                    row['over_odds'],
                    row['under_odds'],
                ])
                total_inserted += 1
            except Exception as e:
                print(f"   ⚠️  Insert error: {e}")
                continue

    if debug:
        print(f"\n   📊 All prop types seen across today's games: {prop_types_seen}")

    con.commit()
    con.close()

    if total_inserted == 0:
        print(f"   ⚠️  No K props inserted")
        print(f"      Run with --debug to see all prop types available")
    else:
        print(f"   ✅ {total_inserted} K prop lines inserted for {target_date}")


def main():
    parser = argparse.ArgumentParser(description="Ingest pitcher K prop lines")
    parser.add_argument('--date',  type=str, default=None,
                        help='Date YYYY-MM-DD (default: today)')
    parser.add_argument('--debug', action='store_true',
                        help='Show all prop types seen in API response')
    args = parser.parse_args()

    target_date = args.date or str(date.today())
    ingest_props_for_date(target_date, debug=args.debug)


if __name__ == "__main__":
    main()
