"""
ingestion/sync_players.py
--------------------------
Pulls all active players from the BDL API and upserts into the players table.
Run this once to bootstrap, then periodically to catch trades/callups.

Usage:
    python ingestion/sync_players.py
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingestion.bdl_client import BDLClient
from ingestion.schema import get_connection, init_db


def sync_players():
    client = BDLClient()
    init_db()
    con = get_connection()

    print("👤 Syncing active players from BDL API...")

    players = list(client.paginate("players/active", {"per_page": 100}))
    print(f"   Found {len(players)} active players")

    inserted = 0
    for p in players:
        team = p.get("team") or {}
        try:
            con.execute("""
                INSERT OR REPLACE INTO players (
                    player_id, full_name, position,
                    bats_throws, team_id, team_name, active
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [
                p.get("id"),
                p.get("full_name"),
                p.get("position"),
                p.get("bats_throws"),
                team.get("id"),
                team.get("display_name"),
                p.get("active", True),
            ])
            inserted += 1
        except Exception as e:
            print(f"   ⚠️  Insert error for player {p.get('id')}: {e}")

    con.commit()
    con.close()
    print(f"   ✅ {inserted} players synced")


if __name__ == "__main__":
    sync_players()
