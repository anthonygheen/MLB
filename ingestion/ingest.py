"""
ingest.py
---------
Main ingestion script. Two modes:

  Daily (default):
    python ingest.py
    Pulls yesterday's completed games + their plate appearances.

  Backfill:
    python ingest.py --season 2025
    python ingest.py --start-date 2025-04-01 --end-date 2025-04-30

Usage:
    python ingest.py                                  # yesterday
    python ingest.py --date 2026-04-12                # specific date
    python ingest.py --season 2025                    # full season backfill
    python ingest.py --start-date 2025-04-01 --end-date 2025-04-30
"""

import os
import json
import argparse
import duckdb
from datetime import date, timedelta, datetime
from tqdm import tqdm
from dotenv import load_dotenv

from bdl_client import BDLClient
from schema import init_db, get_connection

load_dotenv()

RAW_DATA_PATH = os.getenv("RAW_DATA_PATH", "data/raw")


# ----------------------------------------------------------------------
# Upsert helpers
# ----------------------------------------------------------------------

def upsert_game(con: duckdb.DuckDBPyConnection, game: dict):
    htd = game.get("home_team_data", {})
    atd = game.get("away_team_data", {})
    con.execute("""
        INSERT OR REPLACE INTO games (
            game_id, season, season_type, game_date, status,
            home_team_id, away_team_id, home_team_name, away_team_name,
            venue, attendance,
            home_runs, home_hits, home_errors,
            away_runs, away_hits, away_errors,
            home_inning_scores, away_inning_scores
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        game["id"], game.get("season"), game.get("season_type"),
        game.get("date"), game.get("status"),
        game.get("home_team", {}).get("id"), game.get("away_team", {}).get("id"),
        game.get("home_team_name"), game.get("away_team_name"),
        game.get("venue"), game.get("attendance"),
        htd.get("runs"), htd.get("hits"), htd.get("errors"),
        atd.get("runs"), atd.get("hits"), atd.get("errors"),
        json.dumps(htd.get("inning_scores", [])),
        json.dumps(atd.get("inning_scores", [])),
    ])


def upsert_plate_appearance(con: duckdb.DuckDBPyConnection, pa: dict, game_id: int):
    pa_id = f"{game_id}_{pa['pa_number']}"
    con.execute("""
        INSERT OR REPLACE INTO plate_appearances (
            pa_id, game_id, batter_id, pitcher_id,
            inning, half_inning, pa_number, outs_at_start,
            batter_side, pitcher_hand, result, is_ball_in_play_out,
            runner_on_first, runner_on_second, runner_on_third, pitch_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        pa_id, game_id, pa.get("batter_id"), pa.get("pitcher_id"),
        pa.get("inning"), pa.get("half_inning"), pa.get("pa_number"), pa.get("outs"),
        pa.get("batter_side"), pa.get("pitcher_hand"), pa.get("result"),
        pa.get("is_ball_in_play_out"),
        pa.get("runner_on_first"), pa.get("runner_on_second"), pa.get("runner_on_third"),
        len(pa.get("pitches", [])),
    ])

    for pitch in pa.get("pitches", []):
        upsert_pitch(con, pitch, pa_id, game_id,
                     pa.get("batter_id"), pa.get("pitcher_id"), pa.get("pa_number"))


def upsert_pitch(con, pitch, pa_id, game_id, batter_id, pitcher_id, pa_number):
    pitch_id = f"{game_id}_{pa_number}_{pitch['pitch_number']}"
    con.execute("""
        INSERT OR REPLACE INTO pitches (
            pitch_id, pa_id, game_id, batter_id, pitcher_id,
            pitch_number, balls, strikes, pitch_call, pitch_type_code, pitch_type,
            release_speed, plate_speed, spin_rate, release_extension, plate_time,
            horizontal_movement, vertical_movement,
            horizontal_break, vertical_break, induced_vertical_break,
            plate_x, plate_z, strike_zone_top, strike_zone_bottom, strike_zone,
            release_pos_x, release_pos_y, release_pos_z,
            velocity_x, velocity_y, velocity_z,
            acceleration_x, acceleration_y, acceleration_z,
            bat_speed, exit_velocity, launch_angle, hit_distance,
            expected_batting_average, is_barrel,
            hit_coordinate_x, hit_coordinate_y,
            game_pitch_count, pitcher_pitch_count
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, [
        pitch_id, pa_id, game_id, batter_id, pitcher_id,
        pitch.get("pitch_number"), pitch.get("balls"), pitch.get("strikes"),
        pitch.get("pitch_call"), pitch.get("pitch_type_code"), pitch.get("pitch_type"),
        pitch.get("release_speed"), pitch.get("plate_speed"),
        pitch.get("spin_rate"), pitch.get("release_extension"), pitch.get("plate_time"),
        pitch.get("horizontal_movement"), pitch.get("vertical_movement"),
        pitch.get("horizontal_break"), pitch.get("vertical_break"),
        pitch.get("induced_vertical_break"),
        pitch.get("plate_x"), pitch.get("plate_z"),
        pitch.get("strike_zone_top"), pitch.get("strike_zone_bottom"), pitch.get("strike_zone"),
        pitch.get("release_pos_x"), pitch.get("release_pos_y"), pitch.get("release_pos_z"),
        pitch.get("velocity_x"), pitch.get("velocity_y"), pitch.get("velocity_z"),
        pitch.get("acceleration_x"), pitch.get("acceleration_y"), pitch.get("acceleration_z"),
        pitch.get("bat_speed"), pitch.get("exit_velocity"), pitch.get("launch_angle"),
        pitch.get("hit_distance"), pitch.get("expected_batting_average"), pitch.get("is_barrel"),
        pitch.get("hit_coordinate_x"), pitch.get("hit_coordinate_y"),
        pitch.get("game_pitch_count"), pitch.get("pitcher_pitch_count"),
    ])


# ----------------------------------------------------------------------
# Core ingestion functions
# ----------------------------------------------------------------------

def ingest_date(client: BDLClient, con: duckdb.DuckDBPyConnection,
                target_date: str, skip_existing: bool = True):
    print(f"\n📅 Ingesting {target_date}...")
    games = list(client.get_games(dates=[target_date]))
    completed = [g for g in games if g.get("status") == "STATUS_FINAL"]

    if not completed:
        print(f"   No completed games for {target_date}")
        return

    print(f"   Found {len(completed)} completed games")

    for game in tqdm(completed, desc="   Games"):
        game_id = game["id"]

        if skip_existing:
            count = con.execute(
                "SELECT COUNT(*) FROM plate_appearances WHERE game_id = ?", [game_id]
            ).fetchone()[0]
            if count > 0:
                continue

        upsert_game(con, game)
        for pa in client.get_plate_appearances(game_id):
            upsert_plate_appearance(con, pa, game_id)
        con.commit()

    print(f"   ✅ Done")


def ingest_season(client: BDLClient, con: duckdb.DuckDBPyConnection, season: int):
    print(f"\n🗂️  Backfilling season {season}...")
    games = list(client.get_games(seasons=[season], season_type="regular"))
    completed = [g for g in games if g.get("status") == "STATUS_FINAL"]
    print(f"   Found {len(completed)} completed games")

    for game in tqdm(completed, desc=f"   {season}"):
        game_id = game["id"]
        count = con.execute(
            "SELECT COUNT(*) FROM plate_appearances WHERE game_id = ?", [game_id]
        ).fetchone()[0]
        if count > 0:
            continue
        upsert_game(con, game)
        for pa in client.get_plate_appearances(game_id):
            upsert_plate_appearance(con, pa, game_id)
        con.commit()

    print(f"   ✅ Season {season} complete")


def ingest_date_range(client, con, start, end):
    start_dt = datetime.strptime(start, "%Y-%m-%d").date()
    end_dt   = datetime.strptime(end,   "%Y-%m-%d").date()
    current  = start_dt
    while current <= end_dt:
        ingest_date(client, con, current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="BallDontLie MLB ingestion")
    parser.add_argument("--date",       type=str)
    parser.add_argument("--season",     type=int)
    parser.add_argument("--start-date", type=str)
    parser.add_argument("--end-date",   type=str)
    args = parser.parse_args()

    client = BDLClient()
    init_db()
    con = get_connection()

    if args.season:
        ingest_season(client, con, args.season)
    elif args.start_date and args.end_date:
        ingest_date_range(client, con, args.start_date, args.end_date)
    elif args.date:
        ingest_date(client, con, args.date)
    else:
        yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
        ingest_date(client, con, yesterday)

    con.close()


if __name__ == "__main__":
    main()
