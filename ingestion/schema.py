"""
schema.py
---------
Creates and manages the DuckDB schema for the MLB modeling database.
Run this once to initialize the database, or call init_db() from other scripts.
"""

import duckdb
import os
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv("DB_PATH", "data/mlb_modeling.db")


def get_connection() -> duckdb.DuckDBPyConnection:
    """Return a persistent DuckDB connection, creating the DB file if needed."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return duckdb.connect(DB_PATH)


def init_db():
    con = get_connection()

    # ------------------------------------------------------------------
    # GAMES
    # One row per game. Scores and inning-by-inning breakdown.
    # ------------------------------------------------------------------
    con.execute("""
        CREATE TABLE IF NOT EXISTS games (
            game_id             INTEGER PRIMARY KEY,
            season              INTEGER,
            season_type         VARCHAR,   -- 'regular', 'postseason', 'spring_training'
            game_date           TIMESTAMP,
            status              VARCHAR,
            home_team_id        INTEGER,
            away_team_id        INTEGER,
            home_team_name      VARCHAR,
            away_team_name      VARCHAR,
            venue               VARCHAR,
            attendance          INTEGER,
            home_runs           INTEGER,
            home_hits           INTEGER,
            home_errors         INTEGER,
            away_runs           INTEGER,
            away_hits           INTEGER,
            away_errors         INTEGER,
            home_inning_scores  VARCHAR,   -- stored as JSON string e.g. '[0,1,2,0]'
            away_inning_scores  VARCHAR,
            inserted_at         TIMESTAMP DEFAULT current_timestamp
        )
    """)

    # ------------------------------------------------------------------
    # PLAYERS
    # Lightweight player dimension table. Populated on first encounter.
    # ------------------------------------------------------------------
    con.execute("""
        CREATE TABLE IF NOT EXISTS players (
            player_id       INTEGER PRIMARY KEY,
            full_name       VARCHAR,
            position        VARCHAR,
            bats_throws     VARCHAR,   -- e.g. 'Left/Right'
            team_id         INTEGER,
            team_name       VARCHAR,
            active          BOOLEAN,
            inserted_at     TIMESTAMP DEFAULT current_timestamp
        )
    """)

    # ------------------------------------------------------------------
    # PLATE_APPEARANCES
    # One row per plate appearance. Core unit for batter/pitcher modeling.
    # ------------------------------------------------------------------
    con.execute("""
        CREATE TABLE IF NOT EXISTS plate_appearances (
            pa_id               VARCHAR PRIMARY KEY,  -- '{game_id}_{pa_number}'
            game_id             INTEGER,
            batter_id           INTEGER,
            pitcher_id          INTEGER,
            inning              INTEGER,
            half_inning         VARCHAR,   -- 'top' or 'bottom'
            pa_number           INTEGER,
            outs_at_start       INTEGER,
            batter_side         VARCHAR,   -- 'L' or 'R'
            pitcher_hand        VARCHAR,   -- 'L' or 'R'
            result              VARCHAR,   -- 'Single', 'Strikeout', 'Pop Out', etc.
            is_ball_in_play_out BOOLEAN,
            runner_on_first     BOOLEAN,
            runner_on_second    BOOLEAN,
            runner_on_third     BOOLEAN,
            pitch_count         INTEGER,   -- total pitches in this PA
            inserted_at         TIMESTAMP DEFAULT current_timestamp,

            FOREIGN KEY (game_id) REFERENCES games(game_id)
        )
    """)

    # ------------------------------------------------------------------
    # PITCHES
    # One row per pitch. Child of plate_appearances.
    # This is the Statcast-grade table — the heart of stuff+ modeling.
    # ------------------------------------------------------------------
    con.execute("""
        CREATE TABLE IF NOT EXISTS pitches (
            pitch_id                VARCHAR PRIMARY KEY,  -- '{game_id}_{pa_number}_{pitch_number}'
            pa_id                   VARCHAR,              -- FK to plate_appearances
            game_id                 INTEGER,
            batter_id               INTEGER,
            pitcher_id              INTEGER,
            pitch_number            INTEGER,
            balls                   INTEGER,
            strikes                 INTEGER,
            pitch_call              VARCHAR,   -- 'X' (in play), 'S' (strike), 'B' (ball), etc.
            pitch_type_code         VARCHAR,   -- 'FF', 'SL', 'CH', 'CU', etc.
            pitch_type              VARCHAR,   -- '4-Seam Fastball', 'Slider', etc.

            -- Velocity & Movement
            release_speed           DOUBLE,    -- mph at release
            plate_speed             DOUBLE,    -- mph at plate
            spin_rate               DOUBLE,    -- rpm
            release_extension       DOUBLE,    -- feet
            plate_time              DOUBLE,    -- seconds from release to plate
            horizontal_movement     DOUBLE,    -- inches
            vertical_movement       DOUBLE,    -- inches
            horizontal_break        DOUBLE,
            vertical_break          DOUBLE,
            induced_vertical_break  DOUBLE,    -- movement above gravity baseline

            -- Location
            plate_x                 DOUBLE,    -- horizontal position at plate (feet, 0 = center)
            plate_z                 DOUBLE,    -- vertical position at plate (feet)
            strike_zone_top         DOUBLE,
            strike_zone_bottom      DOUBLE,
            strike_zone             INTEGER,   -- zone 1-13

            -- Release Position
            release_pos_x           DOUBLE,
            release_pos_y           DOUBLE,
            release_pos_z           DOUBLE,

            -- Physics
            velocity_x              DOUBLE,
            velocity_y              DOUBLE,
            velocity_z              DOUBLE,
            acceleration_x          DOUBLE,
            acceleration_y          DOUBLE,
            acceleration_z          DOUBLE,

            -- Batted Ball (populated if ball in play)
            bat_speed               DOUBLE,
            exit_velocity           DOUBLE,
            launch_angle            DOUBLE,
            hit_distance            DOUBLE,
            expected_batting_average DOUBLE,
            is_barrel               BOOLEAN,
            hit_coordinate_x        DOUBLE,
            hit_coordinate_y        DOUBLE,

            -- Pitch Count Context (for fatigue modeling)
            game_pitch_count        INTEGER,   -- pitcher's total pitches in game so far
            pitcher_pitch_count     INTEGER,   -- same as above (API field name)

            inserted_at             TIMESTAMP DEFAULT current_timestamp,

            FOREIGN KEY (pa_id) REFERENCES plate_appearances(pa_id)
        )
    """)

    # ------------------------------------------------------------------
    # PLAYER_SPLITS
    # Pre-aggregated splits from the /players/splits endpoint.
    # Covers home/away, L/R, arena-level, and situational splits.
    # ------------------------------------------------------------------
    con.execute("""
        CREATE TABLE IF NOT EXISTS player_splits (
            split_id            VARCHAR PRIMARY KEY,  -- '{player_id}_{season}_{category}_{split_name}'
            player_id           INTEGER,
            season              INTEGER,
            category            VARCHAR,   -- 'batting' or 'pitching'
            split_category      VARCHAR,   -- 'byArena', 'byOpponent', etc.
            split_name          VARCHAR,
            split_abbreviation  VARCHAR,

            -- Batting splits (NULL for pitching rows)
            at_bats             INTEGER,
            runs                INTEGER,
            hits                INTEGER,
            doubles             INTEGER,
            triples             INTEGER,
            home_runs           INTEGER,
            rbis                INTEGER,
            walks               INTEGER,
            strikeouts          INTEGER,
            stolen_bases        INTEGER,
            caught_stealing     INTEGER,
            avg                 DOUBLE,
            obp                 DOUBLE,
            slg                 DOUBLE,
            ops                 DOUBLE,

            -- Pitching splits (NULL for batting rows)
            era                 DOUBLE,
            wins                INTEGER,
            losses              INTEGER,
            saves               INTEGER,
            games_played        INTEGER,
            games_started       INTEGER,
            innings_pitched     DOUBLE,
            hits_allowed        INTEGER,
            runs_allowed        INTEGER,
            earned_runs         INTEGER,
            home_runs_allowed   INTEGER,
            walks_allowed       INTEGER,
            strikeouts_pitched  INTEGER,
            opponent_avg        DOUBLE,

            inserted_at         TIMESTAMP DEFAULT current_timestamp
        )
    """)

    # ------------------------------------------------------------------
    # BETTING_ODDS  (populated from /betting_odds endpoint)
    # ------------------------------------------------------------------
    con.execute("""
        CREATE TABLE IF NOT EXISTS betting_odds (
            odds_id         VARCHAR PRIMARY KEY,
            game_id         INTEGER,
            book            VARCHAR,
            market          VARCHAR,   -- 'moneyline', 'spread', 'total'
            side            VARCHAR,   -- 'home', 'away', 'over', 'under'
            line            DOUBLE,
            odds            INTEGER,   -- American odds e.g. -110
            recorded_at     TIMESTAMP,
            inserted_at     TIMESTAMP DEFAULT current_timestamp
        )
    """)

    # ------------------------------------------------------------------
    # PLAYER_PROPS  (populated from /player_props endpoint)
    # ------------------------------------------------------------------
    con.execute("""
        CREATE TABLE IF NOT EXISTS player_props (
            prop_id         VARCHAR PRIMARY KEY,
            game_id         INTEGER,
            player_id       INTEGER,
            book            VARCHAR,
            market_type     VARCHAR,   -- 'pitcher_strikeouts', 'batter_hits', etc.
            line            DOUBLE,
            over_odds       INTEGER,
            under_odds      INTEGER,
            recorded_at     TIMESTAMP,
            inserted_at     TIMESTAMP DEFAULT current_timestamp
        )
    """)

    # ------------------------------------------------------------------
    # MODEL_PREDICTIONS  (your outputs — write model scores here)
    # ------------------------------------------------------------------
    con.execute("""
        CREATE TABLE IF NOT EXISTS model_predictions (
            prediction_id   VARCHAR PRIMARY KEY,
            game_id         INTEGER,
            player_id       INTEGER,
            model_name      VARCHAR,   -- e.g. 'k_prop_v1'
            market_type     VARCHAR,
            predicted_value DOUBLE,
            line            DOUBLE,    -- book line at time of prediction
            edge            DOUBLE,    -- predicted_value - line
            confidence      DOUBLE,    -- model probability / score
            result          DOUBLE,    -- actual outcome (filled in post-game)
            correct         BOOLEAN,   -- whether over/under call was right
            predicted_at    TIMESTAMP,
            inserted_at     TIMESTAMP DEFAULT current_timestamp
        )
    """)

    con.close()
    print(f"✅ Database initialized at {DB_PATH}")


if __name__ == "__main__":
    init_db()
