"""
bdl_client.py
-------------
Thin wrapper around the BallDontLie MLB API.
Handles auth headers, rate limiting (600 req/min on GOAT), and cursor pagination.
"""

import os
import time
import requests
from typing import Generator, Optional
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.balldontlie.io/mlb/v1"
API_KEY  = os.getenv("BDL_API_KEY")

# GOAT tier: 600 req/min = 10 req/sec. We stay conservative at 8/sec.
_MIN_REQUEST_INTERVAL = 1 / 8


class BDLClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or API_KEY
        if not self.api_key:
            raise ValueError("BDL_API_KEY not set. Add it to your .env file.")
        self.session = requests.Session()
        self.session.headers.update({"Authorization": self.api_key})
        self._last_request_time = 0.0

    def _throttle(self):
        """Enforce minimum interval between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < _MIN_REQUEST_INTERVAL:
            time.sleep(_MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.time()

    def _get(self, endpoint: str, params: dict = None) -> dict:
        """Single GET request with basic error handling."""
        self._throttle()
        url = f"{BASE_URL}/{endpoint}"
        resp = self.session.get(url, params=params or {})

        if resp.status_code == 429:
            print("⚠️  Rate limited — sleeping 10s")
            time.sleep(10)
            return self._get(endpoint, params)

        resp.raise_for_status()
        return resp.json()

    def paginate(self, endpoint: str, params: dict = None) -> Generator[dict, None, None]:
        """
        Cursor-based pagination. Yields individual records from 'data' array.
        Handles all pages automatically.
        """
        params = dict(params or {})
        params.setdefault("per_page", 100)

        while True:
            response = self._get(endpoint, params)
            data = response.get("data", [])

            for record in data:
                yield record

            next_cursor = response.get("meta", {}).get("next_cursor")
            if not next_cursor or len(data) == 0:
                break

            params["cursor"] = next_cursor

    # ------------------------------------------------------------------
    # Endpoint methods
    # ------------------------------------------------------------------

    def get_games(self, dates: list[str] = None, seasons: list[int] = None,
                  team_ids: list[int] = None, season_type: str = None) -> Generator:
        """
        Fetch games. Pass dates=['2026-04-13'] for a single day,
        or seasons=[2025, 2026] for full seasons.
        """
        params = {}
        if dates:
            for i, d in enumerate(dates):
                params[f"dates[]"] = d  # requests will handle repeated keys
            # requests doesn't handle repeated keys cleanly — use list form
            params = {}
            if dates:
                params["dates[]"] = dates
        if seasons:
            params["seasons[]"] = seasons
        if team_ids:
            params["team_ids[]"] = team_ids
        if season_type:
            params["season_type"] = season_type
        yield from self.paginate("games", params)

    def get_plate_appearances(self, game_id: int) -> Generator:
        """Fetch all plate appearances for a specific game."""
        yield from self.paginate("plate_appearances", {"game_id": game_id})

    def get_plays(self, game_id: int) -> Generator:
        """Fetch play-by-play for a specific game."""
        yield from self.paginate("plays", {"game_id": game_id})

    def get_player_splits(self, player_id: int, season: int) -> dict:
        """Fetch splits for a player/season. Returns full response (not paginated)."""
        return self._get("players/splits", {"player_id": player_id, "season": season})

    def get_player_props(self, game_id: int = None, date: str = None) -> Generator:
        """Fetch player props, optionally filtered by game or date."""
        params = {}
        if game_id:
            params["game_id"] = game_id
        if date:
            params["date"] = date
        yield from self.paginate("odds/player_props", params)

    def get_betting_odds(self, game_id: int = None) -> Generator:
        """Fetch betting odds, optionally filtered by game."""
        params = {}
        if game_id:
            params["game_id"] = game_id
        yield from self.paginate("odds/betting_odds", params)

    def get_lineups(self, game_id: int) -> dict:
        """Fetch confirmed lineups for a game."""
        return self._get("lineups", {"game_id": game_id})

    def get_season_stats(self, season: int, player_ids: list[int] = None) -> Generator:
        """Fetch season-level stats."""
        params = {"season": season}
        if player_ids:
            params["player_ids[]"] = player_ids
        yield from self.paginate("season_stats", params)

    def get_player_injuries(self, team_ids: list[int] = None) -> Generator:
        """Fetch current injury report."""
        params = {}
        if team_ids:
            params["team_ids[]"] = team_ids
        yield from self.paginate("player_injuries", params)
