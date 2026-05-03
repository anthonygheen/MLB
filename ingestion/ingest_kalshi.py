"""
ingestion/ingest_kalshi.py
--------------------------
Pulls MLB pitcher strikeout markets from Kalshi's public API and inserts
them into the player_props table.

Kalshi data model notes:
  - Series ticker: KXMLBKS  (MLB pitcher strikeouts)
  - Markets are binary: "Aaron Nola: 6+ strikeouts?" — Yes/No
  - Each pitcher has many markets at different thresholds (3+, 4+, 5+ ...)
  - Prices are in USD dollars 0.0–1.0 (NOT cent integers)
  - Mid price = (yes_bid + yes_ask) / 2  (or yes_ask if no bid)
  - No auth needed for read-only market data

Strategy for choosing the "line":
  - We look at ALL threshold markets for a pitcher in one game event.
  - We pick the threshold whose Yes mid-price is closest to 0.50 (fair coin).
    That threshold IS the line — the market is saying ~50% chance pitcher
    records exactly that many strikeouts.
  - We record over_odds = American odds for Yes (Over) side,
    under_odds = American odds for No (Under) side.

Conversion: Kalshi dollar price → American odds
  - p_over = yes_mid  (0.0–1.0)
  - p_under = 1 - p_over
  - if p >= 0.5: american = -round(p / (1-p) * 100)
  - if p <  0.5: american = +round((1-p) / p * 100)

Usage:
    python ingestion/ingest_kalshi.py                  # today
    python ingestion/ingest_kalshi.py --date 2026-05-03
    python ingestion/ingest_kalshi.py --debug
"""

import os
import sys
import re
import argparse
import logging
import requests
from datetime import date, datetime, timezone
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingestion.schema import get_connection, init_db

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DB_PATH   = os.getenv("DB_PATH", "data/mlb_modeling.db")
BASE_URL  = "https://api.elections.kalshi.com/trade-api/v2"
SERIES    = "KXMLBKS"          # MLB pitcher strikeouts series
PAGE_SIZE = 200                # max per Kalshi page

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Kalshi API helpers
# ---------------------------------------------------------------------------

def _build_session() -> requests.Session:
    """
    Return a requests Session. Kalshi market reads are public (no auth needed).
    If KALSHI_API_KEY is set we attach it per their RSA-key header scheme,
    but the read-only market endpoints work without any credentials.
    """
    session = requests.Session()
    session.headers.update({"Accept": "application/json"})
    api_key = os.getenv("KALSHI_API_KEY")
    if api_key:
        session.headers["KALSHI-ACCESS-KEY"] = api_key
        log.debug("Using KALSHI_API_KEY from environment.")
    return session


def fetch_all_markets(session: requests.Session, series_ticker: str) -> list[dict]:
    """
    Paginate through GET /markets for the given series and return every market.
    Kalshi uses cursor-based pagination; empty cursor means last page.
    """
    markets = []
    cursor  = ""
    page    = 0

    while True:
        page += 1
        params: dict = {
            "series_ticker": series_ticker,
            "status":        "open",
            "limit":         PAGE_SIZE,
        }
        if cursor:
            params["cursor"] = cursor

        resp = session.get(f"{BASE_URL}/markets", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        batch  = data.get("markets", [])
        markets.extend(batch)
        cursor = data.get("cursor", "")

        log.debug("Page %d: fetched %d markets (cursor=%r)", page, len(batch), cursor)

        # Stop when Kalshi returns empty cursor or empty page
        if not cursor or not batch:
            break

    return markets


# ---------------------------------------------------------------------------
# Pricing helpers
# ---------------------------------------------------------------------------

def _mid(bid_str: str | None, ask_str: str | None) -> float | None:
    """
    Compute the mid-price from bid/ask dollar strings.

    Requires a genuine two-sided market: bid >= $0.05 and ask <= $0.95.
    Minimum-tick quotes (bid=$0.01, ask=$0.99) are market-maker placeholders
    and produce a spurious mid of $0.50 — we discard them.
    """
    try:
        bid = float(bid_str or 0)
        ask = float(ask_str or 0)
    except (TypeError, ValueError):
        return None

    # Reject placeholder / minimum-tick quotes
    if bid < 0.05 or ask > 0.95:
        return None

    return (bid + ask) / 2.0


def price_to_american(p: float) -> int:
    """
    Convert an implied probability (0 < p < 1) to American odds (integer).
      p >= 0.5  →  negative (favourite):  -round(p / (1-p) * 100)
      p <  0.5  →  positive (underdog):   +round((1-p) / p * 100)
    """
    p = max(0.001, min(0.999, p))   # guard against divide-by-zero
    if p >= 0.5:
        return -round(p / (1.0 - p) * 100)
    else:
        return round((1.0 - p) / p * 100)


# ---------------------------------------------------------------------------
# Date filtering helpers
# ---------------------------------------------------------------------------

# Ticker date segment examples:  26MAY03  26MAY04  26APR28
_DATE_RE = re.compile(
    r"KXMLBKS-(\d{2})([A-Z]{3})(\d{2})",
    re.IGNORECASE,
)

_MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
    "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
    "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def ticker_date(event_ticker: str) -> date | None:
    """
    Parse the game date from a KXMLBKS event_ticker.
    e.g. 'KXMLBKS-26MAY031607NYMLAA' → date(2026, 5, 3)
    """
    m = _DATE_RE.match(event_ticker)
    if not m:
        return None
    yy, mon, dd = int(m.group(1)), m.group(2).upper(), int(m.group(3))
    month = _MONTH_MAP.get(mon)
    if not month:
        return None
    return date(2000 + yy, month, dd)


def event_ticker_from_market(mkt: dict) -> str:
    return mkt.get("event_ticker", "")


def threshold_from_ticker(ticker: str) -> int | None:
    """
    Extract the strikeout threshold from the market ticker.
    e.g. 'KXMLBKS-26MAY041840PHIMIA-PHIANOLA27-6'  → 6
    """
    parts = ticker.rsplit("-", 1)
    if len(parts) == 2:
        try:
            return int(parts[-1])
        except ValueError:
            pass
    return None


def pitcher_name_from_title(title: str) -> str | None:
    """
    Extract pitcher name from market title.
    e.g. 'Aaron Nola: 6+ strikeouts?' → 'Aaron Nola'
    """
    if ":" in title:
        return title.split(":")[0].strip()
    return None


# ---------------------------------------------------------------------------
# Player matching
# ---------------------------------------------------------------------------

def load_players(con) -> list[dict]:
    """Return all players as dicts with player_id and full_name."""
    rows = con.execute(
        "SELECT player_id, full_name FROM players WHERE full_name IS NOT NULL"
    ).fetchall()
    return [{"player_id": r[0], "full_name": r[1]} for r in rows]


def _name_tokens(name: str) -> set[str]:
    """Lower-cased surname + first-name tokens for fuzzy matching."""
    return set(name.lower().split())


def match_player(name: str, players: list[dict]) -> int | None:
    """
    Fuzzy match a Kalshi pitcher name to a player_id.

    Strategy (in order):
      1. Exact case-insensitive full-name match.
      2. All tokens in Kalshi name appear in DB name (handles middle names).
      3. Last name + first initial match.
    Returns player_id or None.
    """
    name_lower = name.lower().strip()
    tokens     = _name_tokens(name)

    # 1. Exact match
    for p in players:
        if p["full_name"].lower().strip() == name_lower:
            return p["player_id"]

    # 2. Token overlap (all Kalshi tokens present in DB name)
    for p in players:
        db_tokens = _name_tokens(p["full_name"])
        if tokens and tokens.issubset(db_tokens):
            return p["player_id"]

    # 3. Last name + first initial
    parts = name.split()
    if len(parts) >= 2:
        last  = parts[-1].lower()
        first_initial = parts[0][0].lower()
        for p in players:
            db_parts = p["full_name"].split()
            if len(db_parts) >= 2:
                if (db_parts[-1].lower() == last
                        and db_parts[0][0].lower() == first_initial):
                    return p["player_id"]

    return None


# ---------------------------------------------------------------------------
# Game matching
# ---------------------------------------------------------------------------

def load_games_for_date(con, game_date: str) -> list[dict]:
    """
    Load games for a given YYYY-MM-DD date string.
    Returns list of dicts with game_id, home_team_name, away_team_name.
    """
    rows = con.execute("""
        SELECT game_id, home_team_name, away_team_name
        FROM games
        WHERE CAST(game_date AS DATE) = ?
    """, [game_date]).fetchall()
    return [
        {"game_id": r[0], "home_team_name": r[1], "away_team_name": r[2]}
        for r in rows
    ]


def _abbr_from_event_ticker(event_ticker: str) -> tuple[str, str] | None:
    """
    Pull the two 3-letter team abbreviations from the event ticker suffix.
    e.g. 'KXMLBKS-26MAY041840PHIMIA'  →  ('PHI', 'MIA')
         'KXMLBKS-26MAY031335HOUBOS'  →  ('HOU', 'BOS')
         'KXMLBKS-26MAY031607NYMLAA'  →  ('NYM', 'LAA')
    The trailing segment after the time has variable-length 3-char codes.
    """
    # Strip the standard prefix and date-time portion
    # Format:  KXMLBKS-[YY][MON][DD][HHMM][AWAY][HOME]
    # The team segment is everything after stripping KXMLBKS-YYMMDDHHM
    m = re.search(
        r"KXMLBKS-\d{2}[A-Z]{3}\d{2}\d{4}([A-Z]{3,6})$",
        event_ticker,
        re.IGNORECASE,
    )
    if not m:
        return None
    team_str = m.group(1).upper()
    # Split into two equal halves — both codes are the same length
    half = len(team_str) // 2
    if half < 2:
        return None
    return team_str[:half], team_str[half:]


def match_game(event_ticker: str, games: list[dict]) -> int | None:
    """
    Try to match a Kalshi event_ticker to a game_id via team abbreviations.
    Returns game_id or None.
    """
    codes = _abbr_from_event_ticker(event_ticker)
    if not codes:
        return None
    away_abbr, home_abbr = codes[0].upper(), codes[1].upper()

    for g in games:
        home = (g["home_team_name"] or "").upper()
        away = (g["away_team_name"] or "").upper()
        # Match if either abbreviation appears in either team name
        if (away_abbr in home or away_abbr in away) and \
           (home_abbr in home or home_abbr in away):
            return g["game_id"]

    return None


# ---------------------------------------------------------------------------
# Core pivot: markets → one prop row per pitcher per game
# ---------------------------------------------------------------------------

def build_props(markets: list[dict], target_date: date) -> list[dict]:
    """
    Return one row per liquid Kalshi threshold market for target_date.

    Unlike sportsbooks, Kalshi posts a separate binary contract for every
    integer threshold (3+, 4+, 5+, 6+ Ks...).  We store ALL liquid markets
    so the pipeline can evaluate each threshold against the model's NegBinom
    distribution and find the one with the highest EV — rather than
    collapsing to an artificial single "line."

    Liquidity filter: requires a genuine two-sided quote (yes_bid >= $0.05).
    """
    props = []

    for mkt in markets:
        et    = event_ticker_from_market(mkt)
        mdate = ticker_date(et)
        if mdate != target_date:
            continue

        title = mkt.get("title", "")
        name  = pitcher_name_from_title(title)
        if not name:
            continue

        ticker    = mkt.get("ticker", "")
        threshold = threshold_from_ticker(ticker)
        if threshold is None:
            continue

        yes_mid = _mid(mkt.get("yes_bid_dollars"), mkt.get("yes_ask_dollars"))
        if yes_mid is None:
            continue  # illiquid — skip

        no_mid = 1.0 - yes_mid

        # Use floor_strike (half-point, e.g. 4.5 for "5+" market) when available.
        # This matches the NegBinom CDF computation: P(K > floor(line)).
        floor_strike = mkt.get("floor_strike")
        line = float(floor_strike) if floor_strike is not None else float(threshold - 1) + 0.5

        props.append({
            "event_ticker":  et,
            "pitcher_name":  name,
            "ticker":        ticker,
            "threshold":     threshold,   # integer, e.g. 6 for "6+ Ks"
            "line":          line,        # half-point line, e.g. 5.5
            "yes_mid":       yes_mid,
            "over_odds":     price_to_american(yes_mid),
            "under_odds":    price_to_american(no_mid),
        })

    return props


# ---------------------------------------------------------------------------
# Main ingestion function
# ---------------------------------------------------------------------------

def ingest_kalshi_for_date(target_date: str, debug: bool = False):
    init_db()
    con = get_connection()

    log.info("Pulling Kalshi MLB K markets for %s ...", target_date)

    # 1. Fetch all open KXMLBKS markets
    session = _build_session()
    try:
        markets = fetch_all_markets(session, SERIES)
    except requests.HTTPError as exc:
        log.error("HTTP error fetching Kalshi markets: %s", exc)
        con.close()
        return
    except requests.RequestException as exc:
        log.error("Network error fetching Kalshi markets: %s", exc)
        con.close()
        return

    if not markets:
        log.warning("No open %s markets returned from Kalshi — nothing to ingest.", SERIES)
        con.close()
        return

    log.info("Fetched %d raw markets from Kalshi.", len(markets))

    # 2. Load reference data
    try:
        td = date.fromisoformat(target_date)
    except ValueError:
        log.error("Invalid date format: %r (expected YYYY-MM-DD)", target_date)
        con.close()
        return

    players   = load_players(con)
    games     = load_games_for_date(con, target_date)

    if debug:
        log.info("Loaded %d players, %d games from DB for %s.", len(players), len(games), target_date)

    # 3. Clear stale Kalshi rows for this date before re-ingesting.
    # prop_ids changed format between old (one per pitcher) and new (one per threshold),
    # so INSERT OR REPLACE doesn't overwrite old rows — we delete first for a clean slate.
    deleted = con.execute(
        f"DELETE FROM player_props WHERE book='kalshi' "
        f"AND DATE_TRUNC('day', recorded_at)::DATE = '{target_date}'::DATE"
    ).rowcount
    if deleted:
        log.info("Cleared %d stale Kalshi rows for %s.", deleted, target_date)
    con.commit()

    # 4. Build one prop per liquid threshold per pitcher
    props = build_props(markets, td)

    if not props:
        log.warning(
            "No markets found for %s. "
            "Check that Kalshi has posted lines for today's games.", target_date
        )
        con.close()
        return

    log.info("Found %d pitcher markets for %s.", len(props), target_date)

    # 4. Match pitchers and games, then insert
    inserted   = 0
    skipped    = 0
    no_game    = 0

    for prop in props:
        pitcher_name = prop["pitcher_name"]
        event_ticker = prop["event_ticker"]
        ticker       = prop["ticker"]

        # Player match
        player_id = match_player(pitcher_name, players)
        if player_id is None:
            log.warning("Pitcher not matched in DB: %r (skipping)", pitcher_name)
            skipped += 1
            continue

        # Game match (best-effort; NULL is acceptable)
        game_id = match_game(event_ticker, games)
        if game_id is None:
            log.debug("No game match for event_ticker %r — game_id will be NULL.", event_ticker)
            no_game += 1

        prop_id = f"kalshi_{ticker}_{player_id}"

        if debug:
            log.info(
                "  %s  player_id=%s  game_id=%s  line=%.1f  "
                "yes_mid=%.3f  over=%+d  under=%+d",
                pitcher_name, player_id, game_id,
                prop["line"], prop["yes_mid"],
                prop["over_odds"], prop["under_odds"],
            )

        try:
            con.execute("""
                INSERT OR REPLACE INTO player_props (
                    prop_id, game_id, player_id, book,
                    market_type, line, over_odds, under_odds,
                    recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                prop_id,
                game_id,
                player_id,
                "kalshi",
                "pitcher_strikeouts",
                prop["line"],
                prop["over_odds"],
                prop["under_odds"],
                datetime.now(timezone.utc).isoformat(),
            ])
            inserted += 1
        except Exception as exc:
            log.error("Insert error for %r: %s", pitcher_name, exc)

    con.commit()
    con.close()

    log.info(
        "Done. Inserted=%d  skipped_no_player=%d  no_game_match=%d",
        inserted, skipped, no_game,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ingest Kalshi MLB pitcher K markets")
    parser.add_argument(
        "--date", type=str, default=None,
        help="Date YYYY-MM-DD (default: today)"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Show detailed per-pitcher logging"
    )
    args = parser.parse_args()

    target_date = args.date or str(date.today())
    ingest_kalshi_for_date(target_date, debug=args.debug)


if __name__ == "__main__":
    main()
