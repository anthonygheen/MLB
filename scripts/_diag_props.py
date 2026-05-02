import duckdb, os, requests
from dotenv import load_dotenv
load_dotenv()

con = duckdb.connect(os.getenv('DB_PATH'), read_only=True)

print('=== Props for 2026-05-02 (first 20) ===')
df = con.execute("""
    SELECT pp.player_id, pl.full_name, pp.line, pp.book
    FROM player_props pp
    LEFT JOIN players pl ON pp.player_id = pl.player_id
    WHERE market_type = 'pitcher_strikeouts'
      AND DATE_TRUNC('day', recorded_at) = '2026-05-02'::DATE
    GROUP BY pp.player_id, pl.full_name, pp.line, pp.book
    ORDER BY pp.player_id
    LIMIT 20
""").df()
print(df.to_string(index=False))

print()
print('=== Confirmed starters from API ===')
API_KEY  = os.getenv('BDL_API_KEY')
BASE_URL = 'https://api.balldontlie.io/mlb/v1'
headers  = {'Authorization': API_KEY}

games = requests.get(f'{BASE_URL}/games', headers=headers,
    params={'dates[]': '2026-05-02', 'season_type': 'regular', 'per_page': 100}
).json().get('data', [])
gids = [g['id'] for g in games if g.get('status') in ('STATUS_SCHEDULED','scheduled','STATUS_IN_PROGRESS')]

lineups = requests.get(f'{BASE_URL}/lineups', headers=headers,
    params={'game_id': gids[0], 'per_page': 100}
).json().get('data', [])

for e in lineups:
    if e.get('is_probable_pitcher'):
        pid  = e.get('player', {}).get('id')
        name = e.get('player', {}).get('full_name')
        team = e.get('team', {}).get('name')
        print(f'  id={pid}  {name}  ({team})')
