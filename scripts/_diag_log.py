import duckdb, os
from dotenv import load_dotenv
load_dotenv()
con = duckdb.connect(os.getenv('DB_PATH'), read_only=True)

print('=== Games for May 2 ===')
df = con.execute("""
    SELECT game_id, game_date, status, home_team_name, away_team_name
    FROM games WHERE game_date::DATE = '2026-05-02'::DATE
""").df()
print(df.to_string(index=False) if not df.empty else 'No rows found')

print()
print('=== model_predictions count ===')
print(con.execute('SELECT COUNT(*) FROM model_predictions').fetchone()[0])

print()
print('=== plate_appearances for May 2 ===')
count = con.execute("""
    SELECT COUNT(*) FROM plate_appearances pa
    JOIN games g ON pa.game_id = g.game_id
    WHERE g.game_date::DATE = '2026-05-02'::DATE
""").fetchone()[0]
print(count, 'rows')

print()
print('=== player_props for May 2 (sample) ===')
df2 = con.execute("""
    SELECT pp.game_id, pp.player_id, pp.line, g.game_id AS games_game_id, g.status
    FROM player_props pp
    LEFT JOIN games g ON pp.game_id = g.game_id
    WHERE DATE_TRUNC('day', pp.recorded_at) = '2026-05-02'::DATE
    LIMIT 5
""").df()
print(df2.to_string(index=False) if not df2.empty else 'No rows found')
