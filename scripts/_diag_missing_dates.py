import duckdb, os
from dotenv import load_dotenv
load_dotenv()
con = duckdb.connect(os.getenv('DB_PATH'), read_only=True)

print('=== Prop dates with no game data ===')
df = con.execute("""
    SELECT DATE_TRUNC('day', pp.recorded_at)::DATE AS prop_date,
           COUNT(DISTINCT pp.player_id) AS pitchers,
           COUNT(DISTINCT g.game_id) AS games_with_data
    FROM player_props pp
    LEFT JOIN games g ON pp.game_id = g.game_id AND g.status = 'STATUS_FINAL'
    WHERE pp.market_type = 'pitcher_strikeouts'
    GROUP BY 1
    ORDER BY 1
""").df()
print(df.to_string(index=False))

print()
print('=== model_predictions by date ===')
df2 = con.execute("""
    SELECT predicted_at::DATE AS date, COUNT(*) AS logged,
           ROUND(AVG(CASE WHEN correct THEN 1.0 ELSE 0.0 END)*100,1) AS accuracy_pct
    FROM model_predictions
    GROUP BY 1
    ORDER BY 1
""").df()
print(df2.to_string(index=False))
