"""
NegBinom calibration analysis — deduplication-aware.
"""
import os, sys
import duckdb
import numpy as np
import pandas as pd
from scipy.stats import nbinom, binomtest
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

con = duckdb.connect(os.getenv('DB_PATH'), read_only=True)

df = con.execute("""
    SELECT
        prediction_id,
        game_id,
        player_id,
        predicted_value,
        line,
        edge,
        confidence,
        result        AS actual_ks,
        correct,
        predicted_at::DATE AS game_date,
        CASE WHEN edge >= 0 THEN 'OVER' ELSE 'UNDER' END AS direction,
        -- extract book from prediction_id (format: gameid_playerid_book)
        SPLIT_PART(prediction_id, '_', 3) AS book
    FROM model_predictions
    WHERE result IS NOT NULL
    ORDER BY game_date
""").df()

n_total = len(df)
n_games  = df['game_id'].nunique()
n_pitchers = df.groupby(['game_id','player_id']).ngroups
n_books  = df['book'].nunique()

print(f"\n{'='*65}")
print(f"  Sample composition")
print(f"{'='*65}")
print(f"  Total rows:               {n_total}")
print(f"  Distinct games:           {n_games}")
print(f"  Distinct pitcher-games:   {n_pitchers}")
print(f"  Distinct books:           {n_books}")
print(f"  Avg lines per pitcher-game: {n_total/n_pitchers:.1f}")
print()

print(f"  Book breakdown:")
book_counts = df.groupby('book').agg(
    n=('correct','count'),
    acc=('correct','mean'),
    avg_line=('line','mean'),
).sort_values('n', ascending=False)
print(book_counts.to_string())
print()

# ── Per-pitcher-game dedup: same μ, possibly different lines ───────────────
print(f"{'='*65}")
print(f"  Line variance across books (same pitcher-game)")
print(f"{'='*65}")
line_var = df.groupby(['game_id','player_id'])['line'].agg(['min','max','nunique','count'])
print(f"  Pitcher-games where all books have identical line: "
      f"{(line_var['nunique']==1).sum()} / {len(line_var)}")
print(f"  Pitcher-games with 2+ distinct lines:             "
      f"{(line_var['nunique']>1).sum()} / {len(line_var)}")
print(f"  Max line spread within a pitcher-game:            "
      f"{(line_var['max']-line_var['min']).max():.1f} Ks")
print()

# ── Run calibration on each book independently ────────────────────────────
def calibration_summary(sub: pd.DataFrame, label: str):
    n = len(sub)
    if n == 0:
        return
    acc  = sub['correct'].mean()
    mae  = (sub['predicted_value'] - sub['actual_ks']).abs().mean()
    bias = (sub['predicted_value'] - sub['actual_ks']).mean()
    brier = ((sub['confidence'] - sub['correct'].astype(float))**2).mean()
    brier_skill = 1 - brier/0.25

    print(f"\n  [{label}]  n={n}")
    print(f"  Accuracy: {acc:.1%}   MAE: {mae:.3f}   Bias: {bias:+.3f}")
    print(f"  Brier: {brier:.4f}   Brier skill: {brier_skill:.4f}")

    bins   = [0.0, 0.52, 0.55, 0.58, 0.62, 0.67, 0.75, 1.01]
    labels = ['<52%','52-55%','55-58%','58-62%','62-67%','67-75%','75%+']
    sub = sub.copy()
    sub['bucket'] = pd.cut(sub['confidence'], bins=bins, labels=labels, right=False)
    cal = sub.groupby('bucket', observed=True).agg(
        n=('correct','count'),
        avg_conf=('confidence','mean'),
        actual_acc=('correct','mean'),
    ).reset_index()

    print(f"  {'Tier':<10} {'N':>5}  {'Model P':>8}  {'Actual':>8}  {'Delta':>8}")
    print(f"  {'-'*50}")
    for _, row in cal.iterrows():
        if row['n'] == 0:
            continue
        delta = row['actual_acc'] - row['avg_conf']
        print(f"  {str(row['bucket']):<10} {int(row['n']):>5}  "
              f"{row['avg_conf']:>8.1%}  {row['actual_acc']:>8.1%}  {delta:>+8.1%}")

print(f"\n{'='*65}")
print(f"  Calibration by book (independent slices)")
print(f"{'='*65}")

priority_books = ['draftkings','betmgm','fanduel','fanatics','espnbet','caesars']
for book in priority_books:
    sub = df[df['book'] == book]
    if len(sub) >= 30:
        calibration_summary(sub, book)

# ── Deduplicated: one row per pitcher-game, using best available book ──────
print(f"\n\n{'='*65}")
print(f"  Deduplicated: one line per pitcher-game")
print(f"  Priority: draftkings > betmgm > fanduel > any")
print(f"{'='*65}")

def pick_book(group):
    for b in ['draftkings','betmgm','fanduel','fanatics','espnbet','caesars']:
        match = group[group['book'] == b]
        if not match.empty:
            return match.iloc[0]
    return group.iloc[0]

deduped = df.groupby(['game_id','player_id'], group_keys=False).apply(pick_book).reset_index(drop=True)
calibration_summary(deduped, f"Deduped (DK > MGM > FD priority), n={len(deduped)}")

con.close()
