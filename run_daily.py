"""
run_daily.py
------------
Daily workflow runner. Run once per day after yesterday's games are final
(typically morning ET).

Steps:
  1. ingest.py           — pull yesterday's completed games + plate appearances
  2. ingest_props.py     — pull today's K prop lines from sportsbooks
  3. ingest_kalshi.py    — pull today's Kalshi threshold markets
  4. log_results.py      — score yesterday's props against actual Ks
  5. predict_today.py    — generate today's predictions
  6. generate_data.py    — write dashboard JSON

Usage:
    python run_daily.py
    python run_daily.py --date 2026-05-02   # run as if it were this date
    python run_daily.py --min-ev 5.0        # raise EV threshold for predictions
    python run_daily.py --skip-ingest       # skip game ingestion (already done)
"""

import argparse
import subprocess
import sys
from datetime import date, timedelta


def run(cmd: list, label: str, abort_on_fail: bool = True) -> bool:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, env={**__import__('os').environ, 'PYTHONIOENCODING': 'utf-8'})
    if result.returncode != 0:
        print(f"\n  ERROR: {label} failed (exit {result.returncode})")
        if abort_on_fail:
            sys.exit(1)
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date',        type=str,   default=None,
                        help='Target date YYYY-MM-DD (default: today)')
    parser.add_argument('--min-ev',      type=float, default=3.0,
                        help='Minimum EV per $100 to flag a bet (default: 3.0)')
    parser.add_argument('--skip-ingest', action='store_true',
                        help='Skip game ingestion (if already run separately)')
    args = parser.parse_args()

    today     = args.date or str(date.today())
    yesterday = str((date.fromisoformat(today) - timedelta(days=1)))
    py        = sys.executable

    print(f"\nMLB Daily Workflow — {today}")
    print(f"  Scoring:    {yesterday}")
    print(f"  Predicting: {today}")

    step = 1
    total = 5 if args.skip_ingest else 6

    if not args.skip_ingest:
        run([py, 'ingestion/ingest.py', '--date', yesterday],
            f"Step {step}/{total} — Ingest completed games ({yesterday})")
        step += 1

    run([py, 'ingestion/ingest_props.py'],
        f"Step {step}/{total} — Ingest today's sportsbook lines ({today})")
    step += 1

    # Kalshi failure is non-fatal — markets may not be posted yet
    run([py, 'ingestion/ingest_kalshi.py', '--date', today],
        f"Step {step}/{total} — Ingest today's Kalshi markets ({today})",
        abort_on_fail=False)
    step += 1

    run([py, 'scripts/log_results.py', '--date', yesterday],
        f"Step {step}/{total} — Score yesterday's props ({yesterday})")
    step += 1

    run([py, 'scripts/predict_today.py', '--date', today, '--min-ev', str(args.min_ev)],
        f"Step {step}/{total} — Generate today's predictions ({today})")
    step += 1

    run([py, 'scripts/generate_data.py'],
        f"Step {step}/{total} — Write dashboard data")

    print(f"\n{'='*60}")
    print(f"  Done. Promote docs/data/ to repo to update dashboard.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
