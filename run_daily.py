"""
run_daily.py
------------
Daily workflow runner. Run once per day after yesterday's games are final
(typically morning ET).

Steps:
  1. ingest.py         — pull yesterday's completed games + plate appearances
  2. ingest_props.py   — pull today's K prop lines from books
  3. log_results.py    — score yesterday's props against actual Ks
  4. predict_today.py  — generate today's predictions
  5. generate_data.py  — write dashboard JSON

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


def run(cmd: list, label: str) -> bool:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, env={**__import__('os').environ, 'PYTHONIOENCODING': 'utf-8'})
    if result.returncode != 0:
        print(f"\n  ERROR: {label} failed (exit {result.returncode})")
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date',         type=str,  default=None,
                        help='Target date YYYY-MM-DD (default: today)')
    parser.add_argument('--min-ev',       type=float, default=3.0,
                        help='Minimum EV per $100 to flag a bet (default: 3.0)')
    parser.add_argument('--skip-ingest',  action='store_true',
                        help='Skip game ingestion (if already run separately)')
    args = parser.parse_args()

    today     = args.date or str(date.today())
    yesterday = str((date.fromisoformat(today) - timedelta(days=1)))

    print(f"\nMLB Daily Workflow — {today}")
    print(f"  Scoring:    {yesterday}")
    print(f"  Predicting: {today}")

    py = sys.executable

    if not args.skip_ingest:
        ok = run(
            [py, 'ingestion/ingest.py', '--date', yesterday],
            f"Step 1/5 — Ingest completed games ({yesterday})"
        )
        if not ok:
            print("\nAborting — game ingestion failed.")
            sys.exit(1)

    steps = [
        ([py, 'ingestion/ingest_props.py'],
         f"Step {'2' if not args.skip_ingest else '1'}/{'5' if not args.skip_ingest else '4'} — Ingest today's prop lines ({today})"),

        ([py, 'scripts/log_results.py', '--date', yesterday],
         f"Step {'3' if not args.skip_ingest else '2'}/{'5' if not args.skip_ingest else '4'} — Score yesterday's props ({yesterday})"),

        ([py, 'scripts/predict_today.py', '--date', today, '--min-ev', str(args.min_ev)],
         f"Step {'4' if not args.skip_ingest else '3'}/{'5' if not args.skip_ingest else '4'} — Generate today's predictions ({today})"),

        ([py, 'scripts/generate_data.py'],
         f"Step {'5' if not args.skip_ingest else '4'}/{'5' if not args.skip_ingest else '4'} — Write dashboard data"),
    ]

    for cmd, label in steps:
        ok = run(cmd, label)
        if not ok:
            print(f"\nAborting at: {label}")
            sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Done. Promote docs/data/ to repo to update dashboard.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
