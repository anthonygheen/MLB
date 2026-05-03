import json
with open('docs/data/sharp_markets.json') as f:
    d = json.load(f)

print(f"Total markets: {len(d['markets'])}  Flagged: {d['flagged']}")
print()
for m in d['markets']:
    name     = m['pitcher_name']
    thresh   = m['threshold']
    side     = m['best_side']
    mu       = m['predicted_mu']
    p        = m['best_p']
    odds     = m['best_odds']
    ev       = m['best_ev']
    kelly    = m['best_kelly_pct']
    flagged  = m['flagged']
    flag_str = ' *** FLAGGED' if flagged else ''
    print(f"  {name} {thresh}+ Ks | {side} {odds:+d} | model={mu} | P={p:.1%} | EV={ev*100:+.2f}/100 | kelly={kelly}%{flag_str}")
