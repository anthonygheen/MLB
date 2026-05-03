import json
with open('docs/data/predictions.json') as f:
    d = json.load(f)

print('Top plays:')
for p in d.get('top_plays', []):
    name = p['pitcher_name']
    kpct = p['kelly_pct']
    kdir = p['kelly_dir']
    print(f'  {name} | kelly_pct={kpct}% | dir={kdir}')

print()
print('All flagged predictions:')
for p in d['predictions']:
    if p.get('flagged'):
        name = p['pitcher_name']
        kpct = p['kelly_pct']
        kdir = p['kelly_dir']
        book = p['book']
        print(f'  {name} [{book}]: kelly_pct={kpct}%, dir={kdir}')
