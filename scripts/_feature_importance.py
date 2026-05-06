"""One-time analysis: GBM feature importances + SHAP values + descriptions -> CSV."""
import pickle, numpy as np, pandas as pd, shap, sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

DESCRIPTIONS = {
    'strikeouts_last10':      ('Rolling form',  'Total Ks over last 10 starts'),
    'strikeouts_last5':       ('Rolling form',  'Total Ks over last 5 starts'),
    'strikeouts_last3':       ('Rolling form',  'Total Ks over last 3 starts'),
    'k_rate_last10':          ('Rolling form',  'K/PA rate over last 10 starts'),
    'k_rate_last5':           ('Rolling form',  'K/PA rate over last 5 starts'),
    'k_rate_last3':           ('Rolling form',  'K/PA rate over last 3 starts'),
    'swstr_pct_last5':        ('Rolling form',  'Swinging strike % over last 5 starts'),
    'swstr_pct_last3':        ('Rolling form',  'Swinging strike % over last 3 starts'),
    'zone_pct_last5':         ('Rolling form',  'Pitches in strike zone % over last 5 starts'),
    'zone_pct_last3':         ('Rolling form',  'Pitches in strike zone % over last 3 starts'),
    'bb_rate_last5':          ('Rolling form',  'Walk rate (BB/PA) over last 5 starts'),
    'bb_rate_last3':          ('Rolling form',  'Walk rate (BB/PA) over last 3 starts'),
    'pitch_count_last5':      ('Rolling form',  'Average pitch count over last 5 starts'),
    'pitch_count_last3':      ('Rolling form',  'Average pitch count over last 3 starts'),
    'ff_avg_velo':            ('Stuff grades',  '4-seam fastball average velocity (mph, raw)'),
    'ff_velo_z':              ('Stuff grades',  '4-seam fastball velocity z-score vs league avg'),
    'ff_spin_z':              ('Stuff grades',  '4-seam fastball spin rate z-score vs league avg'),
    'ff_whiff_z':             ('Stuff grades',  '4-seam fastball whiff rate z-score vs league avg'),
    'sl_velo_z':              ('Stuff grades',  'Slider velocity z-score vs league avg'),
    'sl_spin_z':              ('Stuff grades',  'Slider spin rate z-score vs league avg'),
    'sl_whiff_z':             ('Stuff grades',  'Slider whiff rate z-score vs league avg'),
    'sl_ivb_z':               ('Stuff grades',  'Slider induced vertical break z-score vs league avg'),
    'ch_velo_z':              ('Stuff grades',  'Changeup velocity z-score vs league avg'),
    'ch_whiff_z':             ('Stuff grades',  'Changeup whiff rate z-score vs league avg'),
    'ch_ivb_z':               ('Stuff grades',  'Changeup induced vertical break z-score vs league avg'),
    'cu_spin_z':              ('Stuff grades',  'Curveball spin rate z-score vs league avg'),
    'cu_whiff_z':             ('Stuff grades',  'Curveball whiff rate z-score vs league avg'),
    'si_velo_z':              ('Stuff grades',  'Sinker velocity z-score vs league avg'),
    'si_whiff_z':             ('Stuff grades',  'Sinker whiff rate z-score vs league avg'),
    'mix_ff_last3':           ('Pitch mix',     '4-seam fastball usage % over last 3 starts'),
    'mix_sl_last3':           ('Pitch mix',     'Slider usage % over last 3 starts'),
    'mix_ch_last3':           ('Pitch mix',     'Changeup usage % over last 3 starts'),
    'mix_cu_last3':           ('Pitch mix',     'Curveball usage % over last 3 starts'),
    'mix_si_last3':           ('Pitch mix',     'Sinker usage % over last 3 starts'),
    'lineup_k_rate_weighted': ('Lineup',        'Opposing lineup K rate weighted by batting order position'),
    'lineup_k_rate_raw':      ('Lineup',        'Opposing lineup raw K rate (unweighted average)'),
    'park_k_factor':          ('Context',       'Ballpark K rate multiplier vs league average'),
    'pitcher_hand_enc':       ('Context',       'Pitcher handedness (0=L, 1=R)'),
    'days_rest':              ('Context',       'Days since last start'),
    'month':                  ('Context',       'Calendar month (1=Apr, captures seasonal fatigue)'),
}

with open('models/saved/k_prop_gbm_20260501_1708.pkl', 'rb') as f:
    obj = pickle.load(f)

model    = obj['model']
features = obj['features']

from features.pitcher_features import build_pitcher_feature_matrix
from features.lineup_features  import build_lineup_feature_matrix

DB_PATH = os.getenv('DB_PATH')
print('Building feature matrix...')
pitcher_df = build_pitcher_feature_matrix(DB_PATH)
lineup_df  = build_lineup_feature_matrix(DB_PATH)
df = pitcher_df.merge(lineup_df, on=['pitcher_id', 'game_id'], how='left')
df = df.dropna(subset=['strikeouts', 'k_rate_last3'])
X  = df[features].fillna(0)

sample = X.sample(500, random_state=42)
print(f'Running SHAP TreeExplainer on {len(sample)} rows...')
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(sample)

mean_abs_shap = np.abs(shap_values).mean(axis=0)

# Direction: correlate feature value with its SHAP value (positive = higher value raises K prediction)
shap_direction = []
for i in range(len(features)):
    corr = np.corrcoef(sample.iloc[:, i].values, shap_values[:, i])[0, 1]
    shap_direction.append(corr)

results = pd.DataFrame({
    'feature':           features,
    'group':             [DESCRIPTIONS.get(f, ('Unknown', ''))[0] for f in features],
    'description':       [DESCRIPTIONS.get(f, ('', f))[1] for f in features],
    'gbm_importance':    model.feature_importances_,
    'shap_mean_abs':     mean_abs_shap,
    'shap_value_corr':   shap_direction,
}).sort_values('shap_mean_abs', ascending=False).reset_index(drop=True)

results.insert(0, 'shap_rank', range(1, len(results) + 1))

# Round for readability
results['gbm_importance'] = results['gbm_importance'].round(4)
results['shap_mean_abs']  = results['shap_mean_abs'].round(4)
results['shap_value_corr'] = results['shap_value_corr'].round(3)

# Console output
print()
print(f"{'#':<4} {'Feature':<35} {'Group':<14} {'GBM Imp':>8} {'SHAP abs':>9} {'Corr':>6}  Description")
print('-' * 110)
for _, row in results.iterrows():
    print(
        f"{row['shap_rank']:<4} {row['feature']:<35} {row['group']:<14} "
        f"{row['gbm_importance']:>8.4f} {row['shap_mean_abs']:>9.4f} {row['shap_value_corr']:>6.3f}  {row['description']}"
    )

out_path = 'docs/data/feature_importance.csv'
results.to_csv(out_path, index=False)
print(f'\nSaved: {out_path}')
