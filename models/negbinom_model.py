"""
models/negbinom_model.py
-------------------------
Negative Binomial regression model for pitcher strikeout prop prediction.

Why Negative Binomial over Poisson:
  - K totals are overdispersed: variance > mean (empirically true for starter Ks)
  - Poisson forces var == mean, underestimates tails
  - NegBinom adds a dispersion parameter alpha to model the extra variance
  - Better calibrated over/under probabilities, especially for high-K pitchers

Outputs the same evaluation metrics as k_prop_model.py for direct comparison.

Usage:
    python models/negbinom_model.py                # train and evaluate
    python models/negbinom_model.py --cv-only      # cross-validation only
    python models/negbinom_model.py --compare      # run alongside GBM for comparison
"""

import os
import sys
import argparse
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats
from scipy.stats import nbinom

import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.pitcher_features import build_pitcher_feature_matrix
from features.lineup_features import build_lineup_feature_matrix

DB_PATH    = os.getenv("DB_PATH", "../data/mlb_modeling.db")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "saved")
os.makedirs(MODELS_DIR, exist_ok=True)

TARGET = 'strikeouts'

# Same feature set as k_prop_model.py for apples-to-apples comparison
ROLLING_FEATURES = [
    'k_rate_last3', 'k_rate_last5', 'k_rate_last10',
    'swstr_pct_last3', 'swstr_pct_last5',
    'zone_pct_last3', 'zone_pct_last5',
    'bb_rate_last3', 'bb_rate_last5',
    'strikeouts_last3', 'strikeouts_last5', 'strikeouts_last10',
    'pitch_count_last3', 'pitch_count_last5',
]
STUFF_FEATURES = [
    'ff_avg_velo', 'ff_velo_z', 'ff_spin_z', 'ff_whiff_z',
    'sl_velo_z', 'sl_spin_z', 'sl_whiff_z', 'sl_ivb_z',
    'ch_velo_z', 'ch_whiff_z', 'ch_ivb_z',
    'cu_spin_z', 'cu_whiff_z',
    'si_velo_z', 'si_whiff_z',
]
MIX_FEATURES = [
    'mix_ff_last3', 'mix_sl_last3', 'mix_ch_last3',
    'mix_cu_last3', 'mix_si_last3',
]
LINEUP_FEATURES = [
    'lineup_k_rate_weighted',
    'lineup_k_rate_raw',
]
CONTEXT_FEATURES = [
    'park_k_factor',
    'pitcher_hand_enc',
    'days_rest',
    'month',
]
ALL_FEATURES = (ROLLING_FEATURES + STUFF_FEATURES +
                MIX_FEATURES + LINEUP_FEATURES + CONTEXT_FEATURES)


# ------------------------------------------------------------------
# Data assembly (mirrors k_prop_model.py)
# ------------------------------------------------------------------

def build_training_data(db_path: str = None):
    path = db_path or DB_PATH

    print("=" * 60)
    print("BUILDING TRAINING DATA")
    print("=" * 60)

    pitcher_df = build_pitcher_feature_matrix(path)
    lineup_df  = build_lineup_feature_matrix(path)

    print("\n🔗 Joining pitcher and lineup features...")
    df = pitcher_df.merge(lineup_df, on=['pitcher_id', 'game_id'], how='left')
    df = df.dropna(subset=[TARGET])
    df = df[df['k_rate_last3'].notna()]

    for col in STUFF_FEATURES + MIX_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    df['lineup_k_rate_weighted'] = df.get('lineup_k_rate_weighted',
                                          pd.Series()).fillna(0.22)
    df['lineup_k_rate_raw']      = df.get('lineup_k_rate_raw',
                                          pd.Series()).fillna(0.22)

    available_features = [f for f in ALL_FEATURES if f in df.columns]
    df = df.sort_values('game_date').reset_index(drop=True)

    print(f"\n✅ Training data ready:")
    print(f"   Rows: {len(df):,}")
    print(f"   Features: {len(available_features)}")

    # Check overdispersion — core justification for NegBinom
    mean_k = df[TARGET].mean()
    var_k  = df[TARGET].var()
    print(f"\n📊 Target distribution:")
    print(f"   Mean:     {mean_k:.4f}")
    print(f"   Variance: {var_k:.4f}")
    print(f"   Variance/Mean ratio: {var_k/mean_k:.4f}  "
          f"({'overdispersed ✓ — NegBinom justified' if var_k > mean_k else 'not overdispersed'})")

    return df, available_features


# ------------------------------------------------------------------
# Negative Binomial model
# ------------------------------------------------------------------

class NegBinomModel:
    """
    Wrapper around statsmodels NegativeBinomial that mirrors
    sklearn's predict() interface for compatibility with evaluation code.
    """

    def __init__(self):
        self.result     = None
        self.scaler     = StandardScaler()
        self.alpha      = None   # dispersion parameter
        self.feature_names = None

    def fit(self, X, y, feature_names=None):
        self.feature_names = feature_names
        X_scaled = self.scaler.fit_transform(X)
        X_const  = sm.add_constant(X_scaled)

        model = NegativeBinomial(y, X_const)
        self.result = model.fit(
            method='bfgs',
            maxiter=200,
            disp=False
        )
        # Extract dispersion parameter alpha
        self.alpha = np.exp(self.result.params[-1]) if hasattr(self.result, 'params') else 1.0
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        X_const  = sm.add_constant(X_scaled, has_constant='add')
        return self.result.predict(X_const)

    def predict_distribution(self, X):
        """
        Returns mean and variance of the NegBinom distribution per prediction.
        Useful for computing over/under probabilities properly.
        """
        mu  = self.predict(X)
        # NegBinom variance: mu + alpha * mu^2
        var = mu + self.alpha * mu ** 2
        return mu, var

    def predict_ou_probability(self, X, line: np.ndarray):
        """
        Compute P(actual > line) using the full NegBinom distribution.
        This is the key advantage over GBM — proper probabilistic over/under.
        """
        mu  = self.predict(X)
        probs_over = []

        for i, (m, l) in enumerate(zip(mu, line)):
            # NegBinom parameterization: n = 1/alpha, p = n/(n+mu)
            if self.alpha and self.alpha > 0:
                n = 1.0 / self.alpha
                p = n / (n + m)
                # P(X > line) = 1 - P(X <= floor(line))
                prob_over = 1 - nbinom.cdf(int(l), n, p)
            else:
                # Fallback to normal approximation
                prob_over = 1 - stats.norm.cdf(l, loc=m, scale=np.sqrt(m))
            probs_over.append(prob_over)

        return np.array(probs_over)

    def summary(self):
        if self.result:
            print(f"\n   Dispersion parameter (alpha): {self.alpha:.4f}")
            print(f"   Log-likelihood: {self.result.llf:.2f}")
            print(f"   AIC: {self.result.aic:.2f}")
            print(f"   Converged: {self.result.mle_retvals.get('converged', 'N/A')}")


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def evaluate_negbinom(model: NegBinomModel, X_test, y_test,
                      model_name: str = "NegBinom",
                      lines: np.ndarray = None):
    preds = np.clip(model.predict(X_test), 0, 20)

    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    print(f"\n📊 {model_name} Evaluation")
    print(f"   MAE:  {mae:.4f} strikeouts")
    print(f"   RMSE: {rmse:.4f} strikeouts")

    if lines is None:
        lines = np.full(len(y_test), np.array(y_test).mean())

    # Standard OU accuracy (point estimate vs line)
    actual_over = (np.array(y_test) > lines)
    pred_over   = (preds > lines)
    ou_accuracy = (actual_over == pred_over).mean()
    print(f"   OU accuracy (point estimate): {ou_accuracy:.1%}")

    # Probabilistic OU accuracy — NegBinom's real advantage
    prob_over     = model.predict_ou_probability(X_test, lines)
    prob_pred_over = (prob_over > 0.5)
    prob_ou_acc   = (actual_over == prob_pred_over).mean()
    print(f"   OU accuracy (probabilistic):  {prob_ou_acc:.1%}  ← key NegBinom metric")

    # Calibration check — how accurate are the probability estimates?
    # Bin predictions by confidence and check actual win rate
    print(f"\n   Calibration (prob_over buckets):")
    for lo, hi in [(0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.0)]:
        mask = (prob_over >= lo) & (prob_over < hi)
        if mask.sum() > 20:
            actual_rate = actual_over[mask].mean()
            pred_rate   = prob_over[mask].mean()
            print(f"     P(over)={lo:.1f}-{hi:.1f}: "
                  f"n={mask.sum():4d}  predicted={pred_rate:.3f}  actual={actual_rate:.3f}  "
                  f"{'✓' if abs(actual_rate - pred_rate) < 0.05 else '⚠'}")

    model.summary()

    return {
        'mae': mae, 'rmse': rmse,
        'ou_accuracy': ou_accuracy,
        'prob_ou_accuracy': prob_ou_acc,
    }


# ------------------------------------------------------------------
# Cross validation
# ------------------------------------------------------------------

def cross_validate_negbinom(df: pd.DataFrame, features: list,
                             n_splits: int = 5) -> dict:
    print(f"\n🔄 NegBinom Time-Series Cross Validation ({n_splits} splits)...")

    X = df[features].values
    y = df[TARGET].values

    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes, ou_accs, prob_ou_accs = [], [], []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        lines = np.full(len(y_test), y_train.mean())

        model = NegBinomModel()
        try:
            model.fit(X_train, y_train, feature_names=features)
        except Exception as e:
            print(f"   Fold {fold+1}: convergence issue — {e}")
            continue

        preds     = np.clip(model.predict(X_test), 0, 20)
        mae       = mean_absolute_error(y_test, preds)
        ou_acc    = ((y_test > lines) == (preds > lines)).mean()
        prob_over = model.predict_ou_probability(X_test, lines)
        prob_ou   = ((y_test > lines) == (prob_over > 0.5)).mean()

        maes.append(mae)
        ou_accs.append(ou_acc)
        prob_ou_accs.append(prob_ou)

        print(f"   Fold {fold+1}: MAE={mae:.4f}  "
              f"OU={ou_acc:.1%}  ProbOU={prob_ou:.1%}  "
              f"alpha={model.alpha:.4f}")

    print(f"\n   NegBinom avg MAE={np.mean(maes):.4f} ± {np.std(maes):.4f}")
    print(f"            avg OU={np.mean(ou_accs):.1%}  "
          f"avg ProbOU={np.mean(prob_ou_accs):.1%}")

    return {
        'mae': np.mean(maes),
        'ou_accuracy': np.mean(ou_accs),
        'prob_ou_accuracy': np.mean(prob_ou_accs),
    }


# ------------------------------------------------------------------
# Save / load
# ------------------------------------------------------------------

def save_model(model: NegBinomModel, features: list,
               metadata: dict = None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    path = os.path.join(MODELS_DIR, f"k_prop_negbinom_{timestamp}.pkl")
    payload = {
        'model':    model,
        'features': features,
        'metadata': metadata or {},
        'saved_at': timestamp,
    }
    with open(path, 'wb') as f:
        pickle.dump(payload, f)
    print(f"\n💾 Model saved: {path}")
    return path


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cv-only', action='store_true')
    parser.add_argument('--compare', action='store_true',
                        help='Also train GBM and compare side by side')
    parser.add_argument('--db', type=str, default=None)
    args = parser.parse_args()

    db_path = args.db or DB_PATH
    df, features = build_training_data(db_path)

    # Cross-validate
    cv_results = cross_validate_negbinom(df, features)

    if args.cv_only:
        return

    # Train/test split — 2025 holdout
    train_df = df[df['season'] < 2025]
    test_df  = df[df['season'] == 2025]

    X_train = train_df[features].values
    y_train = train_df[TARGET].values
    X_test  = test_df[features].values
    y_test  = test_df[TARGET].values

    print(f"\n🚀 Training NegBinom on 2021-2024...")
    print(f"   Train: {len(train_df):,} | Test (2025): {len(test_df):,}")

    nb_model = NegBinomModel()
    nb_model.fit(X_train, y_train, feature_names=features)

    nb_results = evaluate_negbinom(nb_model, X_test, y_test,
                                   "NegBinom (2025 holdout)")

    if args.compare:
        print(f"\n🚀 Training GBM for comparison...")
        from k_prop_model import train_gbm, evaluate_model
        gbm = train_gbm(X_train, y_train)
        gbm_results = evaluate_model(gbm, X_test, y_test,
                                     "GBM (2025 holdout)")

        print("\n" + "=" * 50)
        print("📊 HEAD-TO-HEAD COMPARISON (2025 holdout)")
        print("=" * 50)
        print(f"{'Metric':<25} {'GBM':>10} {'NegBinom':>10}")
        print("-" * 50)
        print(f"{'MAE':<25} {gbm_results['mae']:>10.4f} {nb_results['mae']:>10.4f}")
        print(f"{'RMSE':<25} {gbm_results['rmse']:>10.4f} {nb_results['rmse']:>10.4f}")
        print(f"{'OU Accuracy':<25} {gbm_results['ou_accuracy']:>10.1%} "
              f"{nb_results['ou_accuracy']:>10.1%}")
        print(f"{'Prob OU Accuracy':<25} {'N/A':>10} "
              f"{nb_results['prob_ou_accuracy']:>10.1%}")
        print("=" * 50)
        print("\n  NegBinom advantage: probabilistic OU uses full")
        print("  distribution, not just point estimate vs line.")

    save_model(nb_model, features, {'cv_results': cv_results})
    print("\n✅ Done.")


if __name__ == "__main__":
    main()
