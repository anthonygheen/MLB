"""
models/k_prop_model.py
-----------------------
Trains and evaluates a strikeout prop model using GBM regression.

Target: actual strikeouts recorded by the starting pitcher in a game.
Evaluation: MAE, RMSE, and over/under accuracy vs a hypothetical line.

Usage:
    python k_prop_model.py                   # train and save model
    python k_prop_model.py --evaluate        # load saved model and evaluate
"""

import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from scipy.optimize import minimize_scalar
from scipy.special import gammaln
from dotenv import load_dotenv

load_dotenv()

# Adjust path so we can import from sibling directories
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.pitcher_features import build_pitcher_feature_matrix
from features.lineup_features import build_lineup_feature_matrix

DB_PATH    = os.getenv("DB_PATH", "../data/mlb_modeling.db")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "saved")
os.makedirs(MODELS_DIR, exist_ok=True)


# ------------------------------------------------------------------
# Feature columns used for training
# ------------------------------------------------------------------

# Core rolling performance features
ROLLING_FEATURES = [
    'k_rate_last3', 'k_rate_last5', 'k_rate_last10',
    'swstr_pct_last3', 'swstr_pct_last5',
    'zone_pct_last3', 'zone_pct_last5',
    'bb_rate_last3', 'bb_rate_last5',
    'strikeouts_last3', 'strikeouts_last5', 'strikeouts_last10',
    'pitch_count_last3', 'pitch_count_last5',
]

# Stuff grade features (z-scores vs league)
STUFF_FEATURES = [
    'ff_avg_velo', 'ff_velo_z', 'ff_spin_z', 'ff_whiff_z',
    'sl_velo_z',   'sl_spin_z',  'sl_whiff_z', 'sl_ivb_z',
    'ch_velo_z',   'ch_whiff_z', 'ch_ivb_z',
    'cu_spin_z',   'cu_whiff_z',
    'si_velo_z',   'si_whiff_z',
]

# Pitch mix features
MIX_FEATURES = [
    'mix_ff_last3', 'mix_sl_last3', 'mix_ch_last3',
    'mix_cu_last3', 'mix_si_last3',
]

# Lineup features
LINEUP_FEATURES = [
    'lineup_k_rate_weighted',
    'lineup_k_rate_raw',
]

# Context features
CONTEXT_FEATURES = [
    'park_k_factor',
    'pitcher_hand_enc',
    'days_rest',
    'month',
]

ALL_FEATURES = (ROLLING_FEATURES + STUFF_FEATURES +
                MIX_FEATURES + LINEUP_FEATURES + CONTEXT_FEATURES)

TARGET = 'strikeouts'


# ------------------------------------------------------------------
# Data assembly
# ------------------------------------------------------------------

def build_training_data(db_path: str = None) -> pd.DataFrame:
    """
    Assembles pitcher features + lineup features into one training DataFrame.
    Drops rows with insufficient data (early career starts, missing stuff grades).
    """
    path = db_path or DB_PATH

    print("=" * 60)
    print("BUILDING TRAINING DATA")
    print("=" * 60)

    pitcher_df = build_pitcher_feature_matrix(path)
    lineup_df  = build_lineup_feature_matrix(path)

    print("\n🔗 Joining pitcher and lineup features...")
    df = pitcher_df.merge(lineup_df, on=['pitcher_id', 'game_id'], how='left')

    # Drop rows missing the target
    df = df.dropna(subset=[TARGET])

    # Require at least 3 prior starts for rolling features to be meaningful
    df = df[df['k_rate_last3'].notna()]

    # Fill missing stuff grades with 0 (league average z-score)
    for col in STUFF_FEATURES + MIX_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Fill missing lineup features with league average K rate (~0.22)
    df['lineup_k_rate_weighted'] = df.get('lineup_k_rate_weighted', pd.Series()).fillna(0.22)
    df['lineup_k_rate_raw']      = df.get('lineup_k_rate_raw', pd.Series()).fillna(0.22)

    # Only keep features that actually exist in the dataframe
    available_features = [f for f in ALL_FEATURES if f in df.columns]
    missing = [f for f in ALL_FEATURES if f not in df.columns]
    if missing:
        print(f"⚠️  Missing features (will skip): {missing}")

    df = df.sort_values('game_date').reset_index(drop=True)

    print(f"\n✅ Training data ready:")
    print(f"   Rows: {len(df):,}")
    print(f"   Features: {len(available_features)}")
    print(f"   Date range: {df['game_date'].min()} → {df['game_date'].max()}")
    print(f"   Target (strikeouts) mean: {df[TARGET].mean():.2f}")
    print(f"   Target (strikeouts) std:  {df[TARGET].std():.2f}")

    return df, available_features


# ------------------------------------------------------------------
# Model training
# ------------------------------------------------------------------

def train_gbm(X_train, y_train) -> GradientBoostingRegressor:
    """Train Gradient Boosting regressor."""
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=20,
        subsample=0.8,
        random_state=42,
        validation_fraction=0.1,
        n_iter_no_change=20,
        tol=1e-4,
    )
    model.fit(X_train, y_train)
    return model


def train_poisson(X_train, y_train) -> Pipeline:
    """Train Poisson GLM (better calibrated for count targets)."""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', PoissonRegressor(alpha=0.1, max_iter=300))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


# ------------------------------------------------------------------
# Hyperparameter tuning
# ------------------------------------------------------------------

def tune_gbm(X_train, y_train) -> GradientBoostingRegressor:
    """
    Grid search over key GBM hyperparameters using time-series CV.
    Optimizes for MAE. Takes ~10-15 minutes on a full dataset.
    """
    print("\n🔍 Running grid search (this will take 10-15 minutes)...")

    param_grid = {
        'max_depth':        [3, 4, 5],
        'learning_rate':    [0.01, 0.05, 0.1],
        'min_samples_leaf': [10, 20, 30],
        'subsample':        [0.7, 0.8, 0.9],
    }

    base = GradientBoostingRegressor(
        n_estimators=300,
        random_state=42,
        validation_fraction=0.1,
        n_iter_no_change=20,
        tol=1e-4,
    )

    tscv = TimeSeriesSplit(n_splits=5)

    search = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,          # use all cores
        verbose=1,
        refit=True,         # refit best model on full training set
    )

    search.fit(X_train, y_train)

    print(f"\n✅ Best parameters:")
    for param, val in search.best_params_.items():
        print(f"   {param}: {val}")
    print(f"   Best CV MAE: {-search.best_score_:.4f}")

    # Show top 5 configurations
    results_df = pd.DataFrame(search.cv_results_)
    results_df['mean_mae'] = -results_df['mean_test_score']
    top5 = results_df.nsmallest(5, 'mean_mae')[
        ['mean_mae', 'std_test_score',
         'param_max_depth', 'param_learning_rate',
         'param_min_samples_leaf', 'param_subsample']
    ]
    print(f"\n📊 Top 5 configurations:")
    print(top5.to_string(index=False))

    return search.best_estimator_, search.best_params_


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def evaluate_model(model, X_test, y_test, model_name: str = "Model",
                   lines: pd.Series = None):
    """
    Evaluate regression accuracy and — if book lines are provided —
    simulate over/under accuracy.
    """
    preds = model.predict(X_test)
    preds = np.clip(preds, 0, 20)   # K totals can't be negative or absurd

    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    print(f"\n📊 {model_name} Evaluation")
    print(f"   MAE:  {mae:.3f} strikeouts")
    print(f"   RMSE: {rmse:.3f} strikeouts")

    # Over/under accuracy vs a hypothetical line
    # If no actual book lines, use rolling mean as proxy
    if lines is None:
        lines = pd.Series(y_test).rolling(50, min_periods=10).mean().fillna(
            y_test.mean()
        ).values

    actual_over  = (y_test.values > lines)
    pred_over    = (preds > lines)
    ou_accuracy  = (actual_over == pred_over).mean()
    print(f"   Over/Under accuracy vs line: {ou_accuracy:.1%}")

    # Edge distribution — how far our predictions deviate from the line
    edge = preds - lines
    print(f"   Mean predicted edge: {edge.mean():.2f}")
    print(f"   Edge std: {edge.std():.2f}")

    return {
        'mae': mae,
        'rmse': rmse,
        'ou_accuracy': ou_accuracy,
        'predictions': preds,
    }


def cross_validate_model(df: pd.DataFrame, features: list,
                         n_splits: int = 5) -> dict:
    """
    Time-series cross validation — always train on past, test on future.
    This is critical for sports models to avoid look-ahead bias.
    """
    print(f"\n🔄 Time-Series Cross Validation ({n_splits} splits)...")

    X = df[features].values
    y = df[TARGET].values

    tscv = TimeSeriesSplit(n_splits=n_splits)

    gbm_maes, poisson_maes = [], []
    gbm_ous, poisson_ous   = [], []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Use fold mean as a naive line proxy
        lines = np.full(len(y_test), y_train.mean())

        # GBM
        gbm = train_gbm(X_train, y_train)
        gbm_preds = np.clip(gbm.predict(X_test), 0, 20)
        gbm_mae   = mean_absolute_error(y_test, gbm_preds)
        gbm_ou    = ((y_test > lines) == (gbm_preds > lines)).mean()
        gbm_maes.append(gbm_mae)
        gbm_ous.append(gbm_ou)

        # Poisson
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_train)
        X_te_scaled = scaler.transform(X_test)
        poisson = PoissonRegressor(alpha=0.1, max_iter=300)
        poisson.fit(X_tr_scaled, y_train)
        poisson_preds = np.clip(poisson.predict(X_te_scaled), 0, 20)
        poisson_mae   = mean_absolute_error(y_test, poisson_preds)
        poisson_ou    = ((y_test > lines) == (poisson_preds > lines)).mean()
        poisson_maes.append(poisson_mae)
        poisson_ous.append(poisson_ou)

        print(f"   Fold {fold+1}: GBM MAE={gbm_mae:.3f} OU={gbm_ou:.1%} | "
              f"Poisson MAE={poisson_mae:.3f} OU={poisson_ou:.1%}")

    print(f"\n   GBM    avg MAE={np.mean(gbm_maes):.3f} ± {np.std(gbm_maes):.3f}  "
          f"avg OU={np.mean(gbm_ous):.1%}")
    print(f"   Poisson avg MAE={np.mean(poisson_maes):.3f} ± {np.std(poisson_maes):.3f}  "
          f"avg OU={np.mean(poisson_ous):.1%}")

    return {
        'gbm_mae':     np.mean(gbm_maes),
        'poisson_mae': np.mean(poisson_maes),
        'gbm_ou':      np.mean(gbm_ous),
        'poisson_ou':  np.mean(poisson_ous),
    }


# ------------------------------------------------------------------
# Feature importance
# ------------------------------------------------------------------

def print_feature_importance(model, features: list, top_n: int = 20):
    if hasattr(model, 'feature_importances_'):
        importance = pd.DataFrame({
            'feature':    features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\n🏆 Top {top_n} Features:")
        print(importance.head(top_n).to_string(index=False))
    else:
        print("(Feature importance not available for this model type)")


# ------------------------------------------------------------------
# NegBinom calibration layer
# ------------------------------------------------------------------

def fit_negbinom_dispersion(mu_pred: np.ndarray, y_actual: np.ndarray) -> float:
    """
    Fit NegBinom dispersion parameter r given GBM predicted means and actual counts.

    Uses MLE: find r that maximizes Σ log NegBinom(y_i | μ_i, r).
    r controls tail width — larger r = tighter distribution (→ Poisson as r → ∞).
    Saved with the model so predict_today.py can compute P(K > line) properly.
    """
    mu = np.asarray(mu_pred, dtype=float).clip(0.1, 20)
    y  = np.asarray(y_actual, dtype=float)

    def neg_log_likelihood(r):
        p  = r / (r + mu)
        ll = (gammaln(y + r) - gammaln(r) - gammaln(y + 1)
              + r * np.log(p) + y * np.log(1 - p + 1e-10))
        return -ll.sum()

    result = minimize_scalar(neg_log_likelihood, bounds=(0.5, 100), method='bounded')
    return float(result.x)


# ------------------------------------------------------------------
# Save / load
# ------------------------------------------------------------------

def save_model(model, features: list, model_name: str, metadata: dict = None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    path = os.path.join(MODELS_DIR, f"{model_name}_{timestamp}.pkl")

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


def load_model(path: str) -> tuple:
    with open(path, 'rb') as f:
        payload = pickle.load(f)
    return payload['model'], payload['features'], payload['metadata']


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluate', action='store_true',
                        help='Load latest saved model and evaluate')
    parser.add_argument('--cv-only', action='store_true',
                        help='Run cross-validation only, no final model save')
    parser.add_argument('--tune', action='store_true',
                        help='Run grid search before training final model')
    parser.add_argument('--db', type=str, default=None)
    args = parser.parse_args()

    db_path = args.db or DB_PATH

    # Build data
    df, features = build_training_data(db_path)

    if args.evaluate:
        # Load latest model and evaluate on most recent season
        model_files = sorted(os.listdir(MODELS_DIR))
        if not model_files:
            print("No saved models found. Run without --evaluate first.")
            return
        latest = os.path.join(MODELS_DIR, model_files[-1])
        model, features, meta = load_model(latest)
        print(f"Loaded: {latest}")

        test_df = df[df['season'] == df['season'].max()]
        X_test  = test_df[features].values
        y_test  = test_df[TARGET]
        evaluate_model(model, X_test, y_test, "Saved Model")
        return

    # Cross-validate first
    cv_results = cross_validate_model(df, features)

    if args.cv_only:
        return

    # Hold out last season for final evaluation
    train_df = df[df['season'] < df['season'].max()]
    test_df  = df[df['season'] == df['season'].max()]

    X_train = train_df[features].values
    y_train = train_df[TARGET].values
    X_test  = test_df[features].values
    y_test  = test_df[TARGET]

    print(f"\n🚀 Training final model on full dataset...")
    print(f"   Train: {len(train_df):,} starts | Test (last season): {len(test_df):,} starts")

    if args.tune:
        # Grid search on training set, then evaluate best model on holdout
        gbm, best_params = tune_gbm(X_train, y_train)
        print(f"\n📊 Evaluating tuned model on holdout season...")
    else:
        # Use default hyperparameters
        best_params = {}
        gbm = train_gbm(X_train, y_train)

    evaluate_model(gbm, X_test, y_test, "GBM (holdout last season)")
    print_feature_importance(gbm, features)

    # Fit NegBinom dispersion on holdout predictions
    print(f"\n📐 Fitting NegBinom dispersion on holdout season...")
    holdout_preds = np.clip(gbm.predict(X_test), 0.1, 20)
    negbinom_r = fit_negbinom_dispersion(holdout_preds, y_test.values)
    print(f"   r = {negbinom_r:.4f}  "
          f"(implied variance at mean {holdout_preds.mean():.1f} Ks: "
          f"{holdout_preds.mean() + holdout_preds.mean()**2 / negbinom_r:.2f})")

    # Save
    metadata = {
        'train_seasons': sorted(train_df['season'].unique().tolist()),
        'test_season':   int(test_df['season'].max()),
        'n_train':       len(train_df),
        'cv_results':    cv_results,
        'tuned':         args.tune,
        'best_params':   best_params,
        'negbinom_r':    negbinom_r,
    }
    save_model(gbm, features, 'k_prop_gbm', metadata)

    print("\n✅ Done. Next step: run evaluate/edge_finder.py to compare against book lines.")


if __name__ == "__main__":
    main()
