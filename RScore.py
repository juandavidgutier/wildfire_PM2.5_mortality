# importing required libraries
import os, warnings, random
import dowhy
import econml
from dowhy import CausalModel
import pandas as pd
import numpy as np
import econml
from econml.dml import DML
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
import numpy as np, scipy.stats as st
import scipy.stats as stats
import numpy as np, scipy.stats as st
from sklearn.linear_model import LassoCV
from econml.dml import CausalForestDML
from econml.orf import DMLOrthoForest
from itertools import product
from econml.dml import SparseLinearDML
from econml.dml import LinearDML
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from econml.score import RScorer
from sklearn.model_selection import train_test_split
from econml.dml import CausalForestDML
from joblib import Parallel, delayed
import warnings
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from econml.orf import DROrthoForest, DMLOrthoForest
from econml.utilities import WeightedModelWrapper
from econml.dml import KernelDML
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


np.int = np.int32
np.float = np.float64
np.bool = np.bool_



# Set seeds to make the results more reproducible
def seed_everything(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

seed = 123
seed_everything(seed)
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# import data
data = pd.read_csv("D:/top50_data.csv", encoding='latin-1')

# 1. Label Encoding DANE
le = LabelEncoder()
data['DANE_labeled'] = le.fit_transform(data['DANE'])
scaler = MinMaxScaler()
data['DANE_normalized'] = scaler.fit_transform(
    data[['DANE_labeled']]
)

# 2. Label Encoding DANE_year
le_year = LabelEncoder()
data['DANE_year_labeled'] = le_year.fit_transform(data['DANEYear'])
scaler_year = MinMaxScaler()
data['DANE_year_normalized'] = scaler_year.fit_transform(
    data[['DANE_year_labeled']]
)

# Quick check of variables of interest
variables_interes = ['TLML', 'pop_density']
for variable in variables_interes:
    if variable in data.columns:
        print(f"{variable}: mean={data[variable].mean():.4f}, median={data[variable].median():.4f}, sd={data[variable].std():.4f}")

# Binarize aerosol/pop columns based on medians
columns = data.columns[8:17]
medians = data[columns].median()
data[columns] = data[columns].apply(lambda x: (x > medians[x.name]).astype(int))

# Transform DUSMASS25
data['DUSMASS25'] = data['DUSMASS25'] / 10.0

# Split by wildfire group
data_fire = data[data['wildfire'] == 1].dropna().copy()
data_nofire = data[data['wildfire'] == 0].dropna().copy()

# Ensure outcomes are binary (0/1)
#outcomes_all = ['excess_all_1', 'excess_all_10', 'excess_all_20', 'excess_all_30']
outcomes_all = ['excess_all_1', 'excess_all_2', 'excess_all_3', 'excess_all_4', 'excess_all_5', 'excess_all_6', 'excess_all_7', 'excess_all_8']
for out in outcomes_all:
    if out in data.columns:
        data[out] = (data[out] > 0).astype(int)
        data_fire[out] = (data_fire[out] > 0).astype(int)
        data_nofire[out] = (data_nofire[out] > 0).astype(int)

from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

class ProbClassifierWrapper(BaseEstimator):
    """
    Wraps a sklearn classifier so that predict(X) returns
    the probability of class 1 (p=Pr(Y=1|X)), which is what we want
    for residualization when Y is binary.
    Optionally calibrates with CalibratedClassifierCV.
    """
    def __init__(self, base_clf=None, calibrate=True, random_state=123):
        self.base_clf = base_clf if base_clf is not None else RandomForestClassifier(
            n_estimators=200, n_jobs=1, random_state=random_state, class_weight='balanced'
        )
        self.calibrate = calibrate
        self.random_state = random_state
        self._is_fitted = False

    def fit(self, X, y, **kwargs):
        if self.calibrate:
            self.model_ = CalibratedClassifierCV(estimator=clone(self.base_clf), cv=3)
            self.model_.fit(X, y.ravel())
        else:
            self.model_ = clone(self.base_clf)
            self.model_.fit(X, y.ravel())
        self._is_fitted = True
        return self

    def predict(self, X):
        if not self._is_fitted:
            raise ValueError("ProbClassifierWrapper: fit must be called before predict.")
        return self.model_.predict_proba(X)[:, 1]

    def predict_proba(self, X):
        if not self._is_fitted:
            raise ValueError("ProbClassifierWrapper: fit must be called before predict_proba.")
        return self.model_.predict_proba(X)

def prepare_data_vecs(df, outcome_var):
    """Returns y, t, W, X in correct shapes (y,t 1-D; W,X 2-D)."""
    y = df[outcome_var].astype(int).values.ravel()
    t = df['DUSMASS25'].astype(float).values.ravel()
    W = df[['BCSMASS','DMSSMASS','DUSMASS','OCSMASS','SO2SMASS','SO4SMASS','pop_density','TLML']].values
    X = df[['TLML', 'DANE_normalized', 'DANE_year_normalized']].values
    return y, t, W, X

def nuisance_diagnostics(y_tr, t_tr, W_tr, X_tr, y_val, t_val, W_val, X_val, random_state=123):
    """
    Fits simple nuisance models (E[Y|X,W] and E[T|X,W]) on training and predicts on validation.
    Returns predictions and residuals for variance diagnostics.
    """
    Z_tr = np.hstack([X_tr, W_tr])
    Z_val = np.hstack([X_val, W_val])

    # model_y: classifier wrapper -> probabilities
    my = ProbClassifierWrapper(RandomForestClassifier(n_estimators=200, n_jobs=1, class_weight='balanced', random_state=random_state),
                               calibrate=True, random_state=random_state)
    my.fit(Z_tr, y_tr)
    y_pred_val = my.predict(Z_val)

    # model_t: regressor
    mt = RandomForestRegressor(n_estimators=200, n_jobs=1, random_state=random_state)
    mt.fit(Z_tr, t_tr)
    t_pred_val = mt.predict(Z_val)

    y_res = y_val - y_pred_val
    t_res = t_val - t_pred_val

    diagnostics = {
        'y_pred_val_mean': y_pred_val.mean(),
        'y_res_var': np.var(y_res),
        't_pred_val_mean': t_pred_val.mean(),
        't_res_var': np.var(t_res),
        'y_t_res_corr': np.corrcoef(y_res, t_res)[0,1]
    }
    return diagnostics, y_pred_val, t_pred_val, y_res, t_res

def make_causalforest_models(random_state=123):
    models = []
    for (n_est, depth, min_leaf) in [
        (3400, 35, 80),
        (3600, 33, 100),
        (3700, 31, 90),
        (3500, 39, 68),
        (3400, 30, 70),
        (3600, 32, 60)
    ]:
        name = f"forest_n{n_est}_d{depth}_leaf{min_leaf}"
        model_y_inst = ProbClassifierWrapper(
            RandomForestClassifier(n_estimators=200, n_jobs=1, class_weight='balanced', random_state=random_state),
            calibrate=True, random_state=random_state
        )
        model_t_inst = RandomForestRegressor(n_estimators=200, n_jobs=1, random_state=random_state)

        cf = CausalForestDML(
            model_y=model_y_inst,
            model_t=model_t_inst,
            discrete_outcome=True,
            discrete_treatment=False,
            n_estimators=n_est,
            max_depth=depth,
            min_samples_leaf=min_leaf,
            cv=5,
            random_state=random_state
        )
        models.append((name, cf))
    return models

from sklearn.metrics import mean_squared_error
import numpy as np

def fit_and_evaluate_models_binary(data_df, outcome_var, models_to_try, random_state=123):
    print(f"\n{'='*80}")
    print(f"{'='*80}")
    print(f"--- Processing outcome: {outcome_var} ---")
    print(f"{'='*80}")

    y, t, W, X = prepare_data_vecs(data_df, outcome_var)
    n = len(y)
    positives = y.sum()
    prevalence = positives / n if n>0 else np.nan
    print(f"n={n}, positives={positives}, prevalence={prevalence:.4f}, Var(T)={np.var(t):.6f}")

    # Stratified split (if possible)
    if positives >= 2 and (n - positives) >= 2:
        X_tr, X_val, t_tr, t_val, y_tr, y_val, W_tr, W_val = train_test_split(
            X, t, y, W, test_size=0.4, random_state=random_state, stratify=y
        )
    else:
        print("WARNING: Stratification not possible due to too few cases -> random split")
        X_tr, X_val, t_tr, t_val, y_tr, y_val, W_tr, W_val = train_test_split(
            X, t, y, W, test_size=0.4, random_state=random_state
        )

    # Nuisance diagnostics
    diag, y_pred_val, t_pred_val, y_res, t_res = nuisance_diagnostics(
        y_tr, t_tr, W_tr, X_tr, y_val, t_val, W_val, X_val, random_state=random_state
    )
    print("\nNuisance diagnostics (estimated on validation):")
    for k, v in diag.items():
        print(f"  {k}: {v:.6f}")

    # Baseline constant
    base_pred = y_val.mean()
    base_mse = mean_squared_error(y_val, np.full_like(y_val, base_pred, dtype=float))
    print(f"\nBaseline prob=mean(y_val)={base_pred:.6f}, base MSE={base_mse:.6f}")

    # Fit models in parallel
    print("\nFitting CausalForestDML models in parallel...")
    def fit_model(name, model):
        try:
            model.fit(y_tr, t_tr, X=X_tr, W=W_tr)
            return (name, model, None)
        except Exception as e:
            return (name, None, e)

    results = Parallel(n_jobs=-1, verbose=5, backend="threading")(
        delayed(fit_model)(name, mdl) for name, mdl in models_to_try
    )

    fitted_models = []
    for name, mdl, err in results:
        if err is not None:
            print(f"ERROR fitting {name}: {err}")
        else:
            fitted_models.append((name, mdl))

    # Manual Evaluation
    print("\nPerforming manual evaluation on validation set...")
    eval_scores = []
    for name, mdl in fitted_models:
        try:
            cate_pred_val = mdl.effect(X_val, T0=0, T1=1).ravel()
            correlation = np.corrcoef(cate_pred_val, y_res * t_res)[0, 1]
            eval_scores.append((name, correlation))
            print(f"  {name} -> Manual Eval Score (Correlation) = {correlation:.6f}")
        except Exception as e:
            print(f"Error during manual evaluation for {name}: {e}")
            eval_scores.append((name, np.nan))

    # Select best model
    valid_scores = [(n, s) for n, s in eval_scores if not (s is None or (isinstance(s, float) and np.isnan(s)))]
    if len(valid_scores) == 0:
        print("\nNo valid evaluation scores: check previous errors.")
        best = (None, None)
    else:
        best = max(valid_scores, key=lambda x: x[1])
        print(f"\n{'*'*60}")
        print(f"Best model (by Manual Eval Score) = {best[0]}")
        print(f"Score = {best[1]:.6f}")
        print(f"{'*'*60}")

    return {
        'outcome': outcome_var,
        'fitted_models': fitted_models,
        'eval_scores': eval_scores,
        'best': best,
        'nuisance_diag': diag,
        'base_pred': base_pred,
        'base_mse': base_mse,
        'n': n,
        'prevalence': prevalence
    }

# ============================================================================
# EVALUATION OF ALL OUTCOMES
# ============================================================================

models_to_try = make_causalforest_models(random_state=seed)

# Dictionary to store all results
all_results = {
    'fire': {},
    'nofire': {}
}

# EVALUATE FIRE GROUP FOR ALL OUTCOMES
print("\n" + "="*80)
print("="*80)
print("=== GROUP: FIRE - EVALUATION OF ALL OUTCOMES ===")
print("="*80)
print("="*80)

for outcome in outcomes_all:
    print(f"\n>>> Evaluating {outcome} in FIRE group <<<")
    res = fit_and_evaluate_models_binary(data_fire, outcome, models_to_try, random_state=seed)
    all_results['fire'][outcome] = res

# EVALUATE NOFIRE GROUP FOR ALL OUTCOMES
print("\n" + "="*80)
print("="*80)
print("=== GROUP: NOFIRE - EVALUATION OF ALL OUTCOMES ===")
print("="*80)
print("="*80)

for outcome in outcomes_all:
    print(f"\n>>> Evaluating {outcome} in NOFIRE group <<<")
    res = fit_and_evaluate_models_binary(data_nofire, outcome, models_to_try, random_state=seed)
    all_results['nofire'][outcome] = res

# ============================================================================
# CONSOLIDATED SUMMARY OF RESULTS
# ============================================================================

print("\n" + "="*80)
print("="*80)
print("CONSOLIDATED SUMMARY OF ALL ANALYSES")
print("="*80)
print("="*80)

for group in ['fire', 'nofire']:
    print(f"\n{'#'*80}")
    print(f"GROUP: {group.upper()}")
    print(f"{'#'*80}")

    for outcome in outcomes_all:
        res = all_results[group][outcome]
        print(f"\n{'-'*60}")
        print(f"Outcome: {outcome}")
        print(f"{'-'*60}")
        print(f"N: {res['n']}, Prevalence: {res['prevalence']:.4f}")
        print(f"Baseline MSE: {res['base_mse']:.6f}")
        print(f"\nNuisance Diagnostics:")
        for k, v in res['nuisance_diag'].items():
            print(f"  {k}: {v:.6f}")

        print(f"\nManual Eval Scores (Top 3):")
        sorted_scores = sorted(res['eval_scores'], key=lambda x: x[1] if not np.isnan(x[1]) else -np.inf, reverse=True)
        for i, (name, score) in enumerate(sorted_scores[:3], 1):
            print(f"  {i}. {name}: {score:.6f}")

        if res['best'][0] is not None:
            print(f"\n>>> BEST MODEL: {res['best'][0]} (Score: {res['best'][1]:.6f})")
        else:
            print(f"\n>>> Could not identify best model")

# ============================================================================
# COMPARATIVE TABLE OF BEST MODELS
# ============================================================================

print("\n" + "="*80)
print("COMPARATIVE TABLE: BEST MODELS BY OUTCOME AND GROUP")
print("="*80)

print(f"\n{'Group':<10} {'Outcome':<18} {'Best Model':<35} {'Score':<10} {'Prevalence':<12}")
print("-"*95)

for group in ['fire', 'nofire']:
    for outcome in outcomes_all:
        res = all_results[group][outcome]
        best_name = res['best'][0] if res['best'][0] else "N/A"
        best_score = f"{res['best'][1]:.6f}" if res['best'][1] is not None else "N/A"
        prev = f"{res['prevalence']:.4f}"
        print(f"{group.upper():<10} {outcome:<18} {best_name:<35} {best_score:<10} {prev:<12}")

print("\n" + "="*80)
print("--- Script completed successfully ---")
print("="*80)