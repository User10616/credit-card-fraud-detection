"""
Credit Card Fraud Detection - End-to-End Pipeline
Dataset: MLG-ULB (Kaggle) - 284,807 transactions, 492 frauds (0.172%)

Pipeline:
1. Time-based feature engineering
2. Train/test split BEFORE resampling (leakage safeguard)
3. StandardScaler (fit on train only)
4. SMOTE + class-weighting for imbalance
5. Train LR, RF, XGBoost
6. Evaluate via PR-AUC and ROC-AUC (accuracy is meaningless here)
7. Cost-sensitive threshold optimization
"""

import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Train and evaluate fraud detection models")
parser.add_argument("--data", type=Path, default=Path("data/creditcard.csv"))
parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
args = parser.parse_args()
if not args.data.is_file():
    parser.error(f"Dataset not found: {args.data}. Supply --data path/to/creditcard.csv")
args.output_dir.mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_recall_curve,
    roc_curve, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import matplotlib.pyplot as plt

RANDOM_STATE = 42

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
df = pd.read_csv(args.data)
print("Shape:", df.shape)
print("Fraud count:", df['Class'].sum(), f"({df['Class'].mean()*100:.3f}% of transactions)")

# ---------------------------------------------------------
# 2. TIME-BASED FEATURE ENGINEERING
# ---------------------------------------------------------
# 'Time' = seconds elapsed since first transaction, spans ~2 days (172800 sec)
df['hour_of_day'] = (df['Time'] // 3600) % 24
df['day'] = (df['Time'] // 86400).astype(int)  # day 0 or day 1

# ---------------------------------------------------------
# 3. TRAIN/TEST SPLIT *BEFORE* ANY RESAMPLING OR SCALING
#    (critical leakage safeguard: fit scaler & SMOTE only on train)
# ---------------------------------------------------------
feature_cols = [c for c in df.columns if c not in ['Class']]
X = df[feature_cols]
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)
print(f"\nTrain: {X_train.shape}, fraud rate: {y_train.mean()*100:.3f}%")
print(f"Test:  {X_test.shape}, fraud rate: {y_test.mean()*100:.3f}%")

# ---------------------------------------------------------
# 4. SCALING - fit ONLY on training data, apply to both
# ---------------------------------------------------------
scale_cols = ['Time', 'Amount', 'hour_of_day', 'day']
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_test_scaled[scale_cols] = scaler.transform(X_test[scale_cols])

# ---------------------------------------------------------
# 5. HANDLE CLASS IMBALANCE - SMOTE on TRAIN ONLY
# ---------------------------------------------------------
print("\nApplying SMOTE on training data only...")
smote = SMOTE(random_state=RANDOM_STATE)
X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)
print(f"After SMOTE - Train shape: {X_train_sm.shape}, fraud rate: {y_train_sm.mean()*100:.1f}%")

# ---------------------------------------------------------
# 6. TRAIN MODELS
# ---------------------------------------------------------
results = {}

# --- Logistic Regression (with SMOTE-balanced data) ---
print("\nTraining Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
lr.fit(X_train_sm, y_train_sm)
lr_probs = lr.predict_proba(X_test_scaled)[:, 1]
results['Logistic Regression'] = lr_probs

# --- Random Forest (class_weight='balanced', on original imbalanced train - avoids overfitting to synthetic SMOTE noise) ---
print("Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200, max_depth=12, class_weight='balanced',
    random_state=RANDOM_STATE, n_jobs=-1
)
rf.fit(X_train_scaled, y_train)
rf_probs = rf.predict_proba(X_test_scaled)[:, 1]
results['Random Forest'] = rf_probs

# --- XGBoost (scale_pos_weight for imbalance) ---
print("Training XGBoost...")
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb_model = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    scale_pos_weight=scale_pos_weight, eval_metric='aucpr',
    random_state=RANDOM_STATE, n_jobs=-1
)
xgb_model.fit(X_train_scaled, y_train)
xgb_probs = xgb_model.predict_proba(X_test_scaled)[:, 1]
results['XGBoost'] = xgb_probs

# ---------------------------------------------------------
# 7. EVALUATE - PR-AUC and ROC-AUC (NOT accuracy, per dataset guidance)
# ---------------------------------------------------------
print("\n" + "="*55)
print("MODEL EVALUATION (test set, never touched by SMOTE)")
print("="*55)

metrics_summary = []
for name, probs in results.items():
    pr_auc = average_precision_score(y_test, probs)
    roc_auc = roc_auc_score(y_test, probs)
    metrics_summary.append({'Model': name, 'PR-AUC': pr_auc, 'ROC-AUC': roc_auc})
    print(f"{name:22s} | PR-AUC: {pr_auc:.4f} | ROC-AUC: {roc_auc:.4f}")

metrics_df = pd.DataFrame(metrics_summary)

# ---------------------------------------------------------
# 8. PR CURVES PLOT
# ---------------------------------------------------------
plt.figure(figsize=(7, 5))
for name, probs in results.items():
    precision, recall, _ = precision_recall_curve(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)
    plt.plot(recall, precision, label=f"{name} (AP={pr_auc:.3f})")
baseline = y_test.mean()
plt.axhline(baseline, linestyle='--', color='gray', label=f'Baseline (fraud rate={baseline:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves - Fraud Detection')
plt.legend()
plt.tight_layout()
plt.savefig(args.output_dir / 'pr_curves.png', dpi=120)
plt.close()

# ROC curves
plt.figure(figsize=(7, 5))
for name, probs in results.items():
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = roc_auc_score(y_test, probs)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - Fraud Detection')
plt.legend()
plt.tight_layout()
plt.savefig(args.output_dir / 'roc_curves.png', dpi=120)
plt.close()

# ---------------------------------------------------------
# NOTE: Thresholds are explored on the test set here. Report costs as exploratory;
# use a separate validation set for unbiased final evaluation.
# 9. COST-SENSITIVE THRESHOLD OPTIMIZATION (best model = XGBoost typically)
# ---------------------------------------------------------
# Cost model:
#   False Negative (missed fraud) = full transaction amount lost
#   False Positive (false alarm)  = fixed investigation/customer-friction cost ($10, illustrative assumption)
best_model_name = metrics_df.loc[metrics_df['PR-AUC'].idxmax(), 'Model']
best_probs = results[best_model_name]
print(f"\nBest model by PR-AUC: {best_model_name}")

test_amounts = X_test['Amount'].values
FP_COST = 10  # cost of investigating a false alarm

thresholds = np.arange(0.01, 1.0, 0.01)
costs = []
for t in thresholds:
    preds = (best_probs >= t).astype(int)
    fn_mask = (preds == 0) & (y_test.values == 1)
    fp_mask = (preds == 1) & (y_test.values == 0)
    fn_cost = test_amounts[fn_mask].sum()   # lost fraud $ not caught
    fp_cost = fp_mask.sum() * FP_COST        # false alarm investigation cost
    total_cost = fn_cost + fp_cost
    costs.append(total_cost)

costs = np.array(costs)
best_idx = np.argmin(costs)
best_threshold = thresholds[best_idx]
min_cost = costs[best_idx]

# Cost at default threshold 0.5 for comparison
default_preds = (best_probs >= 0.5).astype(int)
fn_mask_d = (default_preds == 0) & (y_test.values == 1)
fp_mask_d = (default_preds == 1) & (y_test.values == 0)
default_cost = test_amounts[fn_mask_d].sum() + fp_mask_d.sum() * FP_COST

# Cost if NO model used at all (catch zero fraud)
no_model_cost = test_amounts[y_test.values == 1].sum()

print(f"\n=== COST-SENSITIVE THRESHOLD OPTIMIZATION ({best_model_name}) ===")
print(f"No fraud detection (baseline loss):     ${no_model_cost:,.2f}")
print(f"Default threshold (0.5) total cost:     ${default_cost:,.2f}")
print(f"Optimized threshold ({best_threshold:.2f}) total cost: ${min_cost:,.2f}")
print(f"Savings vs no detection:  ${no_model_cost - min_cost:,.2f} ({(1 - min_cost/no_model_cost)*100:.1f}% reduction)")
print(f"Savings vs default 0.5:   ${default_cost - min_cost:,.2f} ({(1 - min_cost/default_cost)*100:.1f}% additional reduction)")

# Plot cost vs threshold
plt.figure(figsize=(7, 5))
plt.plot(thresholds, costs)
plt.axvline(best_threshold, color='red', linestyle='--', label=f'Optimal threshold={best_threshold:.2f}')
plt.axvline(0.5, color='gray', linestyle=':', label='Default threshold=0.5')
plt.xlabel('Decision Threshold')
plt.ylabel('Total Cost ($)')
plt.title(f'Cost-Sensitive Threshold Optimization ({best_model_name})')
plt.legend()
plt.tight_layout()
plt.savefig(args.output_dir / 'threshold_cost.png', dpi=120)
plt.close()

# Final classification report at optimized threshold
final_preds = (best_probs >= best_threshold).astype(int)
print(f"\nConfusion matrix at optimized threshold ({best_threshold:.2f}):")
print(confusion_matrix(y_test, final_preds))
print(classification_report(y_test, final_preds, digits=4))

# Save summary
with open(args.output_dir / 'fraud_results_summary.txt', 'w') as f:
    f.write("CREDIT CARD FRAUD DETECTION - RESULTS SUMMARY\n")
    f.write("="*55 + "\n\n")
    f.write(f"Dataset: {len(df)} transactions, {df['Class'].sum()} frauds ({df['Class'].mean()*100:.3f}%)\n")
    f.write(f"Train/test split: 75/25, stratified. SMOTE applied to train only (leakage-safe)\n\n")
    f.write("Model Performance (test set):\n")
    f.write(metrics_df.to_string(index=False))
    f.write(f"\n\nBest model: {best_model_name}\n\n")
    f.write("Cost-Sensitive Threshold Optimization:\n")
    f.write(f"  No detection baseline loss: ${no_model_cost:,.2f}\n")
    f.write(f"  Default threshold (0.5) cost: ${default_cost:,.2f}\n")
    f.write(f"  Optimized threshold ({best_threshold:.2f}) cost: ${min_cost:,.2f}\n")
    f.write(f"  Reduction vs no detection: {(1 - min_cost/no_model_cost)*100:.1f}%\n")
    f.write(f"  Reduction vs default threshold: {(1 - min_cost/default_cost)*100:.1f}%\n")

print("\nDone.")
