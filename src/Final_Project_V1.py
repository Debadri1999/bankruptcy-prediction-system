# ============================================================
# Bankruptcy - "Best-of-both-worlds" Script
#  - LGBM on engineered features
#  - XGBoost & CatBoost on raw features
#  - 5-fold OOF for all base models
#  - XGBoost meta-stacking (leakage-free)
#  - Weight-searched ensemble vs meta; picks best
#  - Output: bankruptcy_submission.csv
# ============================================================

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# -----------------------------
# 0. Feature engineering (for LGBM only)
# -----------------------------
def add_features(df):
    df = df.copy()

    # 1) Log-like transforms on selected attributes
    log_cols = [
        "Attr2","Attr4","Attr8","Attr9","Attr10",
        "Attr18","Attr19","Attr22","Attr23","Attr35",
        "Attr36","Attr39","Attr42","Attr48","Attr49",
        "Attr56","Attr58","Attr60","Attr61","Attr63","Attr64"
    ]
    for col in log_cols:
        if col in df.columns:
            x = df[col].astype(float)
            df[f"{col}_log1p"] = np.sign(x) * np.log1p(np.abs(x))

    # 2) Simple interactions
    if all(c in df.columns for c in ["Attr2", "Attr9"]):
        df["Attr2_Attr9"] = df["Attr2"] * df["Attr9"]

    if all(c in df.columns for c in ["Attr2", "Attr22"]):
        df["Attr2_Attr22"] = df["Attr2"] * df["Attr22"]

    if all(c in df.columns for c in ["Attr4", "Attr9"]):
        df["Attr4_Attr9"] = df["Attr4"] * df["Attr9"]

    # short-term liabilities / total liabilities
    if all(c in df.columns for c in ["Attr51", "Attr2"]):
        df["short_long_liab_ratio"] = df["Attr51"] / (df["Attr2"].replace(0, np.nan) + 1e-6)

    return df

# -----------------------------
# 1. Load data (EDIT PATHS IF NEEDED)
# -----------------------------
train_raw = pd.read_csv("C:/Users/DOUBLEDO_GAMING/OneDrive/Desktop/PURDUE FALL'25/MOD2 - Fall'25/Data_mining_57100/fall-2025-mgmt-571-final-project/bankruptcy_Train.csv")
test_raw  = pd.read_csv("C:/Users/DOUBLEDO_GAMING/OneDrive/Desktop/PURDUE FALL'25/MOD2 - Fall'25/Data_mining_57100/fall-2025-mgmt-571-final-project/bankruptcy_Test_X.csv")

# Example full paths:
# train_raw = pd.read_csv(r"C:\...\bankruptcy_Train.csv")
# test_raw  = pd.read_csv(r"C:\...\bankruptcy_Test_X.csv")

# Apply feature engineering ONLY for LGBM
train_lgb = add_features(train_raw)
test_lgb  = add_features(test_raw)

# Target
y = train_raw["class"]

# Raw features for XGB & CatBoost
X_raw = train_raw.drop(columns=["class"])
id_col = "ID" if "ID" in test_raw.columns else test_raw.columns[0]
test_ids = test_raw[id_col]
X_test_raw = test_raw.drop(columns=[id_col])

# Engineered features for LGBM
X_lgb = train_lgb.drop(columns=["class"])
X_test_lgb = test_lgb.drop(columns=[id_col])

print("Raw Train shape:", X_raw.shape)
print("Raw Test shape:", X_test_raw.shape)
print("LGBM Train shape (with features):", X_lgb.shape)
print("LGBM Test shape (with features):", X_test_lgb.shape)
print("Class balance:\n", y.value_counts(normalize=True))

# -----------------------------
# 2. Imputation (separate for raw vs eng)
# -----------------------------
imputer_raw = SimpleImputer(strategy="median")
imputer_lgb = SimpleImputer(strategy="median")

X_raw_np = imputer_raw.fit_transform(X_raw)
X_test_raw_np = imputer_raw.transform(X_test_raw)

X_lgb_np = imputer_lgb.fit_transform(X_lgb)
X_test_lgb_np = imputer_lgb.transform(X_test_lgb)

y_np = y.values

pos_weight = (y_np == 0).sum() / (y_np == 1).sum()
print("Positive class weight (majority/minority):", pos_weight)

# -----------------------------
# 3. Level-1 CV (Base models, v1 params)
# -----------------------------
n_splits_level1 = 5
skf1 = StratifiedKFold(n_splits=n_splits_level1, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(y_np))
oof_xgb = np.zeros(len(y_np))
oof_cb  = np.zeros(len(y_np))

test_pred_lgb = np.zeros(len(X_test_lgb_np))
test_pred_xgb = np.zeros(len(X_test_raw_np))
test_pred_cb  = np.zeros(len(X_test_raw_np))

# ---- v1 hyperparameters ----

# LightGBM (on engineered features)
lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.015,
    "num_leaves": 48,
    "max_depth": -1,
    "min_data_in_leaf": 60,
    "feature_fraction": 0.72,
    "bagging_fraction": 0.85,
    "bagging_freq": 2,
    "lambda_l1": 0.2,
    "lambda_l2": 0.4,
    "scale_pos_weight": float(pos_weight),
    "verbose": -1,
    "num_threads": -1,
}

# XGBoost (on raw features)
xgb_base_params = {
    "objective": "binary:logistic",
    "learning_rate": 0.015,
    "n_estimators": 1600,
    "max_depth": 4,
    "min_child_weight": 2,
    "gamma": 0.0,
    "subsample": 0.85,
    "colsample_bytree": 0.75,
    "reg_lambda": 1.3,
    "reg_alpha": 0.3,
    "scale_pos_weight": float(pos_weight),
    "n_jobs": -1,
    "random_state": 42,
}

# CatBoost (on raw features)
cb_params = {
    "iterations": 2500,
    "learning_rate": 0.02,
    "depth": 7,
    "l2_leaf_reg": 5.0,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "random_seed": 42,
    "verbose": False,
    "class_weights": [1.0, float(pos_weight)],
}

print("\n===== Level-1: Training base models with 5-fold CV (best-of-both-worlds) =====")
fold_idx = 1
for train_idx, valid_idx in skf1.split(X_raw_np, y_np):
    print(f"\n--- Fold {fold_idx}/{n_splits_level1} ---")

    # Split raw and engineered views using same indices
    X_tr_raw, X_val_raw = X_raw_np[train_idx], X_raw_np[valid_idx]
    X_tr_lgb, X_val_lgb = X_lgb_np[train_idx], X_lgb_np[valid_idx]
    y_tr, y_val = y_np[train_idx], y_np[valid_idx]

    # ----- LightGBM on engineered features -----
    print("LightGBM (eng feats)...")
    dtrain_lgb = lgb.Dataset(X_tr_lgb, label=y_tr)
    dvalid_lgb = lgb.Dataset(X_val_lgb, label=y_val, reference=dtrain_lgb)

    lgb_model = lgb.train(
        params=lgb_params,
        train_set=dtrain_lgb,
        valid_sets=[dtrain_lgb, dvalid_lgb],
        valid_names=["train", "valid"],
        num_boost_round=5000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=300),
            lgb.log_evaluation(period=200),
        ],
    )

    val_pred_lgb = lgb_model.predict(
        X_val_lgb, num_iteration=lgb_model.best_iteration
    )
    oof_lgb[valid_idx] = val_pred_lgb
    test_pred_lgb += lgb_model.predict(
        X_test_lgb_np, num_iteration=lgb_model.best_iteration
    ) / n_splits_level1

    print("  LGB fold AUC:", roc_auc_score(y_val, val_pred_lgb))

    # ----- XGBoost on raw features -----
    print("XGBoost (raw)...")
    xgb_model = xgb.XGBClassifier(**xgb_base_params)
    xgb_model.fit(X_tr_raw, y_tr)

    val_pred_xgb = xgb_model.predict_proba(X_val_raw)[:, 1]
    oof_xgb[valid_idx] = val_pred_xgb
    test_pred_xgb += xgb_model.predict_proba(X_test_raw_np)[:, 1] / n_splits_level1

    print("  XGB fold AUC:", roc_auc_score(y_val, val_pred_xgb))

    # ----- CatBoost on raw features -----
    print("CatBoost (raw)...")
    cb_model = CatBoostClassifier(**cb_params)
    cb_model.fit(X_tr_raw, y_tr)

    val_pred_cb = cb_model.predict_proba(X_val_raw)[:, 1]
    oof_cb[valid_idx] = val_pred_cb
    test_pred_cb += cb_model.predict_proba(X_test_raw_np)[:, 1] / n_splits_level1

    print("  CB fold AUC:", roc_auc_score(y_val, val_pred_cb))

    fold_idx += 1

# ---- Level-1 OOF performance ----
auc_lgb = roc_auc_score(y_np, oof_lgb)
auc_xgb = roc_auc_score(y_np, oof_xgb)
auc_cb  = roc_auc_score(y_np, oof_cb)

print("\n==== Level-1 Base Model OOF AUCs (best-of-both-worlds) ====")
print(f"LightGBM OOF AUC: {auc_lgb:.5f}")
print(f"XGBoost  OOF AUC: {auc_xgb:.5f}")
print(f"CatBoost OOF AUC: {auc_cb:.5f}")

oof_avg3 = (oof_lgb + oof_xgb + oof_cb) / 3.0
auc_avg3 = roc_auc_score(y_np, oof_avg3)
print(f"Simple avg (LGB+XGB+CB) OOF AUC: {auc_avg3:.5f}")

# -----------------------------
# 4. Weight search on base OOF
# -----------------------------
print("\nSearching best ensemble weights (LGB, XGB, CB)...")
best_auc = 0.0
best_w = None

for w1 in np.linspace(0.1, 0.7, 13):       # LGB weight
    for w2 in np.linspace(0.05, 0.6, 12):  # XGB weight
        w3 = 1.0 - w1 - w2                 # CB weight
        if w3 <= 0 or w3 >= 0.8:
            continue
        blend = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cb
        auc = roc_auc_score(y_np, blend)
        if auc > best_auc:
            best_auc = auc
            best_w = (w1, w2, w3)

print("Best ensemble weights (LGB, XGB, CB):", best_w)
print("Best weighted ensemble OOF AUC:", round(best_auc, 5))

w_lgb, w_xgb, w_cb = best_w
oof_weighted = w_lgb * oof_lgb + w_xgb * oof_xgb + w_cb * oof_cb
test_pred_weighted = (
    w_lgb * test_pred_lgb +
    w_xgb * test_pred_xgb +
    w_cb  * test_pred_cb
)

# -----------------------------
# 5. Level-2 Meta stacking (XGBoost, leak-free)
# -----------------------------
print("\n===== Level-2: Leakage-free meta stacking (XGBoost) =====")

stack_X = np.vstack([oof_lgb, oof_xgb, oof_cb]).T
stack_test = np.vstack([test_pred_lgb, test_pred_xgb, test_pred_cb]).T

n_splits_level2 = 5
skf2 = StratifiedKFold(n_splits=n_splits_level2, shuffle=True, random_state=2025)

oof_meta = np.zeros(len(y_np))
test_pred_meta = np.zeros(len(X_test_raw_np))

meta_params = {
    "objective": "binary:logistic",
    "learning_rate": 0.05,
    "n_estimators": 400,
    "max_depth": 3,
    "min_child_weight": 1,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "n_jobs": -1,
    "random_state": 2025,
}

fold_idx = 1
for train_idx, valid_idx in skf2.split(stack_X, y_np):
    print(f"\n--- Meta Fold {fold_idx}/{n_splits_level2} ---")
    X_meta_tr, X_meta_val = stack_X[train_idx], stack_X[valid_idx]
    y_meta_tr, y_meta_val = y_np[train_idx], y_np[valid_idx]

    meta_model = xgb.XGBClassifier(**meta_params)
    meta_model.fit(X_meta_tr, y_meta_tr)

    val_pred_meta = meta_model.predict_proba(X_meta_val)[:, 1]
    oof_meta[valid_idx] = val_pred_meta

    fold_auc_meta = roc_auc_score(y_meta_val, val_pred_meta)
    print("  Meta fold AUC:", round(fold_auc_meta, 5))

    test_pred_meta += meta_model.predict_proba(stack_test)[:, 1] / n_splits_level2
    fold_idx += 1

auc_meta = roc_auc_score(y_np, oof_meta)
print(f"\nMeta-model OOF AUC (best-of-both-worlds): {auc_meta:.5f}")

# -----------------------------
# 6. Final choice & submission
# -----------------------------
print("\nSummary (best-of-both-worlds):")
print(f"- Best weighted base ensemble OOF AUC: {best_auc:.5f}")
print(f"- Meta XGB stack OOF AUC:             {auc_meta:.5f}")

if auc_meta >= best_auc:
    print("Using META model predictions for final submission.")
    final_test_pred = test_pred_meta
else:
    print("Using WEIGHTED BASE ENSEMBLE predictions for final submission.")
    final_test_pred = test_pred_weighted

submission = pd.DataFrame({
    "ID": test_ids,
    "class": final_test_pred
})

submission.to_csv("C:/Users/DOUBLEDO_GAMING/OneDrive/Desktop/PURDUE FALL'25/MOD2 - Fall'25/Data_mining_57100/fall-2025-mgmt-571-final-project/bankruptcy_submission.csv", index=False)
print("\nSaved bankruptcy_submission.csv")
print(submission.head())
