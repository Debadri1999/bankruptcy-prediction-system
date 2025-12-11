import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

# =========================================================
# Paths (EDIT IF NEEDED)
# =========================================================
TRAIN_PATH = r"C:/Users/DOUBLEDO_GAMING/OneDrive/Desktop/PURDUE FALL'25/MOD2 - Fall'25/Data_mining_57100/fall-2025-mgmt-571-final-project/bankruptcy_Train.csv"
TEST_PATH  = r"C:/Users/DOUBLEDO_GAMING/OneDrive/Desktop/PURDUE FALL'25/MOD2 - Fall'25/Data_mining_57100/fall-2025-mgmt-571-final-project/bankruptcy_Test_X.csv"

OUT_PATH   = r"C:/Users/DOUBLEDO_GAMING/OneDrive/Desktop/PURDUE FALL'25/MOD2 - Fall'25/Data_mining_57100/fall-2025-mgmt-571-final-project/submission_V28_MINIMAL_lgbm_plus_FINAL.csv"

# =========================================================
# 1. Load data
# =========================================================
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

y = train["class"].values
X = train.drop(columns=["class"])
X_test = test.drop(columns=["ID"])
test_ids = test["ID"].values

print("Train shape:", X.shape)
print("Test shape :", X_test.shape)

pos_rate = y.mean()
scale_pos = (1 - pos_rate) / pos_rate
print("Positive class rate:", pos_rate)
print("scale_pos_weight   :", scale_pos)

# =========================================================
# 2. Quantile clipping (for stability)
# =========================================================
def quantile_clip(train_df: pd.DataFrame, test_df: pd.DataFrame,
                  q_low=0.01, q_high=0.99) -> (pd.DataFrame, pd.DataFrame):
    train_df = train_df.copy()
    test_df = test_df.copy()
    for col in train_df.columns:
        lo = train_df[col].quantile(q_low)
        hi = train_df[col].quantile(q_high)
        train_df[col] = train_df[col].clip(lo, hi)
        test_df[col]  = test_df[col].clip(lo, hi)
    return train_df, test_df

X_clip, X_test_clip = quantile_clip(X, X_test, q_low=0.01, q_high=0.99)
print("After clipping - train:", X_clip.shape)
print("After clipping - test :", X_test_clip.shape)

# =========================================================
# 3A. Feature engineering for LGBM (heavy FE on clipped)
# =========================================================
def fe_light(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = df.columns.tolist()
    eps = 1e-6

    row_mean = df[cols].mean(axis=1)
    row_std  = df[cols].std(axis=1)
    row_max  = df[cols].max(axis=1)
    row_min  = df[cols].min(axis=1)

    df["row_mean"] = row_mean
    df["row_std"]  = row_std
    df["row_max"]  = row_max
    df["row_min"]  = row_min

    for c in cols:
        df[f"log1p_{c}"] = np.log1p(np.abs(df[c]))
        df[f"{c}_sq"]    = df[c] ** 2
        df[f"{c}_div_rowmean"] = df[c] / (row_mean + eps)

    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)
    return df

X_all_clip = pd.concat([X_clip, X_test_clip], axis=0).reset_index(drop=True)
X_all_fe = fe_light(X_all_clip)

X_lgb_train = X_all_fe.iloc[:len(X)].reset_index(drop=True).astype("float32").values
X_lgb_test  = X_all_fe.iloc[len(X):].reset_index(drop=True).astype("float32").values

print("LGBM FE train shape:", X_lgb_train.shape)
print("LGBM FE test  shape:", X_lgb_test.shape)

# =========================================================
# 3B. Enhanced raw view for XGB: (clip + log1p + RobustScaler)
# =========================================================
X_raw_enh = X_clip.copy()
X_test_raw_enh = X_test_clip.copy()

for c in X_raw_enh.columns:
    X_raw_enh[f"log1p_{c}"] = np.log1p(np.abs(X_raw_enh[c]))
    X_test_raw_enh[f"log1p_{c}"] = np.log1p(np.abs(X_test_raw_enh[c]))

scaler = RobustScaler()
X_raw_train = scaler.fit_transform(X_raw_enh.astype("float32"))
X_raw_test  = scaler.transform(X_test_raw_enh.astype("float32"))

print("XGB raw-enh train shape:", X_raw_train.shape)
print("XGB raw-enh test  shape:", X_raw_test.shape)

y_arr = y

# =========================================================
# 4. Multi-seed dual-model CV (LGBM + XGB)
# =========================================================
SEEDS = [42, 777, 30251, 123, 2024]  # 5 seeds
N_FOLDS = 10

oof_lgb_all = np.zeros(len(y_arr))
oof_xgb_all = np.zeros(len(y_arr))
test_lgb_all = np.zeros(len(X_lgb_test))
test_xgb_all = np.zeros(len(X_raw_test))

for seed in SEEDS:
    print(f"\n================= SEED {seed} =================")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    oof_lgb_seed = np.zeros(len(y_arr))
    oof_xgb_seed = np.zeros(len(y_arr))
    test_lgb_seed = np.zeros(len(X_lgb_test))
    test_xgb_seed = np.zeros(len(X_raw_test))

    fold_id = 1
    for tr_idx, val_idx in skf.split(X_lgb_train, y_arr):
        X_tr_lgb, X_val_lgb = X_lgb_train[tr_idx], X_lgb_train[val_idx]
        X_tr_raw, X_val_raw = X_raw_train[tr_idx], X_raw_train[val_idx]
        y_tr, y_val = y_arr[tr_idx], y_arr[val_idx]

        # LGBM
        lgb_model = LGBMClassifier(
            n_estimators=900,
            max_depth=-1,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.8,
            objective="binary",
            reg_lambda=1.0,
            random_state=seed + fold_id,
            n_jobs=-1
        )
        lgb_model.fit(X_tr_lgb, y_tr)
        val_lgb = lgb_model.predict_proba(X_val_lgb)[:, 1]
        oof_lgb_seed[val_idx] = val_lgb
        test_lgb_seed += lgb_model.predict_proba(X_lgb_test)[:, 1] / N_FOLDS

        # XGB
        xgb_model = XGBClassifier(
            n_estimators=800,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="auc",
            reg_lambda=1.0,
            reg_alpha=0.0,
            scale_pos_weight=scale_pos,
            tree_method="hist",
            random_state=seed + fold_id,
            n_jobs=-1
        )
        xgb_model.fit(X_tr_raw, y_tr)
        val_xgb = xgb_model.predict_proba(X_val_raw)[:, 1]
        oof_xgb_seed[val_idx] = val_xgb
        test_xgb_seed += xgb_model.predict_proba(X_raw_test)[:, 1] / N_FOLDS

        print(f"  Seed {seed} | Fold {fold_id} AUCs -> "
              f"LGB: {roc_auc_score(y_val, val_lgb):.5f} | "
              f"XGB: {roc_auc_score(y_val, val_xgb):.5f}")
        fold_id += 1

    print(f"Seed {seed} full OOF AUCs: LGBM={roc_auc_score(y_arr, oof_lgb_seed):.5f} | "
          f"XGB={roc_auc_score(y_arr, oof_xgb_seed):.5f}")

    oof_lgb_all += oof_lgb_seed / len(SEEDS)
    oof_xgb_all += oof_xgb_seed / len(SEEDS)
    test_lgb_all += test_lgb_seed / len(SEEDS)
    test_xgb_all += test_xgb_seed / len(SEEDS)

# =========================================================
# 5. 2-model weight search
# =========================================================
auc_lgb = roc_auc_score(y_arr, oof_lgb_all)
auc_xgb = roc_auc_score(y_arr, oof_xgb_all)
print("\n==== Multi-seed Base Model OOF AUCs (V28) ====")
print(f"LGBM OOF AUC: {auc_lgb:.5f}")
print(f"XGB  OOF AUC: {auc_xgb:.5f}")

weights = np.linspace(0, 1, 41)
best_auc = 0.0
best_w = None

for w_xgb in weights:
    w_lgb = 1.0 - w_xgb
    blend_oof = w_xgb * oof_xgb_all + w_lgb * oof_lgb_all
    auc = roc_auc_score(y_arr, blend_oof)
    if auc > best_auc:
        best_auc = auc
        best_w = (w_xgb, w_lgb)

print("\nBest 2-model blend weights (XGB, LGBM):", best_w)
print("Best blended OOF AUC:", round(best_auc, 5))

w_xgb, w_lgb = best_w

# =========================================================
# 6. LGBM_PLUS version (add 5% more LGBM weight)
# =========================================================
w_lgb_plus = min(w_lgb + 0.05, 1.0)
w_xgb_minus = 1.0 - w_lgb_plus

final_test_pred = w_xgb_minus * test_xgb_all + w_lgb_plus * test_lgb_all

submission = pd.DataFrame({
    "ID": test_ids,
    "class": final_test_pred
})
submission.to_csv(OUT_PATH, index=False)

print(f"\n{'='*60}")
print(f"V28 MINIMAL LGBM_PLUS (0.90879 version)")
print(f"{'='*60}")
print(f"Optimal OOF weights: XGB={w_xgb:.3f}, LGBM={w_lgb:.3f}")
print(f"LGBM_PLUS weights:   XGB={w_xgb_minus:.3f}, LGBM={w_lgb_plus:.3f}")
print(f"\nSaved: {OUT_PATH}")
print(f"{'='*60}")
print(submission.head(10))
print(f"\nPrediction stats:")
print(f"  Mean: {final_test_pred.mean():.5f}")
print(f"  Std:  {final_test_pred.std():.5f}")
print(f"  Min:  {final_test_pred.min():.5f}")
print(f"  Max:  {final_test_pred.max():.5f}")