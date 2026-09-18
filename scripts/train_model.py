"""
train_model.py
---------------
Trains two models on the behavioral risk dataset (Dataset B):
  1. Isolation Forest (unsupervised anomaly detection)
  2. XGBoost classifier (supervised, using is_suspicious as label)

Compares both against the dataset's existing rule-based fraud_score.
Saves models to models/, and a comparison report to reports/.

Run:
    python3 scripts/train_model.py
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_recall_fscore_support,
    confusion_matrix,
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    from sklearn.ensemble import GradientBoostingClassifier

CLEAN_DIR = Path(__file__).resolve().parent.parent / "data" / "cleaned"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# NOTE: fraud_score and agent_type are deliberately EXCLUDED from the ML feature
# set. fraud_score is the source the dataset used to derive is_suspicious, so
# including it causes data leakage (the model would just learn to copy it,
# giving a meaningless "perfect" score). agent_type is likewise a near-direct
# proxy for the label. Excluding both forces the model to learn risk purely
# from raw transactional/behavioral signals - a fairer and more realistic test.
FEATURES = [
    "amount", "attempt_count", "hour_of_day", "is_night_transaction", "is_weekend",
    "switching_pct", "velocity_count",
]
CATEGORICAL = ["payment_method", "upi_app", "bank", "status", "amount_slab"]


def load_data():
    df = pd.read_csv(CLEAN_DIR / "fact_risk_signals.csv")
    df["is_night_transaction"] = df["is_night_transaction"].astype(int)
    df["is_weekend"] = df["is_weekend"].astype(int)

    encoders = {}
    for col in CATEGORICAL:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    feature_cols = FEATURES + [c + "_enc" for c in CATEGORICAL]
    return df, feature_cols, encoders


def train_isolation_forest(df, feature_cols):
    X = df[feature_cols].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso = IsolationForest(contamination=0.10, random_state=42, n_estimators=200)
    df["iso_anomaly"] = iso.fit_predict(X_scaled)          # -1 = anomaly, 1 = normal
    df["iso_risk_score"] = -iso.decision_function(X_scaled)  # higher = riskier

    df["iso_flag"] = (df["iso_anomaly"] == -1).astype(int)

    joblib.dump(iso, MODELS_DIR / "isolation_forest.pkl")
    joblib.dump(scaler, MODELS_DIR / "iso_scaler.pkl")

    # Compare against is_suspicious label
    precision, recall, f1, _ = precision_recall_fscore_support(
        df["is_suspicious"], df["iso_flag"], average="binary", zero_division=0
    )
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}


def train_xgboost(df, feature_cols):
    X = df[feature_cols].fillna(0)
    y = df["is_suspicious"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    if HAS_XGB:
        model = XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
            random_state=42, eval_metric="logloss",
        )
    else:
        model = GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred).tolist()

    joblib.dump(model, MODELS_DIR / "xgboost_risk_model.pkl")

    # Feature importance
    importances = dict(zip(feature_cols, model.feature_importances_.tolist()))
    importances = dict(sorted(importances.items(), key=lambda x: -x[1]))

    return {
        "roc_auc": round(auc, 4),
        "precision": round(report["1"]["precision"], 3),
        "recall": round(report["1"]["recall"], 3),
        "f1": round(report["1"]["f1-score"], 3),
        "confusion_matrix": cm,
        "feature_importance": importances,
    }, model, X_test, y_test


def compare_with_rule_based(df):
    """How well does the dataset's own rule-based fraud_score predict is_suspicious?"""
    threshold = df["fraud_score"].quantile(0.90)  # top 10% flagged, matches ~contamination rate
    df["rule_flag"] = (df["fraud_score"] >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        df["is_suspicious"], df["rule_flag"], average="binary", zero_division=0
    )
    return {"threshold": float(threshold), "precision": round(precision, 3),
            "recall": round(recall, 3), "f1": round(f1, 3)}


def main():
    df, feature_cols, encoders = load_data()

    print("Training Isolation Forest...")
    iso_results = train_isolation_forest(df, feature_cols)

    print("Training XGBoost classifier...")
    xgb_results, model, X_test, y_test = train_xgboost(df, feature_cols)

    print("Evaluating existing rule-based fraud_score...")
    rule_results = compare_with_rule_based(df)

    comparison = {
        "rule_based_fraud_score": rule_results,
        "isolation_forest": iso_results,
        "xgboost": {k: v for k, v in xgb_results.items() if k != "feature_importance"},
        "xgboost_top_features": dict(list(xgb_results["feature_importance"].items())[:8]),
    }

    with open(REPORTS_DIR / "model_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    with open(REPORTS_DIR / "model_comparison.md", "w") as f:
        f.write("# Model Comparison: Rule-Based vs Isolation Forest vs XGBoost\n\n")
        f.write("All models evaluated against the `is_suspicious` label.\n\n")
        f.write("| Model | Precision | Recall | F1 |\n|---|---|---|---|\n")
        f.write(f"| Rule-based fraud_score (top 10%) | {rule_results['precision']} | {rule_results['recall']} | {rule_results['f1']} |\n")
        f.write(f"| Isolation Forest (unsupervised) | {iso_results['precision']} | {iso_results['recall']} | {iso_results['f1']} |\n")
        f.write(f"| XGBoost (supervised) | {xgb_results['precision']} | {xgb_results['recall']} | {xgb_results['f1']} |\n\n")
        f.write(f"XGBoost ROC-AUC: **{xgb_results['roc_auc']}**\n\n")
        f.write("## Top features driving XGBoost predictions\n\n")
        for feat, imp in list(xgb_results["feature_importance"].items())[:8]:
            f.write(f"- {feat}: {round(imp, 4)}\n")

    # Save enriched dataset (with model scores) for the dashboard
    df.to_csv(CLEAN_DIR / "fact_risk_signals_scored.csv", index=False)

    print("\n=== SUMMARY ===")
    print(json.dumps(comparison, indent=2, default=str))
    print("\nSaved: models/isolation_forest.pkl, models/xgboost_risk_model.pkl")
    print("Saved: reports/model_comparison.md, reports/model_comparison.json")
    print("Saved: data/cleaned/fact_risk_signals_scored.csv (with model scores)")


if __name__ == "__main__":
    main()
