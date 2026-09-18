"""
clean_pipeline.py
------------------
Cleans the two raw datasets and produces a star schema in data/cleaned/.

Dataset A: upi_transactions_2024.csv        (250,000 rows) - macro/business layer
Dataset B: fraud_risk_signals.csv            (6,126 rows)   - micro/behavioral risk layer

Run:
    python3 scripts/clean_pipeline.py
"""

import json
import re
import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
CLEAN_DIR = Path(__file__).resolve().parent.parent / "data" / "cleaned"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# DATASET A: 2024 UPI Transactions (macro / business layer)
# ---------------------------------------------------------------------
def clean_dataset_a():
    print("Cleaning Dataset A: upi_transactions_2024.csv ...")
    df = pd.read_csv(RAW_DIR / "upi_transactions_2024.csv")

    # Standardize column names
    df.columns = [c.strip().lower().replace(" ", "_").replace("(inr)", "inr") for c in df.columns]
    df = df.rename(columns={"transaction_id": "txn_id", "amount_inr": "amount"})

    # Parse timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    n_bad_ts = df["timestamp"].isna().sum()

    # Flag (don't drop) bad rows
    df["amount_is_valid"] = df["amount"] > 0
    df["timestamp_is_valid"] = df["timestamp"].notna()

    # Standardize categoricals
    df["transaction_status"] = df["transaction_status"].str.upper().str.strip()
    df["transaction_type"] = df["transaction_type"].str.upper().str.strip()
    df["merchant_category"] = df["merchant_category"].str.title().str.strip()
    df["sender_state"] = df["sender_state"].str.title().str.strip()

    # Duplicates
    n_dupes = df.duplicated(subset="txn_id").sum()
    df = df.drop_duplicates(subset="txn_id", keep="first")

    # ---- Build star schema ----
    fact_transactions = df[[
        "txn_id", "timestamp", "transaction_type", "merchant_category", "amount",
        "transaction_status", "sender_state", "sender_bank", "receiver_bank",
        "sender_age_group", "receiver_age_group", "device_type", "network_type",
        "fraud_flag", "hour_of_day", "day_of_week", "is_weekend",
        "amount_is_valid", "timestamp_is_valid",
    ]].copy()

    dim_states = (
        fact_transactions.groupby("sender_state")
        .agg(total_txns=("txn_id", "count"), total_amount=("amount", "sum"),
             fraud_count=("fraud_flag", "sum"))
        .reset_index()
    )

    dim_banks = (
        fact_transactions.groupby("sender_bank")
        .agg(total_txns=("txn_id", "count"), total_amount=("amount", "sum"),
             fraud_count=("fraud_flag", "sum"))
        .reset_index()
    )

    dim_merchant_category = (
        fact_transactions.groupby("merchant_category")
        .agg(total_txns=("txn_id", "count"), total_amount=("amount", "sum"),
             fraud_count=("fraud_flag", "sum"))
        .reset_index()
    )

    fact_transactions.to_csv(CLEAN_DIR / "fact_transactions_a.csv", index=False)
    dim_states.to_csv(CLEAN_DIR / "dim_states.csv", index=False)
    dim_banks.to_csv(CLEAN_DIR / "dim_banks.csv", index=False)
    dim_merchant_category.to_csv(CLEAN_DIR / "dim_merchant_category.csv", index=False)

    report = {
        "raw_rows": len(pd.read_csv(RAW_DIR / "upi_transactions_2024.csv")),
        "clean_rows": len(fact_transactions),
        "duplicates_removed": int(n_dupes),
        "bad_timestamps_flagged": int(n_bad_ts),
        "fraud_rate_pct": round(fact_transactions["fraud_flag"].mean() * 100, 4),
    }
    print("Dataset A summary:", report)
    return report


# ---------------------------------------------------------------------
# DATASET B: Fraud Risk Signals (micro / behavioral layer)
# ---------------------------------------------------------------------
def parse_fraud_reasons(x):
    """Parse the stringified list in fraud_reasons into structured flags."""
    if pd.isna(x):
        return {"has_upi_switching": False, "switching_pct": 0.0,
                "has_high_velocity": False, "velocity_count": 0}
    try:
        reasons = json.loads(x.replace("'", '"'))
    except Exception:
        reasons = re.findall(r'"([^"]+)"', str(x))

    has_switching, switching_pct = False, 0.0
    has_velocity, velocity_count = False, 0

    for r in reasons:
        m1 = re.search(r"UPI switching:\s*([\d.]+)%", r)
        if m1:
            has_switching = True
            switching_pct = float(m1.group(1))
        m2 = re.search(r"High velocity:\s*(\d+)\s*txns", r)
        if m2:
            has_velocity = True
            velocity_count = int(m2.group(1))

    return {"has_upi_switching": has_switching, "switching_pct": switching_pct,
            "has_high_velocity": has_velocity, "velocity_count": velocity_count}


def clean_dataset_b():
    print("Cleaning Dataset B: fraud_risk_signals.csv ...")
    df = pd.read_csv(RAW_DIR / "fraud_risk_signals.csv")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["error_code"] = df["error_code"].fillna("NONE")  # NaN = no error, expected when status=success

    # Parse fraud_reasons into structured columns
    parsed = df["fraud_reasons"].apply(parse_fraud_reasons).apply(pd.Series)
    df = pd.concat([df, parsed], axis=1)

    # Rename device_fingerprint -> device_id (our customer proxy)
    df = df.rename(columns={"device_fingerprint": "device_id"})

    # ---- dim_devices (derived "customer" dimension) ----
    dim_devices = (
        df.groupby("device_id")
        .agg(
            total_txns=("id", "count"),
            avg_amount=("amount", "mean"),
            avg_fraud_score=("fraud_score", "mean"),
            suspicious_txns=("is_suspicious", "sum"),
            dominant_agent_type=("agent_type", lambda x: x.mode().iloc[0]),
            max_attempt_count=("attempt_count", "max"),
        )
        .reset_index()
    )
    dim_devices["suspicious_rate_pct"] = round(
        dim_devices["suspicious_txns"] / dim_devices["total_txns"] * 100, 2
    )

    fact_risk_signals = df[[
        "id", "razorpay_payment_id", "timestamp", "device_id", "ip_address",
        "agent_type", "amount", "amount_slab", "payment_method", "upi_app", "bank",
        "status", "error_code", "fraud_score", "is_suspicious",
        "has_upi_switching", "switching_pct", "has_high_velocity", "velocity_count",
        "hour_of_day", "is_night_transaction", "is_weekend", "attempt_count",
    ]].rename(columns={"id": "txn_id"})

    fact_risk_signals.to_csv(CLEAN_DIR / "fact_risk_signals.csv", index=False)
    dim_devices.to_csv(CLEAN_DIR / "dim_devices.csv", index=False)

    report = {
        "raw_rows": len(df),
        "clean_rows": len(fact_risk_signals),
        "unique_devices": dim_devices.shape[0],
        "suspicious_rate_pct": round(fact_risk_signals["is_suspicious"].mean() * 100, 3),
        "missing_error_code_filled": int(df["error_code"].eq("NONE").sum()),
    }
    print("Dataset B summary:", report)
    return report


if __name__ == "__main__":
    report_a = clean_dataset_a()
    report_b = clean_dataset_b()

    with open(Path(__file__).resolve().parent.parent / "reports" / "data_quality_report.md", "w") as f:
        f.write("# Data Quality Report\n\n")
        f.write("## Dataset A: UPI Transactions 2024 (macro layer)\n\n")
        for k, v in report_a.items():
            f.write(f"- **{k}**: {v}\n")
        f.write("\n## Dataset B: Fraud Risk Signals (micro/behavioral layer)\n\n")
        for k, v in report_b.items():
            f.write(f"- **{k}**: {v}\n")

    print("\nDone. Cleaned files written to data/cleaned/. Report written to reports/data_quality_report.md")
