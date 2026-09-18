"""
compute_metrics.py
-------------------
Computes business KPIs from the cleaned star schema and writes:
  - reports/business_metrics_summary.md   (human-readable)
  - dashboard/metrics.json                (consumed by the Streamlit app)

Run:
    python3 scripts/compute_metrics.py
"""

import json
import pandas as pd
from pathlib import Path

CLEAN_DIR = Path(__file__).resolve().parent.parent / "data" / "cleaned"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
DASHBOARD_DIR = Path(__file__).resolve().parent.parent / "dashboard"
REPORTS_DIR.mkdir(exist_ok=True)
DASHBOARD_DIR.mkdir(exist_ok=True)


def compute():
    fact_a = pd.read_csv(CLEAN_DIR / "fact_transactions_a.csv", parse_dates=["timestamp"])
    fact_b = pd.read_csv(CLEAN_DIR / "fact_risk_signals.csv", parse_dates=["timestamp"])

    metrics = {}

    # --- Dataset A: macro KPIs ---
    metrics["total_transactions"] = int(len(fact_a))
    metrics["total_amount_inr"] = float(fact_a["amount"].sum())
    metrics["avg_transaction_value"] = float(fact_a["amount"].mean())
    metrics["success_rate_pct"] = round((fact_a["transaction_status"] == "SUCCESS").mean() * 100, 2)
    metrics["failed_rate_pct"] = round((fact_a["transaction_status"] == "FAILED").mean() * 100, 2)
    metrics["fraud_rate_pct"] = round(fact_a["fraud_flag"].mean() * 100, 4)
    metrics["fraud_txn_count"] = int(fact_a["fraud_flag"].sum())

    metrics["txn_type_split"] = fact_a["transaction_type"].value_counts().to_dict()
    metrics["top_merchant_categories"] = (
        fact_a.groupby("merchant_category")["amount"].sum().sort_values(ascending=False).head(10).to_dict()
    )
    metrics["fraud_by_state"] = (
        fact_a.groupby("sender_state")["fraud_flag"].agg(["sum", "count"])
        .assign(rate_pct=lambda d: round(d["sum"] / d["count"] * 100, 4))
        .sort_values("rate_pct", ascending=False).head(10)
        .reset_index().to_dict(orient="records")
    )
    metrics["fraud_by_bank"] = (
        fact_a.groupby("sender_bank")["fraud_flag"].agg(["sum", "count"])
        .assign(rate_pct=lambda d: round(d["sum"] / d["count"] * 100, 4))
        .sort_values("rate_pct", ascending=False)
        .reset_index().to_dict(orient="records")
    )
    metrics["hourly_volume"] = fact_a.groupby("hour_of_day")["txn_id"].count().to_dict()
    metrics["device_type_split"] = fact_a["device_type"].value_counts().to_dict()
    metrics["network_type_split"] = fact_a["network_type"].value_counts().to_dict()
    metrics["daily_volume"] = (
        fact_a.assign(date=fact_a["timestamp"].dt.date)
        .groupby("date")["txn_id"].count()
        .reset_index().astype(str).to_dict(orient="records")
    )

    # --- Dataset B: micro/behavioral KPIs ---
    metrics["risk_total_txns"] = int(len(fact_b))
    metrics["risk_unique_devices"] = int(fact_b["device_id"].nunique())
    metrics["suspicious_rate_pct"] = round(fact_b["is_suspicious"].mean() * 100, 2)
    metrics["avg_fraud_score"] = round(fact_b["fraud_score"].mean(), 2)
    metrics["agent_type_split"] = fact_b["agent_type"].value_counts().to_dict()
    metrics["high_velocity_txns"] = int(fact_b["has_high_velocity"].sum())
    metrics["upi_switching_txns"] = int(fact_b["has_upi_switching"].sum())
    metrics["avg_attempt_count"] = round(fact_b["attempt_count"].mean(), 1)
    metrics["max_attempt_count"] = int(fact_b["attempt_count"].max())
    metrics["suspicious_by_agent_type"] = (
        fact_b.groupby("agent_type")["is_suspicious"].agg(["sum", "count"])
        .assign(rate_pct=lambda d: round(d["sum"] / d["count"] * 100, 2))
        .reset_index().to_dict(orient="records")
    )
    metrics["error_code_split"] = (
        fact_b[fact_b["error_code"] != "NONE"]["error_code"].value_counts().to_dict()
    )

    # Save
    with open(DASHBOARD_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    with open(REPORTS_DIR / "business_metrics_summary.md", "w") as f:
        f.write("# Business Metrics Summary\n\n")
        f.write("## Macro layer (Dataset A - 2024 transactions)\n\n")
        f.write(f"- Total transactions: {metrics['total_transactions']:,}\n")
        f.write(f"- Total amount: Rs. {metrics['total_amount_inr']:,.0f}\n")
        f.write(f"- Avg transaction value: Rs. {metrics['avg_transaction_value']:,.2f}\n")
        f.write(f"- Success rate: {metrics['success_rate_pct']}%\n")
        f.write(f"- Fraud rate: {metrics['fraud_rate_pct']}% ({metrics['fraud_txn_count']} txns) "
                f"- NOTE: highly imbalanced, a realistic challenge for fraud modeling\n\n")
        f.write("## Micro/behavioral layer (Dataset B - risk signals)\n\n")
        f.write(f"- Total risk-tagged transactions: {metrics['risk_total_txns']:,}\n")
        f.write(f"- Unique devices: {metrics['risk_unique_devices']}\n")
        f.write(f"- Suspicious rate: {metrics['suspicious_rate_pct']}%\n")
        f.write(f"- Avg rule-based fraud score: {metrics['avg_fraud_score']}\n")
        f.write(f"- Avg attempt count per txn: {metrics['avg_attempt_count']} (max {metrics['max_attempt_count']})\n")

    print("Metrics computed and saved to dashboard/metrics.json and reports/business_metrics_summary.md")
    return metrics


if __name__ == "__main__":
    compute()
