"""
app.py — UPI Fraud & Risk Intelligence Dashboard (Streamlit)

Run locally:
    streamlit run dashboard/app.py
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
CLEAN_DIR = BASE_DIR / "data" / "cleaned"
DASHBOARD_DIR = BASE_DIR / "dashboard"
REPORTS_DIR = BASE_DIR / "reports"

st.set_page_config(page_title="UPI Fraud & Risk Intelligence", layout="wide", page_icon="🛡️")


# ---------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------
@st.cache_data
def load_metrics():
    with open(DASHBOARD_DIR / "metrics.json") as f:
        return json.load(f)


@st.cache_data
def load_fact_a():
    return pd.read_csv(CLEAN_DIR / "fact_transactions_a.csv", parse_dates=["timestamp"])


@st.cache_data
def load_fact_b():
    path = CLEAN_DIR / "fact_risk_signals_scored.csv"
    if not path.exists():
        path = CLEAN_DIR / "fact_risk_signals.csv"
    return pd.read_csv(path, parse_dates=["timestamp"])


@st.cache_data
def load_model_comparison():
    path = REPORTS_DIR / "model_comparison.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


metrics = load_metrics()
fact_a = load_fact_a()
fact_b = load_fact_b()
model_comparison = load_model_comparison()


# ---------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------
st.sidebar.title("🛡️ UPI Risk Intelligence")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Fraud & Disputes", "Geographic & Bank Risk", "Behavioral Risk (ML)", "Model Comparison"],
)
st.sidebar.markdown("---")
st.sidebar.caption(
    f"Dataset A: {metrics['total_transactions']:,} transactions (2024)\n\n"
    f"Dataset B: {metrics['risk_total_txns']:,} risk-tagged transactions"
)


# ---------------------------------------------------------------------
# PAGE 1: Overview
# ---------------------------------------------------------------------
if page == "Overview":
    st.title("📊 Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", f"{metrics['total_transactions']:,}")
    c2.metric("Total Value", f"₹{metrics['total_amount_inr']/1e7:.2f} Cr")
    c3.metric("Success Rate", f"{metrics['success_rate_pct']}%")
    c4.metric("Fraud Rate", f"{metrics['fraud_rate_pct']}%")

    col1, col2 = st.columns(2)

    with col1:
        daily = pd.DataFrame(metrics["daily_volume"])
        daily["date"] = pd.to_datetime(daily["date"])
        fig = px.line(daily, x="date", y="txn_id", title="Daily Transaction Volume (2024)")
        fig.update_layout(yaxis_title="Transactions", xaxis_title="Date")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        split = pd.Series(metrics["txn_type_split"]).reset_index()
        split.columns = ["type", "count"]
        fig = px.pie(split, names="type", values="count", title="Transaction Type Split", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        cat = pd.Series(metrics["top_merchant_categories"]).reset_index()
        cat.columns = ["category", "amount"]
        fig = px.bar(cat, x="category", y="amount", title="Total Value by Merchant Category")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        hourly = pd.Series(metrics["hourly_volume"]).sort_index().reset_index()
        hourly.columns = ["hour", "count"]
        fig = px.bar(hourly, x="hour", y="count", title="Transaction Volume by Hour of Day")
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------
# PAGE 2: Fraud & Disputes
# ---------------------------------------------------------------------
elif page == "Fraud & Disputes":
    st.title("🚨 Fraud & Disputes")
    st.info(
        f"Fraud is extremely rare in this dataset: **{metrics['fraud_rate_pct']}%** "
        f"({metrics['fraud_txn_count']} of {metrics['total_transactions']:,} transactions). "
        "This class imbalance is realistic and is discussed in the model comparison section."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Fraud Transactions", metrics["fraud_txn_count"])
    with c2:
        st.metric("Fraud Rate", f"{metrics['fraud_rate_pct']}%")

    fraud_state = pd.DataFrame(metrics["fraud_by_state"])
    fig = px.bar(
        fraud_state, x="sender_state", y="rate_pct",
        title="Fraud Rate by State (Top 10)", labels={"rate_pct": "Fraud Rate (%)", "sender_state": "State"},
    )
    st.plotly_chart(fig, use_container_width=True)

    fraud_bank = pd.DataFrame(metrics["fraud_by_bank"])
    fig = px.bar(
        fraud_bank, x="sender_bank", y="rate_pct",
        title="Fraud Rate by Bank", labels={"rate_pct": "Fraud Rate (%)", "sender_bank": "Bank"},
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------
# PAGE 3: Geographic & Bank Risk
# ---------------------------------------------------------------------
elif page == "Geographic & Bank Risk":
    st.title("🗺️ Geographic & Bank Risk")

    dim_states = pd.read_csv(CLEAN_DIR / "dim_states.csv")
    dim_banks = pd.read_csv(CLEAN_DIR / "dim_banks.csv")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Transactions by State")
        fig = px.bar(dim_states.sort_values("total_txns", ascending=False).head(15),
                     x="sender_state", y="total_txns")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Transactions by Bank")
        fig = px.bar(dim_banks.sort_values("total_txns", ascending=False), x="sender_bank", y="total_txns")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Device & Network Patterns")
    col3, col4 = st.columns(2)
    with col3:
        dsplit = pd.Series(metrics["device_type_split"]).reset_index()
        dsplit.columns = ["device", "count"]
        fig = px.pie(dsplit, names="device", values="count", title="Device Type Split")
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        nsplit = pd.Series(metrics["network_type_split"]).reset_index()
        nsplit.columns = ["network", "count"]
        fig = px.pie(nsplit, names="network", values="count", title="Network Type Split")
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------
# PAGE 4: Behavioral Risk (Dataset B)
# ---------------------------------------------------------------------
elif page == "Behavioral Risk (ML)":
    st.title("🤖 Behavioral Risk Intelligence")
    st.caption("Device-level, behavioral risk signals from the micro dataset (6,126 transactions, 250 devices)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Suspicious Rate", f"{metrics['suspicious_rate_pct']}%")
    c2.metric("Avg Fraud Score", metrics["avg_fraud_score"])
    c3.metric("High Velocity Txns", metrics["high_velocity_txns"])
    c4.metric("Max Attempt Count", metrics["max_attempt_count"])

    col1, col2 = st.columns(2)
    with col1:
        agent = pd.Series(metrics["agent_type_split"]).reset_index()
        agent.columns = ["agent_type", "count"]
        fig = px.bar(agent, x="agent_type", y="count", title="Agent Type Distribution (Behavioral Segments)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        susp_agent = pd.DataFrame(metrics["suspicious_by_agent_type"])
        fig = px.bar(susp_agent, x="agent_type", y="rate_pct",
                     title="Suspicious Rate by Agent Type", labels={"rate_pct": "Suspicious Rate (%)"})
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Device Risk Explorer")
    dim_devices = pd.read_csv(CLEAN_DIR / "dim_devices.csv")
    top_risky = dim_devices.sort_values("suspicious_rate_pct", ascending=False).head(20)
    st.dataframe(top_risky, use_container_width=True)

    if "iso_risk_score" in fact_b.columns:
        st.subheader("ML Risk Score Distribution (Isolation Forest)")
        fig = px.histogram(fact_b, x="iso_risk_score", color="is_suspicious",
                            title="Isolation Forest Risk Score vs Actual Suspicious Label", nbins=50)
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------
# PAGE 5: Model Comparison
# ---------------------------------------------------------------------
elif page == "Model Comparison":
    st.title("📈 Model Comparison: Rule-Based vs ML")

    if model_comparison is None:
        st.warning("Run `python3 scripts/train_model.py` first to generate model comparison results.")
    else:
        rb = model_comparison["rule_based_fraud_score"]
        iso = model_comparison["isolation_forest"]
        xgb = model_comparison["xgboost"]

        comp_df = pd.DataFrame([
            {"Model": "Rule-based fraud_score (dataset default)", "Precision": rb["precision"],
             "Recall": rb["recall"], "F1": rb["f1"]},
            {"Model": "Isolation Forest (unsupervised, our build)", "Precision": iso["precision"],
             "Recall": iso["recall"], "F1": iso["f1"]},
            {"Model": "XGBoost (supervised, our build)", "Precision": xgb["precision"],
             "Recall": xgb["recall"], "F1": xgb["f1"]},
        ])

        st.dataframe(comp_df, use_container_width=True)

        fig = px.bar(comp_df.melt(id_vars="Model", var_name="Metric", value_name="Score"),
                     x="Model", y="Score", color="Metric", barmode="group",
                     title="Precision / Recall / F1 Comparison")
        st.plotly_chart(fig, use_container_width=True)

        st.metric("XGBoost ROC-AUC", xgb["roc_auc"])

        st.subheader("Top Predictive Features (XGBoost)")
        top_feats = pd.Series(model_comparison["xgboost_top_features"]).reset_index()
        top_feats.columns = ["feature", "importance"]
        fig = px.bar(top_feats, x="feature", y="importance", title="Feature Importance")
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "**Key finding:** `velocity_count` (transactions-per-minute burst behavior) is by far "
            "the strongest predictor of suspicious activity — stronger than transaction amount or "
            "time-of-day patterns. This supports prioritizing velocity-based rate-limiting as a "
            "practical fraud-prevention control."
        )

st.sidebar.markdown("---")
st.sidebar.caption("Built for MBA Capstone Project · Data Science")
