# UPI Fraud & Risk Intelligence

An end-to-end **UPI fraud analytics and machine learning system** combining business-level fraud intelligence with device-level behavioral risk scoring.

Built as an **MBA Capstone Project in Data Science**.

## 🚀 Live Demo

👉 **[Open the UPI Fraud & Risk Intelligence Dashboard](https://upi-fraud-intelligence-fiatpckcgktaaw66h25vns.streamlit.app/)**

The dashboard is deployed using Streamlit Community Cloud and provides interactive fraud analytics, behavioral risk analysis, and ML model comparison.

---

## 🚀 Project Overview

Digital payment systems generate large volumes of transaction data, making manual fraud detection difficult. This project develops a two-layer fraud intelligence framework for UPI transactions:

### 🔹 Layer 1 — Macro Transaction Intelligence

Analyzes **250,000 UPI transactions from 2024** to identify:

- Transaction volume and trends
- Fraud and dispute patterns
- State-wise risk
- Bank-wise risk
- Merchant-category patterns
- Transaction-type distributions
- Hourly transaction behavior
- Geographic patterns

### 🔹 Layer 2 — Micro Behavioral Risk Intelligence

Analyzes **6,126 device-level risk-tagged transactions** using machine learning and behavioral signals such as:

- Device fingerprinting
- Transaction amount
- Retry/attempt velocity
- Behavioral patterns
- Agent type
- Risk signals
- Suspicious transaction patterns

The ML layer compares:

- **Rule-Based Risk Score**
- **Isolation Forest**
- **XGBoost**

The objective is to demonstrate how transaction-level analytics and behavioral ML can be combined into a unified fraud-risk intelligence system.

---

## 🎯 Key Objectives

1. Analyze large-scale UPI transaction data.
2. Identify fraud and dispute patterns across different dimensions.
3. Build a clean analytical star schema.
4. Engineer behavioral risk features.
5. Detect anomalous transactions using Isolation Forest.
6. Build a supervised fraud-risk model using XGBoost.
7. Compare ML models with a rule-based risk approach.
8. Develop an interactive Streamlit dashboard.
9. Generate automated business and model reports.
10. Provide a framework that can potentially be extended to real-time fraud detection.

---

## 🏗️ System Architecture

```text
                    UPI FRAUD & RISK INTELLIGENCE
                              │
             ┌────────────────┴────────────────┐
             │                                 │
             ▼                                 ▼
     DATASET A – MACRO                 DATASET B – MICRO
     250K UPI Transactions             6,126 Risk Signals
             │                                 │
             ▼                                 ▼
     Data Cleaning                      Feature Engineering
             │                                 │
             ▼                                 ▼
       Star Schema                  Behavioral Risk Features
             │                                 │
             ▼                                 ▼
     Business Metrics                 ML Risk Detection
             │                         ┌───────┴────────┐
             │                         │                │
             │                         ▼                ▼
             │                  Isolation Forest     XGBoost
             │                         │                │
             └──────────────┬──────────┴────────────────┘
                            │
                            ▼
                  Streamlit Dashboard
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
           Business       Risk          Model
           Analytics    Analytics     Comparison
```

---

# 📂 Project Structure

```text
upi-fraud-intelligence/
│
├── data/
│   ├── raw/
│   │   └── # Original raw datasets
│   │
│   └── cleaned/
│       ├── dim_banks.csv
│       ├── dim_devices.csv
│       ├── dim_merchant_category.csv
│       ├── dim_states.csv
│       ├── fact_risk_signals.csv
│       ├── fact_risk_signals_scored.csv
│       └── fact_transactions_a.csv
│
├── scripts/
│   ├── clean_pipeline.py
│   ├── compute_metrics.py
│   └── train_model.py
│
├── models/
│   ├── iso_scaler.pkl
│   ├── isolation_forest.pkl
│   └── xgboost_risk_model.pkl
│
├── dashboard/
│   ├── app.py
│   └── metrics.json
│
├── reports/
│   ├── business_metrics_summary.md
│   ├── data_quality_report.md
│   ├── model_comparison.json
│   └── model_comparison.md
│
├── requirements.txt
├── .gitignore
├── .Rhistory
└── README.md
```

---

# 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Anomaly Detection | Isolation Forest |
| Supervised ML | XGBoost |
| Dashboard | Streamlit |
| Visualization | Plotly / Streamlit |
| Data Storage | CSV |
| Model Serialization | Pickle |
| Version Control | Git & GitHub |
| Deployment | Streamlit Community Cloud |

---

# 📊 Datasets

## Dataset A — UPI Transactions

**Purpose:** Macro-level business and transaction intelligence.

- Approximately **250,000 UPI transactions**
- Full year **2024**
- Used for business KPIs, geography, banking and merchant analysis.

Key analytical dimensions include:

- Date
- State
- Bank
- Merchant category
- Transaction type
- Transaction amount
- Fraud/dispute indicators
- Transaction status

---

## Dataset B — Fraud Risk Signals

**Purpose:** Micro-level behavioral fraud-risk analysis.

- Approximately **6,126 device-level risk-tagged transactions**
- Contains behavioral and device-related risk signals.

Important features include concepts such as:

- Device identity
- Transaction amount
- Attempt count
- Retry velocity
- Timing information
- Behavioral signals
- Suspicious indicators
- Risk labels

---

# 🧹 Data Processing Pipeline

The project follows a sequential data-processing pipeline.

```text
Raw Data
   │
   ▼
Data Cleaning
   │
   ▼
Validation & Quality Checks
   │
   ▼
Feature Engineering
   │
   ▼
Star Schema
   │
   ▼
Business Metrics
   │
   ▼
ML Risk Scoring
   │
   ▼
Dashboard & Reports
```

---

# ⚙️ Local Setup

## 1. Clone the Repository

```bash
git clone https://github.com/pklayal/upi-fraud-intelligence.git
cd upi-fraud-intelligence
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📥 Raw Data Setup

If the raw datasets are not included in the repository, place them inside:

```text
data/raw/
```

Expected files:

```text
data/raw/
├── upi_transactions_2024.csv
└── fraud_risk_signals.csv
```

> Raw datasets may be excluded from GitHub using `.gitignore` to keep the repository smaller.

---

# 🔄 Run the Complete Pipeline

Run the scripts in the following order.

## Step 1 — Data Cleaning

```bash
python scripts/clean_pipeline.py
```

This generates the cleaned analytical datasets:

```text
data/cleaned/
```

and:

```text
reports/data_quality_report.md
```

---

## Step 2 — Compute Business Metrics

```bash
python scripts/compute_metrics.py
```

This generates:

```text
dashboard/metrics.json
reports/business_metrics_summary.md
```

---

## Step 3 — Train Machine Learning Models

```bash
python scripts/train_model.py
```

This generates:

```text
models/iso_scaler.pkl
models/isolation_forest.pkl
models/xgboost_risk_model.pkl
```

and:

```text
reports/model_comparison.md
reports/model_comparison.json
```

---

# 📈 Streamlit Dashboard

Launch the dashboard using:

```bash
streamlit run dashboard/app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

---

# 🖥️ Dashboard Pages

The dashboard contains five major analytical sections.

## 1. Overview

Provides high-level transaction intelligence.

Includes:

- Total transaction volume
- Daily transaction trends
- Transaction-type distribution
- Merchant-category analysis
- Hourly transaction patterns
- Key business KPIs

---

## 2. Fraud & Disputes

Analyzes fraud-related patterns across major dimensions.

Includes:

- Fraud rate
- Fraud by state
- Fraud by bank
- Dispute patterns
- Transaction risk distribution

---

## 3. Geographic & Bank Risk

Provides geographical and financial-institution-level analysis.

Includes:

- State-wise transaction distribution
- Bank-wise transaction distribution
- Device/network patterns
- Geographic risk indicators

---

## 4. Behavioral Risk — ML

Provides device-level behavioral risk analysis.

Includes:

- Device-level risk exploration
- Behavioral segments
- Attempt/retry patterns
- Risk signals
- Suspicious transaction analysis
- ML-generated risk scores

---

## 5. Model Comparison

Compares multiple approaches to fraud-risk detection.

Models:

```text
Rule-Based Risk Score
        │
        ├──────────────┐
        ▼              ▼
Isolation Forest     XGBoost
```

Includes:

- Model performance comparison
- Risk predictions
- Feature importance
- Model-level metrics
- Behavioral feature analysis

---

# 🤖 Machine Learning Methodology

## Isolation Forest

Isolation Forest is used for **unsupervised anomaly detection**.

The basic idea is that unusual observations are easier to isolate than normal observations.

```text
Behavioral Data
      │
      ▼
Feature Scaling
      │
      ▼
Isolation Forest
      │
      ▼
Anomaly Score
      │
      ▼
Risk Classification
```

This approach is useful when fraud labels are limited or when the objective is to identify unusual behavioral patterns.

---

## XGBoost

XGBoost is used as a supervised machine-learning model for risk prediction.

The model learns relationships between behavioral features and the target risk label.

Conceptually:

```text
Behavioral Features
        │
        ▼
Feature Engineering
        │
        ▼
XGBoost
        │
        ▼
Risk Probability
        │
        ▼
Risk Classification
```

---

# 🔍 Feature Engineering

Behavioral risk features are designed to capture suspicious transaction behavior.

Examples include:

- Transaction amount
- Number of attempts
- Retry velocity
- Transaction timing
- Device-related signals
- Behavioral patterns
- Frequency-based indicators

These features are used to distinguish normal transaction behavior from potentially suspicious behavior.

---

# ⚠️ Data Leakage Consideration

Data leakage is an important consideration in this project.

The raw `fraud_risk_signals.csv` dataset contains a `fraud_score` field that was used by the dataset-generation process to derive the `is_suspicious` label.

If `fraud_score` is directly supplied to the model as an input feature, the model can effectively reproduce information already embedded in the target.

Therefore:

```text
fraud_score
agent_type
      │
      ▼
Excluded from ML feature set
```

The model instead uses behavioral signals such as:

```text
Transaction Amount
Attempt Count
Retry Velocity
Timing
Behavioral Features
Device Signals
```

This helps reduce the risk of reporting an artificially inflated model performance caused by target leakage.

---

# 📋 Generated Reports

The project automatically generates several reports.

## Data Quality Report

```text
reports/data_quality_report.md
```

Contains information about:

- Missing values
- Data types
- Dataset structure
- Cleaning operations
- Data validation

---

## Business Metrics Report

```text
reports/business_metrics_summary.md
```

Contains:

- Transaction KPIs
- Fraud metrics
- Geographic analysis
- Bank analysis
- Merchant-category analysis

---

## Model Comparison Report

```text
reports/model_comparison.md
```

Contains:

- Model performance
- Rule-based comparison
- Isolation Forest results
- XGBoost results
- Feature importance
- Risk analysis

---

# 🔐 Data & Privacy

The project is designed for academic and analytical purposes.

Sensitive production payment information should not be committed to a public repository.

Recommended practice:

```text
data/raw/
```

should be excluded from Git when the files contain:

- Personally identifiable information
- Sensitive transaction information
- Proprietary data
- Large unnecessary raw datasets

The `.gitignore` file can be used to control which files are uploaded.

---

# 🚀 Deployment

## Streamlit Community Cloud

The dashboard can be deployed using Streamlit Community Cloud.

### Step 1 — Push the Project to GitHub

```bash
git add .
git commit -m "Update UPI fraud intelligence project"
git push
```

---

### Step 2 — Open Streamlit Community Cloud

Go to:

```text
https://share.streamlit.io
```

Sign in using GitHub.

---

### Step 3 — Create a New App

Select:

```text
Repository:
pklayal/upi-fraud-intelligence

Branch:
main

Main file:
dashboard/app.py
```

Then deploy the application.

---

### Step 4 — Generated Files

The deployed application requires the generated files to already exist in the repository.

Make sure these are available:

```text
data/cleaned/
dashboard/metrics.json
models/
reports/model_comparison.json
```

Streamlit Community Cloud runs the dashboard application; it does not automatically execute the entire local data-processing pipeline before launching the app.

---

# 🌐 Live Dashboard

After deployment, the application will have a Streamlit URL similar to:

```text
https://upi-fraud-intelligence.streamlit.app
```

Replace the URL above with the actual deployed URL once the application is published.

---

# 📌 Common Deployment Issues

| Problem | Possible Solution |
|---|---|
| `FileNotFoundError: dashboard/metrics.json` | Run `compute_metrics.py` and commit the generated file |
| `FileNotFoundError` for cleaned CSVs | Run `clean_pipeline.py` and commit `data/cleaned/` |
| `ModuleNotFoundError: xgboost` | Ensure `xgboost` is listed in `requirements.txt` |
| Blank dashboard | Verify cleaned CSV files exist in the repository |
| Model loading error | Verify `.pkl` files exist in `models/` |
| Large repository | Exclude raw datasets using `.gitignore` |
| Dashboard doesn't start | Verify `dashboard/app.py` is the correct main file |
| Dependency installation error | Check package versions in `requirements.txt` |

---

# 📦 Requirements

Main Python libraries used in the project include:

```text
pandas
numpy
scikit-learn
xgboost
streamlit
plotly
```

Complete dependencies are provided in:

```text
requirements.txt
```

Install them using:

```bash
pip install -r requirements.txt
```

---

# 🔬 Project Workflow

```text
                    RAW DATA
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
    UPI Transactions          Risk Signals
       Dataset A                 Dataset B
          │                         │
          ▼                         ▼
    Data Cleaning            Feature Engineering
          │                         │
          ▼                         ▼
    Star Schema              Behavioral Features
          │                         │
          ▼                         ▼
 Business Analytics       ┌─────────┴─────────┐
                          │                   │
                          ▼                   ▼
                   Isolation Forest       XGBoost
                          │                   │
                          └─────────┬─────────┘
                                    │
                                    ▼
                             Risk Intelligence
                                    │
                                    ▼
                           Streamlit Dashboard
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
                Overview       Fraud/Risk       ML Analysis
```

---

# 💡 Key Project Contribution

The project combines two complementary perspectives of fraud detection:

### Macro Perspective

```text
Where and when is fraud occurring?
```

Analyzes:

- States
- Banks
- Merchant categories
- Transaction types
- Time patterns
- Business-level KPIs

### Micro Perspective

```text
Why does a particular transaction/device appear risky?
```

Analyzes:

- Device behavior
- Retry velocity
- Transaction attempts
- Behavioral signals
- Anomaly scores
- ML risk predictions

Together, these layers provide a more comprehensive **UPI Fraud & Risk Intelligence Framework**.

---

# 📚 Academic Use

This project was developed as an:

**MBA Capstone Project — Data Science**

The project demonstrates practical application of:

- Data Engineering
- Data Cleaning
- Exploratory Data Analysis
- Feature Engineering
- Machine Learning
- Anomaly Detection
- Supervised Learning
- Business Intelligence
- Data Visualization
- Dashboard Development
- Model Comparison
- Git/GitHub
- Cloud Deployment

---

# 🔮 Future Scope

Potential extensions include:

1. Real-time UPI transaction monitoring.
2. Kafka-based transaction streaming.
3. Real-time fraud scoring APIs.
4. Online/continuous model learning.
5. Graph-based fraud detection.
6. Device-network relationship analysis.
7. Explainable AI using SHAP.
8. Automated fraud alerts.
9. Model drift monitoring.
10. Integration with production payment systems.
11. Real-time risk dashboards.
12. Advanced ensemble fraud-detection models.

---

# 👨‍💻 Author

**Pohap Kumar Layal**

MBA — Data Science  
Lovely Professional University

---

# ⭐ Project Highlights

```text
✓ 250,000 UPI transactions analyzed
✓ 6,126 device-level risk records analyzed
✓ End-to-end data processing pipeline
✓ Star-schema analytical model
✓ Business KPI generation
✓ Behavioral feature engineering
✓ Isolation Forest anomaly detection
✓ XGBoost risk modeling
✓ Rule-based vs ML model comparison
✓ Interactive 5-page Streamlit dashboard
✓ Automated analytical reports
✓ GitHub version control
✓ Cloud deployment ready
```

---

# 📜 License

This project is developed for **academic and educational purposes**.

Please verify the licensing and permitted use of any third-party datasets before redistribution or commercial use.

---

# ⭐ Acknowledgement

This project was developed as part of an MBA Data Science capstone project, demonstrating the application of data analytics and machine learning techniques to UPI fraud and risk intelligence.
