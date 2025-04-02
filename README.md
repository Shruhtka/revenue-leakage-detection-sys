# 💸 Revenue Leakage Detection Dashboard

This interactive Streamlit dashboard uses unsupervised machine learning models to detect **non-fraudulent financial anomalies** such as:

- Duplicate transactions
- Missing payments
- Unauthorized refunds or discounts

It’s designed to help businesses reduce silent revenue losses and improve transparency using interpretable AI.

---

## 🚀 Features

- 📤 Upload your own CSV datasets in real-time
- 🧠 Toggle between 3 anomaly detection models:
  - Isolation Forest
  - DBSCAN
  - K-Means
- 🎯 View model precision, recall, and F1-score
- 📋 Inspect flagged transactions via interactive tables
- 📈 Visualize anomalies with scatter plots
- 🧠 SHAP Explainability (for Isolation Forest)
- 💾 Download detected anomalies for audit use

---

## 📂 Dataset Format

To use your own data, upload a `.csv` with the following columns:

```csv
Transaction_ID,Invoice_Amount,Discount_Applied,Refund_Issued,Transaction_Type,Anomaly_Tag
