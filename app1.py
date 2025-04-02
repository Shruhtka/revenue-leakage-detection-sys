import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import joblib

st.set_page_config(layout="wide", page_title="Revenue Leakage Detection")

@st.cache_data
def load_default_data():
    return pd.read_csv("Synthetic_Financial_Transactions.csv")

@st.cache_resource
def load_models():
    models = {}
    try:
        models["Isolation Forest"] = joblib.load("if_model.pkl")
    except Exception as e:
        st.warning(f"Isolation Forest failed to load: {e}")
        models["Isolation Forest"] = IsolationForest(random_state=42)

    try:
        models["K-Means"] = joblib.load("kmeans_model.pkl")
    except Exception as e:
        st.warning(f"K-Means failed to load: {e}")
        models["K-Means"] = KMeans(n_clusters=2, random_state=42)
    try:
        models["DBSCAN"] = joblib.load("dbscan_model.pkl")
    except Exception as e:
        st.warning(f"DBSCAN fallback used: {e}")
        models["DBSCAN"] = DBSCAN(eps=0.5, min_samples=5)

    return models

def preprocess(df):
    df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'], errors='coerce')
    df['Payment_Date'] = pd.to_datetime(df['Payment_Date'], errors='coerce')
    df['PaymentDelay'] = (df['Payment_Date'] - df['Transaction_Date']).dt.days.fillna(0)
    df['DiscountApplied'] = df['Discount_Applied']
    df['RefundAmount'] = df['Refund_Issued']
    df['InvoiceAmount'] = df['Invoice_Amount']
    return df, df[['DiscountApplied', 'RefundAmount', 'InvoiceAmount', 'PaymentDelay']]

uploaded = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
data = pd.read_csv(uploaded) if uploaded else load_default_data()
data, X = preprocess(data)  # Now X is defined
# Convert Anomaly_Tag to binary
if 'Anomaly_Tag' in data.columns:
    data['Anomaly_Tag'] = data['Anomaly_Tag'].apply(lambda x: 0 if str(x).strip().lower() == 'normal' else 1)


# Now it's safe to load models
models = load_models()

def run_model(name, X):
    if name == "Hybrid (IF + DBSCAN)":
        if_preds = np.where(models["Isolation Forest"].predict(X) == -1, 1, 0)
        db_preds = np.where(models["DBSCAN"].fit_predict(X) == -1, 1, 0)
        return ((if_preds + db_preds) >= 2).astype(int)
    elif name == "DBSCAN":
        preds = models["DBSCAN"].fit_predict(X)
        return np.where(preds == -1, 1, 0)
    elif name == "K-Means":
        km = models["K-Means"]
        if not hasattr(km, "cluster_centers_"):
            km.fit(X)
        labels = km.predict(X)
        dists = np.linalg.norm(X.values - km.cluster_centers_[labels], axis=1)
        return (dists > np.percentile(dists, 90)).astype(int)
    else:
        return np.where(models[name].predict(X) == -1, 1, 0)

def evaluate(y_true, y_pred):
    y_true = pd.to_numeric(y_true, errors='coerce').fillna(0).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    fpr = cm[0][1] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) > 0 else 0.0
    return precision, recall, f1, fpr

def explain_with_shap(model, X):
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        return explainer.expected_value, shap_values
    except Exception as e:
        st.warning(f"SHAP explainability failed: {e}")
        return None, None

st.title("💸 Revenue Leakage Detection Dashboard")
tabs = st.tabs(["📊 Overview", "🔍 Model Output", "📈 SHAP Explainability", "📊 Compare Models"])

with tabs[0]:
    st.markdown("""
    This dashboard identifies **operational revenue leakages**:
    - Duplicate invoices
    - Missed payments
    - Unauthorized refunds or discounts
    Models supported: Isolation Forest, DBSCAN, K-Means, Hybrid (IF + DBSCAN)
    SHAP explanations are available for Isolation Forest only.
    """)

with tabs[1]:
    model_option = st.selectbox("Choose Model", list(models.keys()) + ["Hybrid (IF + DBSCAN)"])
    predictions = run_model(model_option, X)
    data['Prediction'] = predictions

    if 'Anomaly_Tag' in data.columns:
        precision, recall, f1, fpr = evaluate(data['Anomaly_Tag'], predictions)
    else:
        precision = recall = f1 = fpr = 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Precision", f"{precision:.2f}")
    col2.metric("Recall", f"{recall:.2f}")
    col3.metric("F1 Score", f"{f1:.2f}")
    col4.metric("False Positive Rate", f"{fpr:.2%}")

    st.subheader("🔎 Flagged Anomalies")
    anomalies = data[data['Prediction'] == 1]
    st.dataframe(anomalies, use_container_width=True)
    st.download_button("📥 Download Anomalies", anomalies.to_csv(index=False), "anomalies.csv")

    st.subheader("📈 Feature Scatterplot")
    xcol, ycol = st.selectbox("X-Axis", X.columns), st.selectbox("Y-Axis", X.columns, index=1)
    fig, ax = plt.subplots()
    sns.scatterplot(x=X[xcol], y=X[ycol], hue=predictions, palette=['blue', 'red'], ax=ax)
    ax.set_title("Anomaly Distribution")
    st.pyplot(fig)

with tabs[2]:
    st.subheader("SHAP Global Importance (IF Only)")
    if "Isolation Forest" in models:
        expected_val, shap_values = explain_with_shap(models["Isolation Forest"], X)
        if shap_values is not None:
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, X, plot_type="bar", show=False)
            st.pyplot(fig)

            st.subheader("SHAP Force Plot (Local Explanation)")
            index = st.number_input("Select Index", 0, len(X)-1, 0)
            try:
                st_shap = shap.force_plot(base_value=expected_val,
                                          shap_values=shap_values[index],
                                          features=X.iloc[index],
                                          matplotlib=True, show=False)
                st.pyplot(st_shap)
            except Exception as e:
                st.warning(f"SHAP force plot failed: {e}")
    else:
        st.warning("Isolation Forest model not available or incompatible.")

with tabs[3]:
    st.subheader("📊 Model Performance Comparison")
    results = []
    for name in list(models.keys()) + ["Hybrid (IF + DBSCAN)"]:
        preds = run_model(name, X)
        if 'Anomaly_Tag' in data.columns:
            p, r, f, fpr = evaluate(data['Anomaly_Tag'], preds)
        else:
            p = r = f = fpr = 0.0
        results.append({"Model": name, "Precision": p, "Recall": r, "F1 Score": f, "False Positive Rate": fpr})
    df = pd.DataFrame(results)
    st.dataframe(df.style.format({"Precision": "{:.2f}", "Recall": "{:.2f}", "F1 Score": "{:.2f}", "False Positive Rate": "{:.2%}"}))
    st.download_button("📥 Download Comparison Table", df.to_csv(index=False), "model_comparison.csv")
