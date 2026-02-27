import streamlit as st
import joblib
import pandas as pd
import numpy as np
import time
import os
import plotly.graph_objects as go # For radar chart on Page 2
import matplotlib.pyplot as plt
import shap
import lime
import lime.lime_tabular

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="FraudX AI - Production System", layout="wide", page_icon="🛡️")

# --- 2. LOAD PRE-TRAINED ASSETS ---
def load_fraud_assets():
    # Chemins relatifs vers tes fichiers
    model = joblib.load('Desktop/stat_app/fraudx_model_final.pkl')
    scaler = joblib.load('Desktop/stat_app/scaler_fraud.pkl')
    iso_forest = joblib.load('Desktop/stat_app/iso_forest_model.pkl')
    kmeans = joblib.load('Desktop/stat_app/kmeans_clusterer.pkl')
    return model, scaler, iso_forest, kmeans

try:
    model, scaler, iso_forest, kmeans = load_fraud_assets()
except Exception as e:
    st.error(f"Critical Error: Asset files not found. {e}")

# --- NAVIGATION ADDED ---
st.sidebar.title("🎮 Navigation")
page = st.sidebar.radio("Select Engine:", ["🛡️ Supervised Model (Base)", "🧠 Anomaly Detection (Genius)"])

# --- 5. INPUT SECTION (Sidebar) ---
st.sidebar.header("Transaction Parameters")
st.sidebar.info("Adjust the top predictors to simulate transaction risk:")

# Top 6 Features + Amount based on your SHAP analysis
v14 = st.sidebar.slider("V14 (Primary Indicator)", -20.0, 20.0, 0.0, help="Highest impact. Low values = High Risk.")
v4  = st.sidebar.slider("V4 (Anomaly Index)", -20.0, 20.0, 0.0, help="High values = High Risk.")
v17 = st.sidebar.slider("V17 (Secondary Impact)", -20.0, 20.0, 0.0)
v10 = st.sidebar.slider("V10 (Behavioral Pattern)", -20.0, 20.0, 0.0)
v12 = st.sidebar.slider("V12 (Identity Component)", -20.0, 20.0, 0.0)
v11 = st.sidebar.slider("V11 (Security Factor)", -20.0, 20.0, 0.0)
amount = st.sidebar.number_input("Transaction Amount ($)", min_value=0.0, value=50.0)

# Initialize session state for the audit log
if 'audit_log' not in st.session_state:
    st.session_state.audit_log = pd.DataFrame()

# --- COMMON PRE-PROCESSING ---
features = np.zeros((1, 30))
features[0, 13] = v14 # V14 index
features[0, 3]  = v4  # V4 index
features[0, 16] = v17 # V17 index
features[0, 9]  = v10 # V10 index
features[0, 11] = v12 # V12 index
features[0, 10] = v11 # V11 index

# Robust Scaling for 'Amount' (Index 29)
amount_array = np.array([[amount]])
features[0, 29] = scaler.transform(amount_array)[0, 0]

# =========================================================
# PAGE 1 : SUPERVISED MODEL (XGBOOST - INTACT)
# =========================================================
if page == "🛡️ Supervised Model (Base)":
    st.title("🛡️ FraudX AI Production Dashboard")
    st.markdown("""
    This system performs **Real-Time Fraud Detection** using a tuned **XGBoost** classifier. 
    Inputs are based on the top influencing features identified through your **SHAP analysis** and 
    a **GENIUS** part for predict any type on anomaly based on an isolation Forest and the cluster.
    ---
    """)

    with st.expander("ℹ️ Understanding Behavioral Components & Feature Selection"):
        st.write("""
        **Why these specific variables?**
        Based on the **SHAP (SHapley Additive exPlanations)** global importance plot, variables like **V14, V17, and V12** are the primary drivers 
        of the model's decisions. By exposing these 6 components, we capture approximately **85% of the model's predictive logic**.
        """)
    
    if st.sidebar.button("Run AI Analysis"):
        with st.spinner('Processing multi-dimensional scoring...'):
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0][1]
            
            full_record = {f"V{i+1}": features[0, i] for i in range(28)}
            full_record["Time"] = 0.0 
            full_record["Amount"] = amount
            full_record["Class"] = 1 if prediction == 1 else 0
            full_record["Status"] = "FRAUD" if prediction == 1 else "LEGIT"
            full_record["Risk_Score"] = f"{probability:.2%}"
            full_record["Timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

            new_entry_df = pd.DataFrame([full_record])

            # AUTOMATIC SILENT SAVE
            auto_save_file = 'fraud_retraining_data.csv'
            if not os.path.isfile(auto_save_file):
                new_entry_df.to_csv(auto_save_file, index=False)
            else:
                new_entry_df.to_csv(auto_save_file, mode='a', header=False, index=False)

            st.session_state.audit_log = pd.concat([new_entry_df, st.session_state.audit_log], ignore_index=True)

        st.subheader("Analysis Results")
        res_col1, res_col2 = st.columns(2)

        with res_col1:
            if prediction == 1:
                st.error("🚨 ALERT: FRAUDULENT TRANSACTION DETECTED")
                st.metric("Risk Score", f"{probability:.2%}", delta="High Risk", delta_color="inverse")
            else:
                st.success("✅ TRANSACTION VALIDATED (LEGITIMATE)")
                st.metric("Risk Score", f"{probability:.2%}", delta="Safe", delta_color="normal")

        with res_col2:
            st.write("**Model Intelligence:**")
            st.info(f"The XGBoost model analyzed 30 dimensions. Primary drivers for this score were V14, V4, and V17. Simulated amount: ${amount:,.2f}.")

# --- NEW: EXPLAINABILITY SECTION (SHAP & LIME) ---
        st.divider()
        st.subheader("🧪 Explainability Suite (XAI)")
        
        tab1, tab2 = st.tabs(["📊 SHAP Explanation", "📉 LIME Local Diagnostic"])
        
        with tab1:
            st.write("**Local Contribution (SHAP Force Plot)**")
            # Explainer needs to be initialized (ideally with a sample of your training data)
            # For the demo, we use the model's base values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(features)
            
            # Generating the Force Plot
            fig_shap, ax_shap = plt.subplots(figsize=(10, 2))
            shap.force_plot(
                explainer.expected_value, 
                shap_values[0], 
                features, 
                feature_names=[f"V{i+1}" for i in range(28)] + ["Time", "Amount"],
                matplotlib=True,
                show=False
            )
            st.pyplot(plt.gcf())
            st.info("💡 Red bars push the risk HIGHER, blue bars push it LOWER.")

        with tab2:
            st.write("**Feature Importance (LIME)**")
            # For LIME, we simulate the training background if not loaded
            # You can also use a pre-saved lime_explainer.pkl if you have one
            explainer_lime = lime.lime_tabular.LimeTabularExplainer(
                training_data=np.random.normal(0, 1, (100, 30)), # Replace with a sample of X_train for better accuracy
                feature_names=[f"V{i+1}" for i in range(28)] + ["Time", "Amount"],
                class_names=['Legit', 'Fraud'],
                mode='classification'
            )
            exp = explainer_lime.explain_instance(features[0], model.predict_proba, num_features=6)
            
            # Displaying LIME as a plot
            fig_lime = exp.as_pyplot_figure()
            st.pyplot(fig_lime)

# =========================================================
# PAGE 2 : GENIUS ENGINE (ISOLATION FOREST + SMART DIAGNOSTIC)
# =========================================================
else:
    st.title("🧠 FraudX - Genius Anomaly Engine")
    st.info("Detection based on statistical outliers with a strict sensitivity threshold of 0.16.")

    if st.sidebar.button("Activate Genius Detection"):
        # Calculate Decision Score
        score_brut = iso_forest.decision_function(features)[0]
        
        # Apply strict threshold 0.16
        if score_brut < 0.16:
            is_anomaly = -1
        else:
            is_anomaly = 1

        col1, col2 = st.columns(2)
        with col1:
            if is_anomaly == -1:
                st.warning("🔍 RESULT: STATISTICAL ANOMALY DETECTED")
                
                # --- SMART DIAGNOSTIC LOGIC ---
                if amount > 1000:
                    diagnosis = "Volume Outlier (Extreme Transaction Value)"
                elif abs(v12) > 10 or abs(v10) > 10:
                    diagnosis = "Identity Theft Pattern (Behavioral Breach)"
                elif v4 > 10 or v11 > 10:
                    diagnosis = "High Anomaly Index (System Security Alert)"
                else:
                    diagnosis = "Structural Risk (Atypical Component Deviation)"
                
                st.error(f"**Anomaly Category: {diagnosis}**")
            else:
                st.success("✨ RESULT: PATTERN IS STATISTICALLY NORMAL")
            
            st.metric("Anomaly Score", f"{score_brut:.4f}", help="Score < 0.16 triggers alert")

        with col2:
            st.subheader("Behavioral Fingerprint")
            categories = ['V14', 'V4', 'V17', 'V10', 'V12', 'V11']
            values = [v14, v4, v17, v10, v12, v11]
            fig = go.Figure(data=go.Scatterpolar(r=values, theta=categories, fill='toself'))
            st.plotly_chart(fig, use_container_width=True)

# --- 8. AUDIT LOG & EXPORT ---
st.divider()
st.subheader("📜 System Audit Log")
if not st.session_state.audit_log.empty:
    display_df = st.session_state.audit_log[["Timestamp", "Amount", "Risk_Score", "Status"]]
    st.dataframe(display_df, use_container_width=True)
