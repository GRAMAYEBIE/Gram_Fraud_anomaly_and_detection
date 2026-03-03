# 🛡️ FraudX AI: End-to-End Fraud Detection System

## 📌 Overview

### FraudX AI is a high-performance, production-ready fraud detection solution. Moving beyond simple classification, it implements a Hybrid Multi-Model Architecture to identify known fraud patterns while simultaneously detecting and profiling unseen anomalies.

### The system architecture follows a "Box-to-Box" approach, bridgings the gap between Real-Time Inference (Streamlit), Behavioral Diagnostics (Clustering), and Automated MLOps (Retraining Pipeline).



## 🔄 System Architecture & Data Flow

The system operates in a closed loop to ensure the model never becomes obsolete:

Production Interface (app.py): A web dashboard where users input transaction parameters.

#### Dual-Engine Prediction (app.py): * Layer 1 (XGBoost): Supervised classification for known fraud signatures.

#### Layer 2 (Isolation Forest): Unsupervised anomaly detection for "out-of-distribution" behaviors.

#### Strategic Profiling (K-Means): Anomalies are automatically segmented into risk categories (Whale, High-Value, or Behavioral).

#### Silent Logging: Every transaction analyzed is automatically archived in a temporary buffer file (fraud_retraining_data.csv) to feed the continuous learning cycle.

#### Retraining Pipeline (retrain_pipeline.py): A dedicated MLOps script that merges new production data with the historical baseline (creditcard.csv),updates models, and ensures zero-downtime deployment.

#### Automatic Cleanup: After a successful update, the system purges temporary files to maintain strict data hygiene and prevent redundancy.

## 🛠️ Key Features
Real-Time Scoring: Instant risk probability and classification combined with anomaly detection.

Behavioral DNA Mapping: Integrated Radar Charts (Plotly) that provide a visual "fingerprint" of the transaction's risk profile for instant forensic analysis.

Explainable AI (XAI): Feature selection guided by SHAP (SHapley Additive exPlanations) to ensure the model focuses on the most predictive behavioral patterns.

Anomaly Priority Index: Automatic categorization of threats:

🔴 Critical Whale Anomaly: High financial impact + extreme technical deviation.

🟠 High-Value Outlier: Significant amount requiring manual audit.

🟡 Standard Behavioral Anomaly: Technical inconsistency with moderate risk.

MLOps Ready: Automated retraining script that manages data merging and model deployment. / Full automation of model maintenance and data hygiene.

Robust Preprocessing: Integrated RobustScaler to handle high-variance transaction amounts effectively.

## 🚀 Getting Started

### Prerequisites

- Python 3.9+

- Required Libraries: streamlit, xgboost, pandas, joblib, scikit-learn 


- Data: Place the original creditcard.csv in the root directory.

### Installation & Usage

- pip install -r requiremnts.txt

Launch the Dashboard:

### Bash

- streamlit run app.py

#### Simulate Transactions: Use the sidebar sliders to input data and click "Run AI Analysis".

### The monitoring dashoborad 

- stremalit run monitoring_app.py

#### Trigger Retraining: Once new data is collected, update the model by running:

### Bash

- python retrain_pipeline.py

### 📊 Model Performance
The core engine is an XGBoost Classifier optimized for imbalanced datasets. It focuses on maximizing Precision-Recall AUC, ensuring that legitimate customers are not inconvenienced while high-risk frauds are caught.

## 📝 Conclusion
FraudX AI demonstrates a modern approach to financial security by combining real-time interactivity with automated backend maintenance. This ensures the system remains resilient against emerging fraud tactics.
