import pandas as pd
import joblib
from xgboost import XGBClassifier
import os
import time

# --- PATH CONFIGURATION ---
ORIGINAL_DATA = 'creditcard.csv'
NEW_DATA = 'fraud_retraining_data.csv'
MODEL_PATH = 'fraudx_model_final.pkl'

def run_box_to_box_retraining():
    print("--- 🚀 Starting FraudX AI Retraining Pipeline ---")
    start_time = time.time()

    # 1. Check for the presence of new data
    if not os.path.exists(NEW_DATA):
        print(f"✅ System is already clean. No new data to process in '{NEW_DATA}'.")
        return

    try:
        # 2. Loading and cleaning new transactions
        print(f"📦 Loading newly collected data...")
        df_new = pd.read_csv(NEW_DATA)
        
        # Strict column selection to match the initial model
        features_cols = [f'V{i}' for i in range(1, 29)] + ['Amount']
        target_col = 'Class'
        required_cols = features_cols + [target_col]
        
        # Filter only necessary columns for the AI
        df_new_cleaned = df_new[required_cols]

        # 3. Merging with original Dataset (creditcard.csv)
        if os.path.exists(ORIGINAL_DATA):
            print(f"📖 Merging with historical dataset '{ORIGINAL_DATA}'...")
            df_orig = pd.read_csv(ORIGINAL_DATA)
            # Ensure the original dataset has the same column structure
            df_final = pd.concat([df_orig[required_cols], df_new_cleaned], ignore_index=True)
        else:
            print("⚠️ Warning: Original dataset not found. Retraining on new data only.")
            df_final = df_new_cleaned

        # 4. Retraining Phase (XGBoost)
        print(f"🧠 Training XGBoost model on {len(df_final)} samples...")
        
        X = df_final.drop(target_col, axis=1)
        y = df_final[target_col]

        # Robust configuration to prevent overfitting
        updated_model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        updated_model.fit(X, y)

        # 5. Saving and Deployment
        print(f"💾 Saving the updated model to '{MODEL_PATH}'...")
        joblib.dump(updated_model, MODEL_PATH)

        # 6. AUTOMATIC CLEANUP (Data Hygiene)
        print("🧹 Cleaning up temporary training data...")
        os.remove(NEW_DATA)
        print(f"✨ File '{NEW_DATA}' successfully deleted.")

        # 7. Final Statistics
        duration = round(time.time() - start_time, 2)
        print("-" * 50)
        print(f"✅ RETRAINING SUCCESSFUL in {duration} seconds!")
        print(f"📊 Total final dataset size: {len(df_final)} rows.")
        print(f"📈 The model is now up-to-date and ready for production.")
        print("-" * 50)

    except Exception as e:
        print(f"💥 A critical error occurred: {e}")

if __name__ == "__main__":
    run_box_to_box_retraining()