"""
modelling.py
Training Baseline Model dengan MLflow Autolog (Kriteria 2 - Basic)
Dataset: Heart Disease
Nama Siswa: Maulana Haekal Noval Akbar
"""

import os
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

def load_data(data_dir: str):
    train_path = os.path.join(data_dir, "heart_train.csv")
    test_path = os.path.join(data_dir, "heart_test.csv")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    target_col = 'Heart_Disease_Status'
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    
    return X_train, X_test, y_train, y_test

def main():
    parser = argparse.ArgumentParser(description="Baseline Model Training with MLflow Autolog")
    parser.add_argument("--data_dir", default="heart_disease_preprocessing", help="Directory of preprocessed data")
    args = parser.parse_args()
    
    X_train, X_test, y_train, y_test = load_data(args.data_dir)
    
    # Jika dijalankan via mlflow run, run sudah otomatis dibuat oleh mlflow run
    # Hanya panggil set_experiment jika dijalankan langsung secara mandiri
    if not os.environ.get("MLFLOW_RUN_ID"):
        mlflow.set_experiment("Heart_Disease_Baseline")
    
    # Mengaktifkan MLflow autolog sesuai syarat Kriteria 2 Basic
    mlflow.sklearn.autolog()
    
    with mlflow.start_run(run_name="RandomForest_Autolog_Baseline") as run:
        print(f"[INFO] Started MLflow Run: {run.info.run_id}")
        
        # Training baseline model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluasi
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        print(f"[INFO] Baseline Model Trained for Maulana Haekal Noval Akbar.")
        print(f" - Accuracy: {acc:.4f}")
        print(f" - F1-Score: {f1:.4f}")
        print(f"[SUCCESS] MLflow Autolog successfully logged metrics, parameters, and model artifact.")

if __name__ == "__main__":
    main()
