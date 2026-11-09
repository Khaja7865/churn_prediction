
import os
import sys

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml


def configure_mlflow(config):
    tracking_uri = (
        os.getenv("MLFLOW_TRACKING_URI")
        or config["mlflow"].get("tracking_uri")
    )

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)


def clean_dataframe(df, config):
    df = df.copy()

    target_column = config["data"]["target_column"]
    id_column = config["data"]["id_column"]

    if id_column in df.columns:
        df = df.drop(columns=[id_column])

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"], errors="coerce"
        )

    if target_column in df.columns:
        df = df.drop(columns=[target_column])

    return df


def run_inference_demo():

    print("[INFO] Starting Production Inference...", flush=True)

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    configure_mlflow(config)

    model_name = config["registry"]["model_name"]
    model_uri = f"models:/{model_name}@Production"

    try:
        print(f"[INFO] Loading model: {model_uri}", flush=True)

        model = mlflow.sklearn.load_model(model_uri)

        print("[SUCCESS] Production model loaded.", flush=True)

    except Exception as exc:
        print(f"[ERROR] Could not load Production model: {exc}", flush=True)
        print("[ERROR] Run 08_automate_lifecycle.py first.", flush=True)
        return False

    # Load preprocessor and dataset
    preprocessor_path = config["paths"]["preprocessor"]
    raw_data_path = config["data"]["raw_path"]

    if not os.path.exists(preprocessor_path):
        print(f"[ERROR] Preprocessor not found: {preprocessor_path}", flush=True)
        return False

    if not os.path.exists(raw_data_path):
        print(f"[ERROR] Dataset not found: {raw_data_path}", flush=True)
        return False

    preprocessor = joblib.load(preprocessor_path)
    df = pd.read_csv(raw_data_path)

    if df.empty:
        print("[ERROR] Dataset is empty.", flush=True)
        return False

    id_column = config["data"]["id_column"]

    if id_column not in df.columns:
        print(f"[ERROR] Customer ID column '{id_column}' not found.", flush=True)
        return False

    # Select customers
    sample = df.sample(
        n=min(5, len(df)),
        random_state=7
    )

    customer_ids = sample[id_column].tolist()

    X_sample = clean_dataframe(sample, config)
    X_transformed = preprocessor.transform(X_sample)

    # Generate predictions and probabilities
    predictions = model.predict(X_transformed)

    if not hasattr(model, "predict_proba"):
        print("[ERROR] Model does not support predict_proba().", flush=True)
        return False

    all_probabilities = model.predict_proba(X_transformed)
    classes = list(model.classes_)

    # Identify the churn-positive class
    yes_indices = [
        i for i, cls in enumerate(classes)
        if str(cls).strip().lower() in
        {"yes", "1", "true", "churn", "churned"}
    ]

    if yes_indices:
        positive_index = yes_indices[0]
    elif len(classes) == 2:
        positive_index = 1
    else:
        print("[ERROR] Could not identify the churn-positive class.", flush=True)
        return False

    probabilities = all_probabilities[:, positive_index]

    # Save all predictions
    result = sample.copy()
    result["Predicted_Churn"] = predictions
    result["Churn_Probability"] = np.round(probabilities, 4)

    os.makedirs("outputs", exist_ok=True)

    output_path = config["paths"]["inference_output"]
    result.to_csv(output_path, index=False)

    # Display first customer's actual prediction
    first_prediction = predictions[0]
    first_probability = float(probabilities[0]) * 100

    prediction_text = (
        "YES" if str(first_prediction).strip().lower()
        in {"yes", "1", "true", "churn", "churned"}
        else "NO"
    )

    print("\n========================================", flush=True)
    print("       CHURN PREDICTION OUTPUT", flush=True)
    print("========================================", flush=True)

    print(f"\nCustomer ID       : {customer_ids[0]}", flush=True)
    print(f"Churn Prediction  : {prediction_text}", flush=True)
    print(f"Churn Probability : {first_probability:.1f}%", flush=True)

    print("\n----------------------------------------", flush=True)
    print("ADDITIONAL CUSTOMER PREDICTIONS", flush=True)
    print("----------------------------------------", flush=True)

    print(
        result[
            [id_column, "Predicted_Churn", "Churn_Probability"]
        ].to_string(index=False),
        flush=True
    )

    print("----------------------------------------", flush=True)

    print(f"\n[SUCCESS] Predictions saved to {output_path}", flush=True)

    return True


if __name__ == "__main__":
    passed = run_inference_demo()

    sys.exit(0 if passed else 1)
