import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def run_preprocessing():
    print("Starting Preprocessing Pipeline...")

    # 1. Load the raw data
    data_path = "data/raw/churn.csv"
    df = pd.read_csv(data_path)

    # 2. Data Cleaning
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    ).fillna(0)

    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    # Convert Churn to binary
    df["Churn"] = (
        df["Churn"]
        .apply(lambda x: 1 if str(x).strip().lower() == "yes" else 0)
        .astype(int)
    )

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # 3. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 4. Identify column types
    cat_cols = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    num_cols = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    # 5. Create preprocessing components
    scaler = StandardScaler()

    ohe = OneHotEncoder(
        drop="first",
        sparse_output=False,
        handle_unknown="ignore"
    )

    # 6. Create the combined preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", scaler, num_cols),
            ("cat", ohe, cat_cols)
        ],
        remainder="drop"
    )

    # 7. Fit ONLY on training data
    X_train_final = preprocessor.fit_transform(X_train)
    X_test_final = preprocessor.transform(X_test)

    # Make sure outputs are NumPy arrays
    X_train_final = np.asarray(X_train_final)
    X_test_final = np.asarray(X_test_final)

    # 8. Create required directories
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # 9. Save processed data
    np.save(
        "data/processed/X_train_final.npy",
        X_train_final
    )

    np.save(
        "data/processed/X_test_final.npy",
        X_test_final
    )

    np.save(
        "data/processed/y_train.npy",
        y_train.to_numpy(dtype=np.int64)
    )

    np.save(
        "data/processed/y_test.npy",
        y_test.to_numpy(dtype=np.int64)
    )

    # 10. Save individual preprocessing artifacts
    joblib.dump(
        scaler,
        "models/scaler.pkl"
    )

    joblib.dump(
        ohe,
        "models/ohe.pkl"
    )

    # 11. Save the COMPLETE fitted preprocessing pipeline
    joblib.dump(
        preprocessor,
        "models/preprocessor.pkl"
    )

    # 12. Save metadata
    metadata = {
        "dataset_name": "Telco Customer Churn",
        "train_shape": list(X_train_final.shape),
        "test_shape": list(X_test_final.shape),
        "numerical_features": num_cols,
        "categorical_features": cat_cols,
        "preprocessor": "ColumnTransformer",
        "scaler": "StandardScaler",
        "encoder": "OneHotEncoder",
        "random_state": 42
    }

    with open(
        "data/processed/dataset_metadata.json",
        "w"
    ) as f:
        json.dump(metadata, f, indent=4)

    print("Preprocessing completed successfully!")
    print(f"Training shape: {X_train_final.shape}")
    print(f"Testing shape: {X_test_final.shape}")
    print("Saved:")
    print("  - models/scaler.pkl")
    print("  - models/ohe.pkl")
    print("  - models/preprocessor.pkl")


if __name__ == "__main__":
    run_preprocessing()