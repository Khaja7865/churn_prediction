import json
import os

import mlflow
from mlflow.tracking import MlflowClient

import yaml


def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def automate_model_lifecycle():
    config = load_config()

    print("[INFO] Starting Automated Model Lifecycle...")

    # ---------------------------------------------------------
    # MLflow configuration
    # ---------------------------------------------------------
    tracking_uri = config["mlflow"]["tracking_uri"]
    model_name = config["registry"]["model_name"]

    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()

    # ---------------------------------------------------------
    # Check registered model
    # ---------------------------------------------------------
    try:
        versions = client.search_model_versions(
            f"name='{model_name}'"
        )
    except Exception as e:
        print(
            f"[ERROR] Unable to retrieve registered "
            f"model versions: {e}"
        )
        return False

    if not versions:
        print(
            f"[ERROR] No registered versions found "
            f"for model: {model_name}"
        )
        return False

    # ---------------------------------------------------------
    # Find latest model version
    # ---------------------------------------------------------
    latest_version = max(
        versions,
        key=lambda version: int(version.version)
    )

    version_number = latest_version.version

    print(
        f"[INFO] Latest registered model version: "
        f"{version_number}"
    )

    # ---------------------------------------------------------
    # MLflow versions may use aliases instead of stages.
    # We use the 'Production' alias when possible.
    # ---------------------------------------------------------
    production_alias = "Production"

    try:
        client.set_registered_model_alias(
            model_name,
            production_alias,
            version_number
        )

        print(
            f"[SUCCESS] Model version {version_number} "
            f"assigned to alias '{production_alias}'."
        )

    except Exception as e:
        print(
            f"[WARNING] Could not assign MLflow alias: {e}"
        )

    # ---------------------------------------------------------
    # Create lifecycle report
    # ---------------------------------------------------------
    report = {
        "model_name": model_name,
        "latest_version": str(version_number),
        "production_alias": production_alias,
        "status": "SUCCESS"
    }

    os.makedirs(
        "artifacts",
        exist_ok=True
    )

    report_path = (
        "artifacts/"
        "model_lifecycle_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            report,
            f,
            indent=2
        )

    print(
        f"[INFO] Lifecycle report saved: "
        f"{report_path}"
    )

    print(
        "[SUCCESS] Automated Model Lifecycle completed."
    )

    return True


if __name__ == "__main__":

    try:
        passed = automate_model_lifecycle()

        if passed:
            raise SystemExit(0)

        raise SystemExit(1)

    except Exception as e:
        print(f"[ERROR] {e}")
        raise