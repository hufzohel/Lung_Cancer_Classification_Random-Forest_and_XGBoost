from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, recall_score, roc_auc_score,
)

from common import ROOT, load_processed_data, save_json

MODEL_PATH = ROOT / "results/random_forest/rf_model.joblib"
DATA_PATH = ROOT / "data/processed/test.csv"
OUTPUT_DIR = ROOT / "results/random_forest/holdout"


def load_test_inputs(
        model_path: Path,
        data_path: Path
):
    bundle = joblib.load(model_path)
    X, y = load_processed_data(data_path)
    features = bundle["feature_columns"]
    if set(features) != set(X.columns):
        raise ValueError("Test's feature columns don't match the model's.")
    return bundle, X.loc[:, features], y

def evaluate_model(
        model,
        X: pd.DataFrame,
        y: pd.Series
):
    prediction = model.predict(X)
    positive_column = list(model.classes_).index(1)
    probability = model.predict_proba(X)[:, positive_column]
    metrics = {
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "f1_macro": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "recall_no": float(recall_score(y, prediction, pos_label=0, zero_division=0)),
        "recall_yes": float(recall_score(y, prediction, pos_label=1, zero_division=0)),
    }
    report = classification_report(y, prediction, labels=[0, 1], target_names=["NO", "YES"],
                                   output_dict=True, zero_division=0)
    matrix = confusion_matrix(y, prediction, labels=[0, 1])
    predictions = pd.DataFrame({
        "row_id": np.arange(len(y)), "y_true": y.to_numpy(),
        "y_pred": prediction, "probability_yes": probability,
    })
    return metrics, report, matrix, predictions

def save_test_outputs(metrics, report, matrix, predictions, output):
    output.mkdir(parents=True, exist_ok=True)
    save_json(metrics, output / "test_metrics.json")
    save_json(report, output / "classification_report.json")
    pd.DataFrame(matrix, index=["actual_NO", "actual_YES"],
                 columns=["predicted_NO", "predicted_YES"]).to_csv(output / "confusion_matrix.csv")
    predictions.to_csv(output / "test_predictions.csv", index=False)

def main():
    """
    1. Load the trained model and test data.
    2. Evaluate the model on the test data.
    3. Save the evaluation metrics, classification report, confusion matrix, and predictions to the specified output directory.
    4. Print the evaluation metrics and confusion matrix to the console.
    :return: None
    """
    bundle, X, y = load_test_inputs(MODEL_PATH, DATA_PATH)
    metrics, report, matrix, predictions = evaluate_model(bundle["model"], X, y)

    metrics.update({"n_test": len(y),
                    "selected_model": bundle["metadata"]["selected_model"]})
    save_test_outputs(metrics, report, matrix, predictions, OUTPUT_DIR)

    print(pd.Series({k: v for k, v in metrics.items() if isinstance(v, float)}).to_string())
    print("Confusion matrix (rows=actual, columns=predicted; order NO, YES):")
    print(matrix)
    print(f"Saved holdout results: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()