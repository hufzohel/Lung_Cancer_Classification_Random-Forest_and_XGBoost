import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import platform

import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV

from common import load_processed_data, save_json
from common import ROOT, SEED, FOLDS, JOBS

SCORING = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "f1_macro": "f1_macro",
    "roc_auc": "roc_auc",
    "recall_no": make_scorer(recall_score, pos_label=0, zero_division=0),
    "recall_yes": make_scorer(recall_score, pos_label=1, zero_division=0),
}
PARAMETERS = {
    "max_depth": [4, 8, None],
    "min_samples_leaf": [1, 2, 4],
    "class_weight": [None, "balanced"],
}
OUTPUT_DIR = ROOT / "results/random_forest"
TRAIN_PATH = ROOT / "data/processed/train.csv"


def evaluate_baselines(
        X: pd.DataFrame,
        y: pd.Series,
        folds,
        seed: int,
        jobs: int,
        output: Path
) -> tuple[RandomForestClassifier, float]:
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=seed,
        n_jobs=1
    )
    models = {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "rf_baseline": rf
    }
    summaries = []

    for name, model in models.items():
        scores = cross_validate(
            model, X, y, cv=folds, scoring=SCORING,
            return_train_score=True, n_jobs=jobs, error_score="raise",
        )
        table = pd.DataFrame(scores)
        table.insert(0, "fold", np.arange(1, len(folds) + 1))
        table.to_csv(output / f"{name}_cv.csv", index=False)
        row: dict = {"model": name}
        for metric in SCORING:
            row[f"mean_{metric}"] = float(np.mean(scores[f"test_{metric}"]))
            row[f"std_{metric}"] = float(np.std(scores[f"test_{metric}"], ddof=0))
            row[f"mean_train_{metric}"] = float(np.mean(scores[f"train_{metric}"]))
        summaries.append(row)
        print(f"{name}: CV balanced accuracy = {row['mean_balanced_accuracy']:.4f}")

    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "baseline_summary.csv", index=False)
    return rf, summaries[1]["mean_balanced_accuracy"]


def tune_random_forest(
        X: pd.DataFrame,
        y:pd.Series,
        folds,
        seed: int,
        jobs: int
) -> GridSearchCV:
    search = GridSearchCV(
        estimator=RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1),
        param_grid=PARAMETERS,
        scoring=SCORING, refit="balanced_accuracy", cv=folds,
        n_jobs=jobs, return_train_score=True,
        error_score="raise", verbose=1,
    )
    search.fit(X, y)
    return search

def save_training_outputs(
        model,
        search,
        metadata,
        output
):
    pd.DataFrame(search.cv_results_).to_csv(output / "search_results.csv", index=False)
    save_json(metadata, output / "training_metadata.json")
    joblib.dump({
        "model": model,
        "feature_columns": metadata["feature_columns"],
        "target_mapping": {"NO": 0, "YES": 1},
        "metadata": metadata,
    }, output / "rf_model.joblib")

def main() -> None:
    """
    1. Load the processed training data and split it into X and y.
    2. Create the cross-validation folds once (StratifiedKFold) and reuse every model.
    3. Evaluate the baselines (DummyClassifier and default RandomForestClassifier) with cross-validation; save per-fold scores and a summary to CSV.
    4. Tune RandomForestClassifier with GridSearchCV on the same folds; the best
       configuration is refit on the full training set.
    5. Select the final model by comparing the baseline's CV score with the tuned
       model's best CV score; if the baseline wins or ties, refit it on the full training set.
    6. Save the outputs: rf_model.joblib (model, feature columns, metadata),
       search_results.csv, and training_metadata.json.
    :return: None
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_processed_data(TRAIN_PATH)
    cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    folds = list(cv.split(X, y))

    baseline, baseline_score = evaluate_baselines(X, y, folds, SEED, JOBS, OUTPUT_DIR)
    tuned_model = tune_random_forest(X, y, folds, SEED, JOBS)

    # Comparing and selecting the final model
    if tuned_model.best_score_ > baseline_score:
        final_model = tuned_model.best_estimator_
        selected_model_name, selected_model_score = "tuned_random_forest", tuned_model.best_score_
    else:
        final_model = clone(baseline).fit(X, y)
        selected_model_name, selected_model_score = "baseline", baseline_score

    metadata = {
        "n_train": len(y), "feature_columns": X.columns.tolist(),
        "seed": SEED, "n_splits": FOLDS,
        "selection_metric": "balanced_accuracy",
        "selected_model": selected_model_name, "selected_cv_score": selected_model_score,
        "baseline_cv_score": baseline_score,
        "selected_params": {key: final_model.get_params()[key] for key in
                            ["n_estimators", "max_depth", "min_samples_leaf",
                             "class_weight", "max_features", "min_samples_split"]},
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__},
    }
    save_training_outputs(final_model, tuned_model, metadata, OUTPUT_DIR)
    print(metadata)
    print(f"Selected: {selected_model_name}; CV balanced accuracy = {selected_model_score:.4f}")
    print(f"Refit on {len(y)} rows. Saved: {OUTPUT_DIR / 'rf_model.joblib'}")


if __name__ == "__main__":
    main()