from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
FOLDS = 5
JOBS = 2

def load_processed_data(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    file_path: Path = Path(path)
    data: pd.DataFrame = pd.read_csv(file_path)
    X: pd.DataFrame = data.drop(columns=["LUNG_CANCER"])
    y: pd.Series = data["LUNG_CANCER"]

    return X, y

def save_json(value: dict, path: str | Path) -> None:
    file_path: Path = Path(path)

    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

