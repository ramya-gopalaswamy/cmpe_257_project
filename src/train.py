import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from models import get_model

logger = logging.getLogger(__name__)

N_FOLDS = 5
DATA_DIR = Path("data/normalized")


def load_fold_data(window: int, fold: int) -> tuple:
    fold_dir = DATA_DIR / f"window_{window}" / f"fold_{fold}"
    X_train = pd.read_csv(fold_dir / "X_train.csv")
    y_train = pd.read_csv(fold_dir / "y_train.csv").squeeze()
    X_test = pd.read_csv(fold_dir / "X_test.csv")
    y_test = pd.read_csv(fold_dir / "y_test.csv").squeeze()
    return X_train, y_train, X_test, y_test


def compute_metrics(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> dict:
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def run_cross_validation(
    model_config: dict,
    windows: list[int] = [2, 3, 4, 5],
    save_dir: Path | None = None,
) -> pd.DataFrame:
    results = []
    model_name = model_config["name"]
    model_params = model_config["params"]

    if save_dir is not None:
        models_dir = save_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

    for window in windows:
        for fold in range(N_FOLDS):
            logger.debug(f"Training {model_name} | {window=} | {fold=}")

            X_train, y_train, X_test, y_test = load_fold_data(window, fold)

            model = get_model(model_name, model_params)
            model.fit(X_train, y_train)

            if save_dir is not None:
                model_path = models_dir / f"window_{window}_fold_{fold}.pkl"
                joblib.dump(model, model_path)
                logger.debug(f"Saved model to {model_path}")

            y_pred = model.predict(X_test)
            metrics = compute_metrics(y_test, y_pred)

            results.append(
                {
                    "model": model_name,
                    "window": window,
                    "fold": fold,
                    **metrics,
                }
            )

    return pd.DataFrame(results)
