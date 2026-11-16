"""Tune candidate regressors, evaluate the winner, and save it."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from xgboost.core import XGBoostError

from src.components.data_transformation import DataTransformationArtifact
from src.exception import ModelTrainingError
from src.logger import get_logger
from src.utils import ensure_directory, save_json, save_object


logger = get_logger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ModelSearchSpec = tuple[BaseEstimator, dict[str, list[Any]]]


@dataclass(frozen=True)
class ModelTrainerConfig:
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "artifacts")
    random_state: int = 42
    cv_folds: int = 5


@dataclass(frozen=True)
class ModelTrainerArtifact:
    model_path: Path
    metrics_path: Path
    best_model_name: str
    metrics: dict[str, Any]


class ModelTrainer:
    """Tune multiple regressors and select the best by cross-validation MAE."""

    def __init__(self, config: ModelTrainerConfig | None = None) -> None:
        self.config = config or ModelTrainerConfig()

    def _model_search_specs(self) -> dict[str, ModelSearchSpec]:
        random_state = self.config.random_state
        return {
            "Linear regression": (LinearRegression(), {}),
            "Ridge": (
                Ridge(),
                {"model__alpha": [0.1, 1.0, 10.0, 30.0, 100.0]},
            ),
            "Decision tree": (
                DecisionTreeRegressor(random_state=random_state),
                {
                    "model__max_depth": [None, 5, 10],
                    "model__min_samples_leaf": [1, 5],
                },
            ),
            "Random forest": (
                RandomForestRegressor(random_state=random_state, n_jobs=1),
                {
                    "model__n_estimators": [100, 250],
                    "model__max_depth": [None, 10],
                    "model__min_samples_leaf": [1, 5],
                },
            ),
            "Extra trees": (
                ExtraTreesRegressor(random_state=random_state, n_jobs=1),
                {
                    "model__n_estimators": [100, 250],
                    "model__max_depth": [None, 10],
                    "model__min_samples_leaf": [1, 5],
                },
            ),
            "K-nearest neighbors": (
                KNeighborsRegressor(),
                {
                    "model__n_neighbors": [5, 10, 20],
                    "model__weights": ["uniform", "distance"],
                },
            ),
            "Support vector regression": (
                SVR(),
                {
                    "model__C": [0.1, 1.0, 10.0],
                    "model__epsilon": [0.1, 1.0, 5.0],
                },
            ),
            "XGBoost": (
                XGBRegressor(
                    objective="reg:squarederror",
                    random_state=random_state,
                    n_jobs=1,
                    verbosity=0,
                ),
                {
                    "model__n_estimators": [100, 250],
                    "model__max_depth": [1, 3],
                    "model__learning_rate": [0.03, 0.1],
                    "model__min_child_weight": [1, 10],
                },
            ),
            "Mean baseline": (DummyRegressor(strategy="mean"), {}),
        }

    def initiate_model_trainer(
        self,
        transformation_artifact: DataTransformationArtifact,
    ) -> ModelTrainerArtifact:
        if self.config.cv_folds < 2:
            raise ModelTrainingError("cv_folds must be at least 2.")
        if len(transformation_artifact.y_train) < self.config.cv_folds:
            raise ModelTrainingError(
                f"Training rows must be at least cv_folds ({self.config.cv_folds})."
            )

        cross_validation = KFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )
        searches: dict[str, GridSearchCV] = {}
        results: dict[str, dict[str, Any]] = {}

        current_model = "model initialization"
        try:
            for name, (estimator, parameter_grid) in self._model_search_specs().items():
                current_model = name
                pipeline = Pipeline(
                    steps=[
                        ("preprocessor", transformation_artifact.preprocessor),
                        ("model", estimator),
                    ]
                )
                search = GridSearchCV(
                    estimator=pipeline,
                    param_grid=parameter_grid,
                    scoring={
                        "mae": "neg_mean_absolute_error",
                        "mse": "neg_mean_squared_error",
                    },
                    refit="mae",
                    cv=cross_validation,
                    n_jobs=1,
                    error_score="raise",
                    return_train_score=False,
                )
                search.fit(
                    transformation_artifact.X_train,
                    transformation_artifact.y_train,
                )
                best_index = search.best_index_
                mean_test_mse = search.cv_results_["mean_test_mse"][best_index]
                results[name] = {
                    "cv_mae": float(-search.cv_results_["mean_test_mae"][best_index]),
                    "cv_rmse": float(np.sqrt(-mean_test_mse)),
                    "best_params": search.best_params_,
                }
                searches[name] = search
                logger.info(
                    "%s: CV MAE %.3f, CV RMSE %.3f, parameters %s",
                    name,
                    results[name]["cv_mae"],
                    results[name]["cv_rmse"],
                    results[name]["best_params"],
                )
        except (TypeError, ValueError, RuntimeError, XGBoostError) as error:
            logger.exception("Model search failed for %s.", current_model)
            raise ModelTrainingError(
                f"Model search failed for '{current_model}'."
            ) from error

        best_model_name = min(results, key=lambda name: results[name]["cv_mae"])
        best_search = searches[best_model_name]
        best_model = best_search.best_estimator_
        try:
            predictions = best_model.predict(transformation_artifact.X_test)
        except (TypeError, ValueError, RuntimeError) as error:
            logger.exception("Could not evaluate the selected model.")
            raise ModelTrainingError("Could not evaluate the selected model.") from error

        metrics: dict[str, Any] = {
            "target_column": "math_score",
            "selection_metric": "cross-validation MAE",
            "best_model": best_model_name,
            "best_params": best_search.best_params_,
            "cv_results": results,
            "test_mae": float(
                mean_absolute_error(transformation_artifact.y_test, predictions)
            ),
            "test_rmse": float(
                np.sqrt(mean_squared_error(transformation_artifact.y_test, predictions))
            ),
            "test_r2": float(r2_score(transformation_artifact.y_test, predictions)),
            "train_rows": int(len(transformation_artifact.y_train)),
            "test_rows": int(len(transformation_artifact.y_test)),
            "features": list(transformation_artifact.feature_columns),
        }

        output_dir = ensure_directory(self.config.output_dir)
        model_path = output_dir / "model.joblib"
        metrics_path = output_dir / "metrics.json"
        save_object(best_model, model_path)
        save_json(metrics, metrics_path)
        logger.info(
            "Selected %s with CV MAE %.3f; test MAE %.3f, RMSE %.3f, R2 %.3f.",
            best_model_name,
            results[best_model_name]["cv_mae"],
            metrics["test_mae"],
            metrics["test_rmse"],
            metrics["test_r2"],
        )
        return ModelTrainerArtifact(
            model_path=model_path,
            metrics_path=metrics_path,
            best_model_name=best_model_name,
            metrics=metrics,
        )
