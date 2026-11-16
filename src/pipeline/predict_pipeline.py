"""Load a trained model and predict math scores for new student records."""

from pathlib import Path

import pandas as pd

from src.exception import PredictionError, ProjectError
from src.logger import get_logger
from src.utils import load_object


logger = get_logger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class PredictionPipeline:
    """Predict math scores with a model produced by the training pipeline."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        self.model_path = (
            Path(model_path)
            if model_path is not None
            else PROJECT_ROOT / "artifacts" / "model.joblib"
        )

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of the input rows with predicted math scores."""
        if not isinstance(features, pd.DataFrame):
            raise PredictionError("features must be a pandas DataFrame.")
        if features.empty:
            raise PredictionError("At least one student row is required for prediction.")

        try:
            model = load_object(self.model_path)
        except ProjectError as error:
            raise PredictionError(
                f"Could not load the trained model from '{self.model_path}'."
            ) from error

        model_features = getattr(model, "feature_names_in_", None)
        if model_features is None:
            raise PredictionError("The saved model does not expose its expected feature names.")

        expected_columns = list(model_features)
        missing_columns = set(expected_columns) - set(features.columns)
        if missing_columns:
            names = ", ".join(sorted(missing_columns))
            raise PredictionError(f"Prediction data is missing required features: {names}")

        try:
            predictions = model.predict(features.loc[:, expected_columns])
        except (TypeError, ValueError, RuntimeError) as error:
            logger.exception("Could not generate math-score predictions.")
            raise PredictionError("Could not generate math-score predictions.") from error

        result = features.copy()
        result["predicted_math_score"] = predictions
        return result
