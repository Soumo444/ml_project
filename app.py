"""Flask web application for predicting student math scores."""

from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request

from src.exception import PredictionError
from src.logger import get_logger
from src.pipeline.predict_pipeline import PredictionPipeline


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "artifacts" / "model.joblib"
logger = get_logger(__name__)

CATEGORICAL_OPTIONS = {
    "gender": ("female", "male"),
    "race_ethnicity": tuple(f"group {letter}" for letter in "ABCDE"),
    "parental_level_of_education": (
        "some high school",
        "high school",
        "some college",
        "associate's degree",
        "bachelor's degree",
        "master's degree",
    ),
    "lunch": ("standard", "free/reduced"),
    "test_preparation_course": ("none", "completed"),
}
SCORE_FIELDS = ("reading_score", "writing_score")
FIELD_LABELS = {
    "gender": "Gender",
    "race_ethnicity": "Race/ethnicity group",
    "parental_level_of_education": "Parent/guardian education",
    "lunch": "Lunch type",
    "test_preparation_course": "Test preparation course",
    "reading_score": "Reading score",
    "writing_score": "Writing score",
}


def create_app() -> Flask:
    """Create the Flask application."""
    app = Flask(__name__)

    @app.route("/", methods=["GET", "POST"])
    def index():
        form_fields = (*CATEGORICAL_OPTIONS, *SCORE_FIELDS)
        form_data = (
            {field: request.form.get(field, "").strip() for field in form_fields}
            if request.method == "POST"
            else {}
        )
        prediction = None
        error = None
        status_code = 200

        if request.method == "POST":
            error = _validate_form(form_data)
            if error is not None:
                status_code = 400
            elif not MODEL_PATH.is_file():
                logger.error("Trained model artifact is missing: %s", MODEL_PATH)
                error = (
                    "The trained model was not found. Train the model first by running "
                    "`python -m src.pipeline.train_pipeline`."
                )
                status_code = 503
            else:
                try:
                    features = pd.DataFrame(
                        [
                            {
                                **{
                                    field: form_data[field]
                                    for field in CATEGORICAL_OPTIONS
                                },
                                **{
                                    field: int(form_data[field])
                                    for field in SCORE_FIELDS
                                },
                            }
                        ]
                    )
                    result = PredictionPipeline(MODEL_PATH).predict(features)
                    prediction = float(result.loc[0, "predicted_math_score"])
                except PredictionError as exception:
                    logger.exception("Could not process a math-score prediction.")
                    error = f"Prediction failed: {exception}"
                    status_code = 503

        return (
            render_template(
                "index.html",
                categorical_options=CATEGORICAL_OPTIONS,
                field_labels=FIELD_LABELS,
                form_data=form_data,
                prediction=prediction,
                error=error,
            ),
            status_code,
        )

    return app


def _validate_form(form_data: dict[str, str]) -> str | None:
    """Return a user-facing validation message, or None for a valid form."""
    for field in (*CATEGORICAL_OPTIONS, *SCORE_FIELDS):
        value = form_data[field]
        if not value:
            return f"{FIELD_LABELS[field]} is required."

    for field, options in CATEGORICAL_OPTIONS.items():
        if form_data[field] not in options:
            return f"Please choose a valid option for {FIELD_LABELS[field]}."

    for field in SCORE_FIELDS:
        try:
            score = int(form_data[field])
        except ValueError:
            return f"{FIELD_LABELS[field]} must be a whole number from 0 to 100."
        if not 0 <= score <= 100:
            return f"{FIELD_LABELS[field]} must be from 0 to 100."

    return None


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
