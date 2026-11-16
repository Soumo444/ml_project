import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import app as flask_app


VALID_FORM = {
    "gender": "female",
    "race_ethnicity": "group B",
    "parental_level_of_education": "bachelor's degree",
    "lunch": "standard",
    "test_preparation_course": "completed",
    "reading_score": "72",
    "writing_score": "74",
}


class FlaskAppTests(unittest.TestCase):
    def setUp(self) -> None:
        flask_app.app.config["TESTING"] = True
        self.client = flask_app.app.test_client()

    def test_home_page_displays_prediction_form(self) -> None:
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Math Score Predictor", response.data)
        self.assertIn(b"reading_score", response.data)

    def test_valid_form_displays_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "model.joblib"
            model_path.touch()
            with (
                patch("app.MODEL_PATH", model_path),
                patch("app.PredictionPipeline.predict") as predict,
            ):
                predict.return_value = pd.DataFrame(
                    {"predicted_math_score": [81.5]}
                )
                response = self.client.post("/", data=VALID_FORM)

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"81.50", response.data)
        predict.assert_called_once()

    def test_score_outside_valid_range_is_rejected(self) -> None:
        invalid_form = {**VALID_FORM, "reading_score": "101"}

        response = self.client.post("/", data=invalid_form)

        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Reading score must be from 0 to 100", response.data)

    def test_missing_model_shows_training_instructions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing_model = Path(temporary_directory) / "missing-model.joblib"
            with patch("app.MODEL_PATH", missing_model):
                response = self.client.post("/", data=VALID_FORM)

        self.assertEqual(response.status_code, 503)
        self.assertIn(b"Train the model first", response.data)


if __name__ == "__main__":
    unittest.main()
