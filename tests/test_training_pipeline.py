import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.exception import DataIngestionError, PredictionError
from src.pipeline.predict_pipeline import PredictionPipeline
from src.pipeline.train_pipeline import run_training_pipeline


def make_student_data(row_count: int = 60) -> pd.DataFrame:
    random = np.random.default_rng(42)
    rows = []
    for index in range(row_count):
        reading_score = int(random.integers(35, 101))
        writing_score = int(random.integers(35, 101))
        preparation = "completed" if index % 2 else "none"
        math_score = int(
            np.clip(
                0.35 * reading_score
                + 0.35 * writing_score
                + (5 if preparation == "completed" else 0)
                + random.normal(0, 8),
                0,
                100,
            )
        )
        rows.append(
            {
                "gender": "female" if index % 2 else "male",
                "race_ethnicity": f"group {chr(65 + index % 5)}",
                "parental_level_of_education": [
                    "some college",
                    "bachelor's degree",
                    "high school",
                ][index % 3],
                "lunch": "standard" if index % 3 else "free/reduced",
                "test_preparation_course": preparation,
                "math_score": math_score,
                "reading_score": reading_score,
                "writing_score": writing_score,
            }
        )
    return pd.DataFrame(rows)


class TrainingPipelineTests(unittest.TestCase):
    def test_ingestion_rejects_missing_required_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_path = root / "invalid.csv"
            pd.DataFrame({"math_score": [70]}).to_csv(source_path, index=False)
            ingestion = DataIngestion(
                DataIngestionConfig(
                    source_path=source_path,
                    output_dir=root / "artifacts",
                )
            )

            with self.assertRaises(DataIngestionError):
                ingestion.initiate_data_ingestion()

    def test_training_saves_model_metrics_and_supports_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_path = root / "students.csv"
            data = make_student_data()
            data.to_csv(source_path, index=False)

            training = run_training_pipeline(
                source_path=source_path,
                artifact_dir=root / "artifacts",
                cv_folds=3,
            )

            self.assertTrue(training.model_path.is_file())
            self.assertTrue(training.metrics_path.is_file())
            self.assertEqual(training.metrics["target_column"], "math_score")
            self.assertEqual(training.metrics["train_rows"], 48)
            self.assertEqual(training.metrics["test_rows"], 12)
            self.assertEqual(
                set(training.metrics["cv_results"]),
                {
                    "Linear regression",
                    "Ridge",
                    "Decision tree",
                    "Random forest",
                    "Extra trees",
                    "K-nearest neighbors",
                    "Support vector regression",
                    "XGBoost",
                    "Mean baseline",
                },
            )
            self.assertTrue(
                all(
                    "best_params" in result
                    for result in training.metrics["cv_results"].values()
                )
            )

            predictions = PredictionPipeline(training.model_path).predict(
                data.drop(columns="math_score").iloc[:3]
            )
            self.assertEqual(len(predictions), 3)
            self.assertIn("predicted_math_score", predictions.columns)
            self.assertTrue(predictions["predicted_math_score"].notna().all())
            incomplete_features = data.drop(columns="math_score").iloc[:1].drop(
                columns="gender"
            )
            with self.assertRaises(PredictionError):
                PredictionPipeline(training.model_path).predict(incomplete_features)


if __name__ == "__main__":
    unittest.main()
