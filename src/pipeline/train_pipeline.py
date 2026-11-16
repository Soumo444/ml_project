"""Run the complete student-score model training workflow."""

from pathlib import Path

from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.data_transformation import (
    DataTransformation,
    DataTransformationConfig,
)
from src.components.model_trainer import ModelTrainer, ModelTrainerArtifact, ModelTrainerConfig
from src.logger import get_logger


logger = get_logger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_training_pipeline(
    source_path: str | Path | None = None,
    artifact_dir: str | Path | None = None,
    cv_folds: int = 5,
) -> ModelTrainerArtifact:
    """Train a math-score regressor and save its model and evaluation metrics."""
    artifacts = Path(artifact_dir) if artifact_dir is not None else PROJECT_ROOT / "artifacts"
    dataset = (
        Path(source_path)
        if source_path is not None
        else PROJECT_ROOT / "notebook" / "data" / "stud.csv"
    )

    ingestion = DataIngestion(
        DataIngestionConfig(
            source_path=dataset,
            output_dir=artifacts / "data",
        )
    ).initiate_data_ingestion()
    transformed = DataTransformation(
        DataTransformationConfig(target_column=ingestion.target_column)
    ).initiate_data_transformation(
        train_path=ingestion.train_path,
        test_path=ingestion.test_path,
    )
    result = ModelTrainer(
        ModelTrainerConfig(output_dir=artifacts, cv_folds=cv_folds)
    ).initiate_model_trainer(transformed)
    logger.info("Training pipeline completed. Model saved to %s", result.model_path)
    return result


if __name__ == "__main__":
    training_result = run_training_pipeline()
    print(f"Selected model: {training_result.best_model_name}")
    print(f"Test MAE: {training_result.metrics['test_mae']:.2f}")
    print(f"Test RMSE: {training_result.metrics['test_rmse']:.2f}")
    print(f"Test R^2: {training_result.metrics['test_r2']:.3f}")
    print(f"Model: {training_result.model_path}")
