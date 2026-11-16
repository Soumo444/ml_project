"""Read, validate, and split the student score dataset."""

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import DataIngestionError, ProjectError
from src.logger import get_logger
from src.utils import ensure_directory


logger = get_logger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET_COLUMN = "math_score"
NUMERIC_SCORE_COLUMNS = {"math_score", "reading_score", "writing_score"}
REQUIRED_COLUMNS = {
    "gender",
    "race_ethnicity",
    "parental_level_of_education",
    "lunch",
    "test_preparation_course",
    "math_score",
    "reading_score",
    "writing_score",
}


@dataclass(frozen=True)
class DataIngestionConfig:
    source_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "notebook" / "data" / "stud.csv"
    )
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "artifacts" / "data")
    test_size: float = 0.2
    random_state: int = 42


@dataclass(frozen=True)
class DataIngestionArtifact:
    train_path: Path
    test_path: Path
    target_column: str = TARGET_COLUMN


class DataIngestion:
    """Create reproducible train/test CSV splits from the source dataset."""

    def __init__(self, config: DataIngestionConfig | None = None) -> None:
        self.config = config or DataIngestionConfig()

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        source_path = Path(self.config.source_path)
        if not source_path.is_file():
            raise DataIngestionError(f"Dataset file does not exist: {source_path}")
        if not 0 < self.config.test_size < 1:
            raise DataIngestionError("test_size must be greater than 0 and less than 1.")

        try:
            data = pd.read_csv(source_path)
        except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
            logger.exception("Could not read dataset %s", source_path)
            raise DataIngestionError(f"Could not read dataset '{source_path}'.") from error

        if data.empty:
            raise DataIngestionError(f"Dataset is empty: {source_path}")

        missing_columns = REQUIRED_COLUMNS - set(data.columns)
        if missing_columns:
            names = ", ".join(sorted(missing_columns))
            raise DataIngestionError(f"Dataset is missing required columns: {names}")
        for column in NUMERIC_SCORE_COLUMNS:
            if not pd.api.types.is_numeric_dtype(data[column]):
                raise DataIngestionError(f"Score column '{column}' must be numeric.")

        original_row_count = len(data)
        data = data.dropna(subset=[TARGET_COLUMN]).copy()
        if data.empty:
            raise DataIngestionError(
                f"Dataset has no rows with a valid '{TARGET_COLUMN}' target."
            )
        if len(data) != original_row_count:
            logger.warning(
                "Dropped %d rows with missing targets.",
                original_row_count - len(data),
            )

        try:
            train_data, test_data = train_test_split(
                data,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
            )
            output_dir = ensure_directory(self.config.output_dir)
            train_path = output_dir / "train.csv"
            test_path = output_dir / "test.csv"
            train_data.to_csv(train_path, index=False)
            test_data.to_csv(test_path, index=False)
        except (OSError, ValueError, ProjectError) as error:
            logger.exception("Could not create train/test data splits.")
            raise DataIngestionError("Could not create train/test data splits.") from error

        logger.info(
            "Created train/test splits with %d and %d rows.",
            len(train_data),
            len(test_data),
        )
        return DataIngestionArtifact(train_path, test_path)
