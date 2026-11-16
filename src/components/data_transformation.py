"""Build preprocessing steps for numeric and categorical model inputs."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import DataTransformationError
from src.logger import get_logger


logger = get_logger(__name__)


@dataclass(frozen=True)
class DataTransformationConfig:
    target_column: str = "math_score"


@dataclass(frozen=True)
class DataTransformationArtifact:
    preprocessor: ColumnTransformer
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    feature_columns: tuple[str, ...]


class DataTransformation:
    """Separate features and target and configure training-only preprocessing."""

    def __init__(self, config: DataTransformationConfig | None = None) -> None:
        self.config = config or DataTransformationConfig()

    def initiate_data_transformation(
        self,
        train_path: str | Path,
        test_path: str | Path,
    ) -> DataTransformationArtifact:
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
        except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
            logger.exception("Could not read ingested train/test data.")
            raise DataTransformationError("Could not read ingested train/test data.") from error

        target = self.config.target_column
        if target not in train_data.columns or target not in test_data.columns:
            raise DataTransformationError(f"Target column '{target}' is missing.")
        if train_data.empty or test_data.empty:
            raise DataTransformationError("Train and test datasets must both contain rows.")

        feature_columns = [column for column in train_data.columns if column != target]
        if not feature_columns:
            raise DataTransformationError("No feature columns are available for training.")
        if set(feature_columns) != set(column for column in test_data.columns if column != target):
            raise DataTransformationError("Train and test feature columns do not match.")

        X_train = train_data[feature_columns]
        X_test = test_data[feature_columns]
        y_train = train_data[target]
        y_test = test_data[target]
        if not pd.api.types.is_numeric_dtype(y_train) or not pd.api.types.is_numeric_dtype(y_test):
            raise DataTransformationError(f"Target column '{target}' must be numeric.")
        if y_train.isna().any() or y_test.isna().any():
            raise DataTransformationError(f"Target column '{target}' contains missing values.")

        numeric_features = X_train.select_dtypes(include="number").columns.tolist()
        categorical_features = X_train.select_dtypes(exclude="number").columns.tolist()
        if not numeric_features and not categorical_features:
            raise DataTransformationError("No supported feature columns are available.")

        transformers = []
        if numeric_features:
            numeric_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            transformers.append(("numeric", numeric_pipeline, numeric_features))
        if categorical_features:
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                ]
            )
            transformers.append(("categorical", categorical_pipeline, categorical_features))

        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
        logger.info(
            "Prepared %d numeric and %d categorical features.",
            len(numeric_features),
            len(categorical_features),
        )
        return DataTransformationArtifact(
            preprocessor=preprocessor,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            feature_columns=tuple(feature_columns),
        )
