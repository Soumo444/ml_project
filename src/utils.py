"""Shared filesystem and JSON helpers."""

import json
import pickle
from pathlib import Path
from typing import Any

import joblib

from src.exception import ProjectError
from src.logger import logger


def ensure_directory(path: str | Path) -> Path:
    """Create a directory and return its path."""
    directory = Path(path)
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        logger.exception("Could not create directory %s", directory)
        raise ProjectError(f"Could not create directory '{directory}'") from error
    return directory


def save_json(data: Any, file_path: str | Path) -> Path:
    """Write JSON data to a file, creating its parent directory if needed."""
    target = Path(file_path)
    ensure_directory(target.parent)
    try:
        with target.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
            file.write("\n")
    except (OSError, TypeError, ValueError) as error:
        logger.exception("Could not write JSON file %s", target)
        raise ProjectError(f"Could not write JSON file '{target}'") from error
    return target


def load_json(file_path: str | Path) -> Any:
    """Read JSON data from a file."""
    target = Path(file_path)
    try:
        with target.open(encoding="utf-8") as file:
            return json.load(file)
    except (OSError, ValueError) as error:
        logger.exception("Could not read JSON file %s", target)
        raise ProjectError(f"Could not read JSON file '{target}'") from error


def save_object(value: Any, file_path: str | Path) -> Path:
    """Serialize a project-generated Python artifact with joblib."""
    target = Path(file_path)
    ensure_directory(target.parent)
    try:
        joblib.dump(value, target)
    except (OSError, TypeError, ValueError, pickle.PicklingError) as error:
        logger.exception("Could not save artifact %s", target)
        raise ProjectError(f"Could not save artifact '{target}'") from error
    return target


def load_object(file_path: str | Path) -> Any:
    """Load a trusted artifact previously created by this project."""
    target = Path(file_path)
    if not target.is_file():
        raise ProjectError(f"Artifact file does not exist: {target}")
    try:
        return joblib.load(target)
    except (OSError, EOFError, ImportError, ValueError, pickle.UnpicklingError) as error:
        logger.exception("Could not load artifact %s", target)
        raise ProjectError(f"Could not load artifact '{target}'") from error
