import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.exception import ProjectError
from src.logger import get_logger
from src.utils import ensure_directory, load_json, save_json


class ProjectUtilitiesTests(unittest.TestCase):
    def test_logger_writes_to_file_and_does_not_duplicate_handlers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            log_file = Path(temporary_directory) / "test.log"
            with patch("src.logger.LOG_FILE", log_file):
                test_logger = get_logger("tests.logger")
                handlers = list(test_logger.handlers)
                try:
                    self.assertEqual(len(handlers), 2)
                    self.assertIs(get_logger("tests.logger"), test_logger)
                    self.assertEqual(len(test_logger.handlers), 2)

                    test_logger.info("logger test message")
                    for handler in handlers:
                        handler.flush()

                    self.assertIn("logger test message", log_file.read_text(encoding="utf-8"))
                finally:
                    for handler in handlers:
                        test_logger.removeHandler(handler)
                        handler.close()

    def test_ensure_directory_creates_nested_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory) / "nested" / "output"
            self.assertEqual(ensure_directory(directory), directory)
            self.assertTrue(directory.is_dir())

    def test_json_round_trip_and_invalid_json_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            json_file = Path(temporary_directory) / "nested" / "data.json"
            expected = {"items": [1, 2], "name": "sample"}

            self.assertEqual(save_json(expected, json_file), json_file)
            self.assertEqual(load_json(json_file), expected)

            json_file.write_text("{invalid", encoding="utf-8")
            with self.assertRaises(ProjectError) as raised:
                load_json(json_file)
            self.assertIsNotNone(raised.exception.__cause__)

    def test_ensure_directory_reports_file_path_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            file_path = Path(temporary_directory) / "not-a-directory"
            file_path.write_text("file", encoding="utf-8")

            with self.assertRaises(ProjectError) as raised:
                ensure_directory(file_path)
            self.assertIsNotNone(raised.exception.__cause__)


if __name__ == "__main__":
    unittest.main()
