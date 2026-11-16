"""Create a minimal Elastic Beanstalk source bundle for the saved model."""

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "student-score-beanstalk.zip"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing student-score-beanstalk.zip.",
    )
    arguments = parser.parse_args()

    required_files = (
        PROJECT_ROOT / "app.py",
        PROJECT_ROOT / "Procfile",
        PROJECT_ROOT / "requirements.txt",
        PROJECT_ROOT / "artifacts" / "model.joblib",
    )
    missing_files = [path for path in required_files if not path.is_file()]
    if missing_files:
        missing = ", ".join(str(path) for path in missing_files)
        raise FileNotFoundError(
            f"Cannot build the deployment bundle; required file(s) missing: {missing}"
        )
    if OUTPUT_PATH.exists() and not arguments.force:
        raise FileExistsError(
            f"{OUTPUT_PATH} already exists. Re-run with --force to replace it."
        )

    with ZipFile(OUTPUT_PATH, mode="w", compression=ZIP_DEFLATED) as bundle:
        for path in (PROJECT_ROOT / "app.py", PROJECT_ROOT / "Procfile"):
            bundle.write(path, arcname=path.relative_to(PROJECT_ROOT).as_posix())

        requirements_path = PROJECT_ROOT / "requirements.txt"
        bundle.write(requirements_path, arcname="requirements.txt")

        for folder in (PROJECT_ROOT / "src", PROJECT_ROOT / "templates"):
            for path in sorted(folder.rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts:
                    bundle.write(
                        path,
                        arcname=path.relative_to(PROJECT_ROOT).as_posix(),
                    )

        model_path = PROJECT_ROOT / "artifacts" / "model.joblib"
        bundle.write(model_path, arcname="artifacts/model.joblib")

    print(f"Created deployment bundle: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
