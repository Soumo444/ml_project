from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent
requirements = [
    line.strip()
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.lstrip().startswith("#")
]

setup(
    name="project-beg",
    version="0.1.0",
    description="Data science, machine learning, deep learning, and NLP learning project",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=requirements,
)
