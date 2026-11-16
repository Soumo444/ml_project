# Data Science and Machine Learning Project

Learning project for data ingestion, transformation, model training, and prediction.

## Live demo

The app is deployed on AWS Elastic Beanstalk in `ap-southeast-2`:

http://Student-math-score-env.eba-2bxkhthf.ap-southeast-2.elasticbeanstalk.com/

## Set up the development environment

Install the full local data-science stack:

```powershell
python -m pip install -r requirements-dev.txt
```

`requirements.txt` is the minimal runtime set used by Elastic Beanstalk;
`requirements-dev.txt` adds the notebooks, training, and experimentation
packages.

## Train the model

From the project root, activate your Python environment and run:

```powershell
python -m src.pipeline.train_pipeline
```

By default, the pipeline reads `notebook/data/stud.csv`, uses `math_score` as
the target, and uses the remaining dataset columns as features. It creates a
reproducible 80/20 train/test split, tunes linear, tree-based, neighbor,
support-vector, and XGBoost regressors using five-fold cross-validation, selects
the model with the lowest cross-validation MAE, then evaluates it once on the
held-out test set. Splits, the trained pipeline, and metrics are written under
`artifacts/`.

The model uses `reading_score` and `writing_score` to predict `math_score`;
those scores must be available when making predictions.

The trained model is a trusted local joblib artifact. Do not load model files
from untrusted sources.

To predict from a pandas DataFrame after training:

```python
import pandas as pd

from src.pipeline.predict_pipeline import PredictionPipeline

student = pd.DataFrame(
    [{
        "gender": "female",
        "race_ethnicity": "group B",
        "parental_level_of_education": "bachelor's degree",
        "lunch": "standard",
        "test_preparation_course": "completed",
        "reading_score": 72,
        "writing_score": 74,
    }]
)
predictions = PredictionPipeline().predict(student)
print(predictions["predicted_math_score"])
```

## Run the Flask prediction app

After training the model, start the web app from the project root:

```powershell
python app.py
```

Open `http://127.0.0.1:5000` in your browser and enter the student's details.
The app validates the inputs and displays a math-score estimate. The trained
model must exist at `artifacts/model.joblib`.

## Deploy to AWS Elastic Beanstalk

The repository is deployment-ready: the root `requirements.txt` contains only
the pinned runtime dependencies, and `artifacts/model.joblib` is committed.

### CI/CD with CodePipeline (recommended)

Pushing to the `main` branch of
[`Soumo444/ml_project`](https://github.com/Soumo444/ml_project) triggers AWS
CodePipeline, which deploys the repository to the existing Elastic Beanstalk
environment:

- Application: `student-math-score`
- Environment: `Student-math-score-env`
- Pipeline: `student-math-score-pipeline`
- Region: `ap-southeast-2`
- Platform: Python 3.15 on 64-bit Amazon Linux 2023

Pipeline console: [student-math-score-pipeline](https://ap-southeast-2.console.aws.amazon.com/codesuite/codepipeline/pipelines/student-math-score-pipeline/view?region=ap-southeast-2)

Deployment flow: GitHub push to `main` → `Source` → `Deploy` → `Student-math-score-env`.

The pipeline source is the GitHub repository, and the deploy stage creates a
new Elastic Beanstalk application version from each commit. The GitHub
connection is authorized once in **Developer Tools > Settings > Connections**
in the AWS console. To release a change, commit and push:

```powershell
git add .
git commit -m "Describe the change"
git push origin main
```

### Verify the deployment

After the pipeline reports **Succeeded**, check the environment URL:

```powershell
Invoke-WebRequest http://Student-math-score-env.eba-2bxkhthf.ap-southeast-2.elasticbeanstalk.com/ -UseBasicParsing
```

The page should return HTTP 200 and show the prediction form. You can also check
the environment health in the Elastic Beanstalk console.

### Manual bundle (fallback)

If you are not using the pipeline, build the minimal source bundle:

```powershell
python scripts\package_beanstalk.py --force
```

This creates `student-score-beanstalk.zip`, which includes the trained model
and the same pinned `requirements.txt`. Upload it as a new application version
in the Elastic Beanstalk console and deploy it to the environment.

The bundle uses a Gunicorn `Procfile`. See the
[AWS Flask deployment guide](https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/create-deploy-python-flask.html)
and [Procfile reference](https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/python-configuration-procfile.html).

AWS resources can incur charges. When finished experimenting, delete the
Elastic Beanstalk environment (and application if no longer needed).

## Shared utilities

Use the configured logger with `from src.logger import get_logger` and
`logger = get_logger(__name__)`. It writes to the console and rotates
`logs/project_beg.log`.

The `src.utils` module provides `ensure_directory`, `save_json`, `load_json`,
`save_object`, and `load_object`. Filesystem and serialization failures are
raised as `ProjectError` exceptions.

Run the utility tests from the project root with:

```powershell
python -m unittest discover -s tests
```
