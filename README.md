# DataScience
This repository collects assorted data science practice files.

## Example pipeline

The repository now contains `industry_pipeline.py`, a self-contained example of an
"industry-style" machine learning workflow. It loads the Iris dataset, trains a
`LogisticRegression` model using a `Pipeline`, reports evaluation metrics, and
persists the trained model using `joblib`.

Run the example with:

```bash
python industry_pipeline.py
```

The script will output a classification report and save `iris_model.joblib` in the
current directory.
