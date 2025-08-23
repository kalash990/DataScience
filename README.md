# DataScience
This repository collects assorted data science practice files.

## Example pipeline

The repository now contains two simple machine-learning examples:

- `industry_pipeline.py` trains a logistic regression classifier on the Iris dataset.
- `comcast_pipeline.py` trains a text classifier on the `comcast.csv` complaints dataset and
  saves the resulting model to `comcast_model.joblib`.

Run the Comcast example with:

```bash
python comcast_pipeline.py
```

The script prints a classification report and writes the trained model artifact to disk.
