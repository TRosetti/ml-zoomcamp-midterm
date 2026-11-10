# Online Shopper Purchase Prediction

Midterm project for [ML Zoomcamp 2026](https://github.com/DataTalksClub/machine-learning-zoomcamp).

## Problem description

<!-- TODO: what problem, who uses the model, and how the prediction is used in practice -->

## Dataset

[Online Shoppers Purchasing Intention](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
from the UCI Machine Learning Repository, licensed under CC BY 4.0.

- 12,330 browsing sessions, each from a different user over one year
- 17 features (page visits and durations, bounce and exit rates, page value, month, visitor type, traffic source, ...)
- Target: `Revenue` (whether the session ended in a purchase), about 15.5% positive
- No missing values

The CSV is included in this repository at `data/online_shoppers_intention.csv`.

## Project structure

```
data/                 dataset
notebook.ipynb        EDA, feature importance, model selection and tuning
train.py              trains the final model and saves it        (TODO)
predict.py            serves the model as a web service (FastAPI) (TODO)
Dockerfile            container for the web service              (TODO)
pyproject.toml        dependencies (managed with uv)
uv.lock               pinned dependency versions
```

## How to run

### 1. Set up the environment

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then from the project folder:

```bash
uv sync
```

This creates a virtual environment in `.venv` with the exact versions from `uv.lock`.

> macOS only: XGBoost needs the OpenMP runtime (`brew install libomp`).

### 2. Explore the notebook

```bash
uv run jupyter notebook notebook.ipynb
```

### 3. Train the model

<!-- TODO -->

### 4. Run the web service

<!-- TODO -->

### 5. Run with Docker

<!-- TODO -->

## Deployment

<!-- TODO: cloud deployment steps + URL or screenshot/video -->
