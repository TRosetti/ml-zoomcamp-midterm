# Online Shopper Purchase Prediction

Midterm project for [ML Zoomcamp 2026](https://github.com/DataTalksClub/machine-learning-zoomcamp).

## Problem description

Most visits to an online store end without a purchase. If the store could tell, while a
visitor is still browsing, how likely the session is to end in a purchase, the marketing
and product teams could act on it: offer a coupon or open a live chat only for undecided
visitors, without interrupting the ones who were going to buy anyway, and without giving
discounts to people who would have paid full price.

This project trains a model that takes the data of a browsing session (pages visited and
time spent on them, bounce and exit rates, page value, month, visitor type, traffic
source, ...) and returns the **probability that the session ends in a purchase**. The model
is served as a web service: the store's backend sends a session as JSON and gets back the
probability and a yes/no decision.

Only about 15% of the sessions end in a purchase, so accuracy is misleading (always
answering "no purchase" is already 85% accurate). Models are compared with **ROC AUC**.

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
notebook.ipynb        data preparation, EDA, feature importance, model selection and tuning
train.py              trains the final model and saves it to model.bin
model.bin             trained model (DictVectorizer + XGBoost pipeline)
predict.py            serves the model as a web service (FastAPI)
test.py               sends two sample sessions to the web service
Dockerfile            container for the web service
.dockerignore         keeps the Docker build context to the files the image needs
pyproject.toml        dependencies (managed with uv)
uv.lock               pinned dependency versions
```

## Results

Main findings from the EDA (details in `notebook.ipynb`):

- **`pagevalues` is the strongest feature.** Sessions with `pagevalues > 0` (22% of them)
  end in a purchase 56% of the time, against 4% for the others.
- **November converts the most** (24.8%, 1.6 times the average); February the least (1.3%).
- **New visitors buy more** (24.8%) than returning visitors (13.8%).
- High `exitrates` and `bouncerates` mean the visitor leaves without buying.
- `region` and `browser` carry almost no information about a purchase.
- 125 exact duplicate rows were removed before splitting the data.

Data split: 60% train, 20% validation, 20% test (`random_state=1`). Four models were tuned
on the validation set:

| Model | Best parameters | Validation AUC |
|---|---|---|
| **XGBoost** | `learning_rate=0.05`, `max_depth=4`, `min_child_weight=10`, `n_estimators=200` | **0.927** |
| Random forest | `n_estimators=200`, `max_depth=15`, `min_samples_leaf=3` | 0.921 |
| Decision tree | `max_depth=6`, `min_samples_leaf=200` | 0.918 |
| Logistic regression | `C=0.01` (chosen with 5-fold cross-validation) | 0.891 |

The final model is XGBoost retrained on train + validation: **ROC AUC of 0.937 on the test set**.

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

```bash
uv run python train.py
```

This trains the final model, prints the test AUC (`0.937`) and saves it to `model.bin`.
A trained `model.bin` is already included, so this step is optional.

### 4. Run the web service

```bash
uv run python predict.py
```

The service listens on port 9696. Open http://localhost:9696/docs to try it in the browser.

In another terminal, send two sample sessions:

```bash
uv run python test.py
```

Expected output:

```
likely buyer: {'purchase_probability': 0.7589, 'will_purchase': True}
likely leaver: {'purchase_probability': 0.0054, 'will_purchase': False}
```

Or with `curl`:

```bash
curl -X POST http://localhost:9696/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "administrative": 4, "administrative_duration": 120.0,
    "informational": 1, "informational_duration": 30.0,
    "productrelated": 45, "productrelated_duration": 1800.0,
    "bouncerates": 0.0, "exitrates": 0.01, "pagevalues": 40.0, "specialday": 0.0,
    "month": "Nov", "operatingsystems": 2, "browser": 2, "region": 1,
    "traffictype": 2, "visitortype": "New_Visitor", "weekend": false
  }'
```

Invalid input (a missing field, `bouncerates` above 1, an unknown month, ...) returns
HTTP 422 with a description of the problem.

### 5. Run with Docker

Stop the service from step 4 first (Ctrl+C), since the container uses the same port.

```bash
docker build -t purchase-prediction .
docker run -it --rm -p 9696:9696 purchase-prediction
```

Then, in another terminal:

```bash
uv run python test.py
```

## Deployment

The web service is deployed locally with Docker (step 5). It is not deployed to a cloud provider.

## AI usage

I used Claude (Anthropic) as a coding assistant for the project setup and boilerplate:
the environment, the project structure and the README skeleton.
The data analysis, feature engineering, model selection and tuning are my own work.
