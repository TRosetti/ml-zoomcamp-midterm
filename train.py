"""Train the final model and save it to model.bin.

This is the notebook reduced to what the final model needs: the data preparation and the
training of the selected model (XGBoost with the parameters tuned in notebook.ipynb).

Usage (from the project folder):
    uv run python train.py
"""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

# paths relative to this file, so the script works from any folder
HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / 'data' / 'online_shoppers_intention.csv'
MODEL_PATH = HERE / 'model.bin'

NUMERICAL = [
    'administrative', 'administrative_duration',
    'informational', 'informational_duration',
    'productrelated', 'productrelated_duration',
    'bouncerates', 'exitrates', 'pagevalues', 'specialday', 'weekend',
]
CATEGORICAL = [
    'month', 'operatingsystems', 'browser', 'region', 'traffictype', 'visitortype',
]
FEATURES = NUMERICAL + CATEGORICAL

# parameters selected in the notebook (section 6)
XGB_PARAMS = dict(learning_rate=0.05, max_depth=4, min_child_weight=10, n_estimators=200)


def load_data(path):
    """Read the CSV and apply the same preparation as section 1 of the notebook."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()

    df['revenue'] = df['revenue'].astype(int)
    df['weekend'] = df['weekend'].astype(int)
    # category codes: converted to strings so they are one-hot encoded
    for c in ['operatingsystems', 'browser', 'region', 'traffictype']:
        df[c] = df[c].astype(str)

    return df.drop_duplicates().reset_index(drop=True)


def train(df):
    """Train DictVectorizer + XGBoost together, in a single pipeline.

    The pipeline keeps both steps in one object: callers pass a list of dictionaries to
    pipeline.predict_proba() and the one-hot encoding happens inside.
    """
    dicts = df[FEATURES].to_dict(orient='records')
    y = df['revenue'].values

    pipeline = make_pipeline(
        DictVectorizer(sparse=False),
        XGBClassifier(random_state=1, n_jobs=4, **XGB_PARAMS),
    )
    pipeline.fit(dicts, y)
    return pipeline


def main():
    df = load_data(DATA_PATH)

    # same split as the notebook: 80% to train the final model, 20% for test
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

    pipeline = train(df_full_train)

    test_dicts = df_test[FEATURES].to_dict(orient='records')
    y_pred = pipeline.predict_proba(test_dicts)[:, 1]
    auc = roc_auc_score(df_test['revenue'].values, y_pred)
    print(f'test ROC AUC: {auc:.3f}')

    with open(MODEL_PATH, 'wb') as f_out:
        pickle.dump(pipeline, f_out)
    print(f'model saved to {MODEL_PATH}')


if __name__ == '__main__':
    main()
