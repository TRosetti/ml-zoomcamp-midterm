"""Web service that returns the purchase probability of a browsing session.

Usage (from the project folder):
    uv run python predict.py

Then open http://localhost:9696/docs to see and try the API in the browser.
"""

import pickle
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parent / 'model.bin'

# the model is loaded once, when the service starts, not on every request
with open(MODEL_PATH, 'rb') as f_in:
    pipeline = pickle.load(f_in)


class Session(BaseModel):
    """The request body: one session, with the dataset columns.

    Pydantic validates the input before it reaches the model. If a field is missing or a
    value is out of range (for example, bouncerates = 5), the API answers with a 422 error
    explaining the problem, instead of returning a meaningless prediction.
    """

    administrative: int = Field(ge=0)
    administrative_duration: float = Field(ge=0)
    informational: int = Field(ge=0)
    informational_duration: float = Field(ge=0)
    productrelated: int = Field(ge=0)
    productrelated_duration: float = Field(ge=0)
    bouncerates: float = Field(ge=0, le=1)
    exitrates: float = Field(ge=0, le=1)
    pagevalues: float = Field(ge=0)
    specialday: float = Field(ge=0, le=1)
    month: Literal['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    operatingsystems: int
    browser: int
    region: int
    traffictype: int
    visitortype: Literal['Returning_Visitor', 'New_Visitor', 'Other']
    weekend: bool


class Prediction(BaseModel):
    purchase_probability: float
    will_purchase: bool


app = FastAPI(title='purchase-prediction')


def prepare(session: Session) -> dict:
    """Put the session in the same format used in training.

    In training, the codes (operatingsystems, browser, region, traffictype) were converted
    to strings. If they arrived here as numbers, the DictVectorizer would not recognize
    'browser=2' and would silently ignore the column: the model would still answer, but
    worse. That is why they are converted to str.
    """
    record = session.model_dump()
    for c in ['operatingsystems', 'browser', 'region', 'traffictype']:
        record[c] = str(record[c])
    record['weekend'] = int(record['weekend'])
    return record


@app.post('/predict')
def predict(session: Session) -> Prediction:
    record = prepare(session)
    probability = float(pipeline.predict_proba([record])[0, 1])
    return Prediction(
        purchase_probability=round(probability, 4),
        will_purchase=probability >= 0.5,
    )


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=9696)
