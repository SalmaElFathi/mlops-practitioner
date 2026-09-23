import pickle

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from prodml.features import features
from prodml.utils import timed


class SalePricePredictor:
    def __init__(self, model_path: str):
        self.model = load_model(model_path)

    @timed
    def predict_one(self, df: pd.DataFrame) -> np.ndarray:
        return predict(df, self.model)

    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        return predict(df, self.model)


def load_model(path: str) -> Pipeline:
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


def predict(df: pd.DataFrame, model: Pipeline) -> np.ndarray:
    df = features(df)
    df = df.drop("Id", axis=1)
    prediction = model.predict(df)
    return prediction
