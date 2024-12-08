from typing import Tuple

from sklearn.metrics import fbeta_score, precision_score, recall_score

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib


def get_train_test_data(X: pd.DataFrame, y: pd.Series) -> tuple:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


def save_model(model, encoder, model_path, encoder_path):
    joblib.dump(model, model_path)
    joblib.dump(encoder, encoder_path)


def load_model(model_path, encoder_path):
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    return model, encoder


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)


def compute_model_metrics(y, preds) -> Tuple[float, float, float]:
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta
