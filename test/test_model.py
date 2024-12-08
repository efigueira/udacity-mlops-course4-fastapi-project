import sys
from pathlib import Path
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.environment import Config  # noqa: E402
from src.ml.data import CleanData, process_data  # noqa: E402
from src.ml.model import \
    (train_model, inference, compute_model_metrics)  # noqa: E402


# Setup common test data for all tests
@pytest.fixture
def test_data():
    config = Config()
    name = "census_cleaned.csv"
    cat_features = ["workclass", "education", "marital-status", "occupation",
                    "relationship", "race", "sex", "native-country"]
    # Load data and preprocess
    df = CleanData().read_data(config.data_dir_path, name)
    X, y, encoder, lb = process_data(X=df, categorical_features=cat_features,
                                     label='salary', training=True)

    return X, y, encoder


def test_preprocess_data(test_data):
    """Test that preprocessing returns the expected types."""
    X, y, encoder = test_data
    assert isinstance(X, np.ndarray), "X should be a numpy array."
    assert isinstance(y, np.ndarray), "y should be a numpy array."
    assert isinstance(encoder, OneHotEncoder), "Should be a OneHotEncoder."


def test_train_model(test_data):
    """Test that the model training returns a RandomForestClassifier."""
    X, y, _ = test_data
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier), \
        "Model should be a RandomForestClassifier."


def test_inference(test_data):
    """Test that the model inference returns a numpy array."""
    X, y, _ = test_data
    model = train_model(X, y)
    predictions = inference(model, X)

    assert len(predictions) == len(X), \
        "Predictions should match the number of samples."
    assert predictions.dtype.name in ["int64", "float64"], \
        "Predictions should be numeric."


def test_compute_model_metrics(test_data):
    """Test that the classification metrics return a tuple of 3."""
    X, y, _ = test_data
    model = train_model(X, y)
    predictions = inference(model, X)
    metrics = compute_model_metrics(y, predictions)

    assert isinstance(metrics, tuple), "Metrics should be a tuple."
    assert len(metrics) == 3, \
        "Metrics should include precision, recall, and fbeta."
