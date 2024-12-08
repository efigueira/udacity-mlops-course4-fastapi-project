import sys
import os
import pytest
from fastapi.testclient import TestClient
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import app # noqa: E402


client = TestClient(app)


@pytest.fixture
def user_greater_than_50k():
    return {
        "age": 40,
        "workclass": "Private",
        "fnlgt": 284582,
        "education": "Masters",
        "education-num": 14,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 10000,
        "capital-loss": 0,
        "hours-per-week": 50,
        "native-country": "United-States"
    }


@pytest.fixture
def user_less_equal_50k():
    return {
        "age": 25,
        "workclass": "Private",
        "fnlgt": 150000,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Never-married",
        "occupation": "Handlers-cleaners",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 20,
        "native-country": "United-States"
    }


def test_get_greeting():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"greeting": "Welcome to the API!"}


def test_predict_greater_than_50k(user_greater_than_50k):
    response = client.post("/predict/", json=user_greater_than_50k)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] == ">50K"


def test_predict_less_equal_50k(user_less_equal_50k):
    response = client.post("/predict/", json=user_less_equal_50k)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] == "<=50K"
