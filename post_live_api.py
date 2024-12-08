import requests

url = "https://udacity-mlops-course4-fastapi-project.onrender.com/predict/"

payload = {
    "age": 35,
    "workclass": "Private",
    "fnlgt": 284582,
    "education": "Masters",
    "education-num": 14,
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 50,
    "native-country": "United-States"
}

response = requests.post(url, json=payload)
print(f"Status Code: {response.status_code}")
print(f"Response JSON: {response.json()}")
