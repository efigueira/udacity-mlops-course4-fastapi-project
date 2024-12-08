from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn

from src.environment import Config
from src.ml.data import CleanData, process_data
from src.ml.model import load_model, inference


app = FastAPI()


class InferenceData(BaseModel):
    age: int = Field(...,
                     example=49)
    workclass: str = Field(...,
                           example="Private")
    fnlgt: int = Field(...,
                       example=94638)
    education: str = Field(...,
                           example="HS-grad")
    education_num: int = Field(...,
                               alias='education-num',
                               example=9)
    marital_status: str = Field(...,
                                alias='marital-status',
                                example="Separated")
    occupation: str = Field(...,
                            example="Adm-clerical")
    relationship: str = Field(...,
                              example="Unmarried")
    race: str = Field(...,
                      example="White")
    sex: str = Field(...,
                     example="Male")
    capital_gain: int = Field(...,
                              alias='capital-gain',
                              example=0)
    capital_loss: int = Field(...,
                              alias='capital-loss',
                              example=0)
    hours_per_week: int = Field(...,
                                alias='hours-per-week',
                                example=40)
    native_country: str = Field(...,
                                alias='native-country',
                                example="United-States")

    class Config:
        json_schema_extra = {
            "example": {
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
                "hours-per-week": 50,  # Updated key to match the alias exactly
                "native-country": "United-States",
            }
        }


@app.get("/")
def greeting():
    return {"greeting": "Welcome to the API!"}


@app.post("/predict/")
def predict(data: InferenceData):
    config = Config()

    # Load the model and encoder
    model, encoder, lb = load_model(model_path=config.model_path,
                                    encoder_path=config.encoder_path,
                                    lb_path=config.lb_path)

    df = CleanData().process_inference(data)

    # Load model and encoder for inference
    loaded_model, loaded_encoder, lb = load_model(
        model_path=config.model_path,
        encoder_path=config.encoder_path,
        lb_path=config.lb_path)

    data_processed, _, _, _ = process_data(
        X=df,
        categorical_features=config.cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb)
    prediction = inference(loaded_model, data_processed)[0]
    pred_dict = {1: ">50K", 0: "<=50K"}
    return {'prediction': pred_dict[prediction]}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
