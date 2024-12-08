from pydantic_settings import BaseSettings
from pathlib import Path


class Config(BaseSettings):
    data_dir_path: Path = Path(__file__).parent.parent / 'data'
    model_dir_path: Path = Path(__file__).parent.parent / 'model'
    data_file: str = "census.csv"
    data_file_clean: str = "census_cleaned.csv"
    model_path: Path = model_dir_path / "model.pkl"
    lb_path: Path = model_dir_path / "lb.pkl"
    encoder_path: Path = model_dir_path / "encoder.pkl"

    cat_features: list = ["workclass", "education", "marital-status",
                          "occupation", "relationship", "race", "sex",
                          "native-country"]

    class Config:
        env_file = ".env"
        protected_namespaces = ('settings_',)


if __name__ == "__main__":
    config = Config()
    print("Data Directory Path:", config.data_dir_path)
    print("Model Directory Path:", config.model_dir_path)
    print("Model Path:", config.model_path)
    print("Label Binarizer Path:", config.lb_path)
    print("Encoder Path:", config.encoder_path)
