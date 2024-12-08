from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path
from typing import List

from environment import Config
from ml import process_data, compute_model_metrics, load_model, CleanData


def slice_feature(data, feature, model, encoder, lb, cat_features):
    slice_info = []
    for cls in data[feature].unique():
        slice_info.append(f"{feature}: {cls}")
        df_temp = data[data[feature] == cls]

        X_test, y_test, _, _ = process_data(
            df_temp, categorical_features=cat_features, label='salary', training=False,
            encoder=encoder, lb=lb
        )

        y_pred = model.predict(X_test)

        precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
        slice_info.append(f"\tPrecision: {precision}")
        slice_info.append(f"\tRecall: {recall}")
        slice_info.append(f"\tFbeta: {fbeta}")
    return slice_info


def compute_metrics_on_slices(df: pd.DataFrame, cat_features: List[str], model_path: Path, encoder_path: Path,
                              lb_path: Path):
    model, encoder, lb = load_model(model_path=model_path, encoder_path=encoder_path, lb_path=lb_path)
    train, test = train_test_split(df, test_size=0.20)
    slices_info = []
    for cat_feature in cat_features:
        slice_info = slice_feature(data=test, feature=cat_feature, model=model, encoder=encoder, lb=lb,
                                   cat_features=cat_features)
        slices_info += slice_info
    info = '\n'.join(slices_info)

    with open(model_path.parent / 'slice_output.txt', 'w') as f:
        f.write(info)


if __name__ == "__main__":
    config = Config()
    df = CleanData().read_data(data_path=config.data_dir_path, name=config.data_file_clean)
    compute_metrics_on_slices(df=df,
                              cat_features=config.cat_features,
                              model_path=config.model_path,
                              encoder_path=config.encoder_path,
                              lb_path=config.lb_path)