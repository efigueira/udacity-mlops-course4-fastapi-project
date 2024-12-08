import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
import pandas as pd
from pathlib import Path


class CleanData:
    _int_cols = ["age", "fnlgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
    _cat_cols = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "native-country"]
    _bin_cols = ["sex", "salary"]

    def process(self, data_path: Path, name: str, save_cleaned: bool = False) -> pd.DataFrame:
        df = self._read_data(data_path, name)
        self._print_info(df)
        df = self._remove_whitespaces(df)
        df = self._assign_correct_type_to_features(df)
        self._print_info(df)
        if save_cleaned:
            self._save_clean_data(df=df, data_folder=data_path, name=name)
        return df

    @staticmethod
    def _read_data(data_path: Path, name: str) -> pd.DataFrame:
        file_path = data_path / name
        return pd.read_csv(file_path)

    @staticmethod
    def _save_clean_data(df: pd.DataFrame, data_folder: Path, name: str) -> pd.DataFrame:
        cleaned_file_path = data_folder / (Path(name).stem + "_cleaned.csv")
        df.to_csv(cleaned_file_path, index=False)
        return df

    @staticmethod
    def _print_info(df: pd.DataFrame):
        print(df.head(5))
        print(df.info())

    def _assign_correct_type_to_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self._int_cols] = df[self._int_cols].astype("int")
        df[self._cat_cols] = df[self._cat_cols].astype("category")
        df[self._bin_cols] = df[self._bin_cols].apply(
            lambda col: col.map({"Male": 1, "Female": 0}) if col.name == "sex" else col.map({">50K": 1, "<=50K": 0}))
        return df

    @staticmethod
    def _remove_whitespaces(df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [col.replace(" ", "") for col in df.columns]
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        return df

    @property
    def cat_cols(self):
        return self._cat_cols


def process_data(
    X, categorical_features=[], label=None, training=True, encoder=None, lb=None
):
    """ Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb
