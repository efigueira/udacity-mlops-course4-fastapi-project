from sklearn.model_selection import train_test_split

from ml.data import CleanData, process_data
from ml.model import (train_model, save_model, load_model, inference,
                      compute_model_metrics)
from environment import Config

config = Config()

# Clean data
df = CleanData().process(data_path=config.data_dir_path, name=config.data_file)

# Split data
train, test = train_test_split(df, test_size=0.2)

# Process data
X_train, y_train, encoder, lb = process_data(
    X=train,
    categorical_features=config.cat_features,
    label='salary',
    training=True)

X_test, y_test, _, _ = process_data(X=test,
                                    categorical_features=config.cat_features,
                                    label='salary',
                                    training=False,
                                    encoder=encoder,
                                    lb=lb)

# Train model
model = train_model(X_train, y_train)

# Save model and encoder
save_model(model, encoder, lb, model_path=config.model_path,
           encoder_path=config.encoder_path, lb_path=config.lb_path)
# Load model and encoder for inference
loaded_model, loaded_encoder, lb = load_model(model_path=config.model_path,
                                              encoder_path=config.encoder_path,
                                              lb_path=config.lb_path)

# Perform inference
predictions = inference(loaded_model, X_test)

# Evaluate metrics
precision, recall, fbeta = compute_model_metrics(y_test, predictions)
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Fbeta: {fbeta}")
