from pathlib import Path
from ml import CleanData, process_data
from ml import get_train_test_data, train_model, save_model, load_model, inference, compute_model_metrics


cat_features = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]

DATA_DIR_PATH = Path(__file__).parent.parent / 'data'
MODEL_DIR_PATH = Path(__file__).parent.parent / 'model'
model_path = MODEL_DIR_PATH / "model.pkl"
encoder_path = MODEL_DIR_PATH / "encoder.pkl"

df = CleanData().process(data_path=DATA_DIR_PATH, name='census.csv')
X, y, encoder, lb = process_data(X=df, categorical_features=cat_features, label='salary', training=True)

# Split data
X_train, X_test, y_train, y_test = get_train_test_data(X, y)

# Train model
model = train_model(X_train, y_train)

# Save model and encoder
save_model(model, encoder, model_path=model_path, encoder_path=encoder_path)
# Load model and encoder for inference
loaded_model, loaded_encoder = load_model(model_path=model_path, encoder_path=encoder_path)

# Perform inference
predictions = inference(loaded_model, X_test)

# Evaluate metrics
precision, recall, fbeta = compute_model_metrics(y_test, predictions)
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Fbeta: {fbeta}")
