
from argparse import ArgumentParser
import os
import pandas as pd
import numpy as np
import joblib
import json
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score

def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, "model.joblib"))

def input_fn(request_body, request_content_type):
    input_data_list = json.loads(request_body)
    return np.array(input_data_list)

def predict_fn(input_data, model):
    prediction = model.predict(input_data)
    return prediction.tolist()

def output_fn(prediction, content_type):
    return json.dumps(prediction)

if __name__ == "__main__":
    print("Extracting arguments...")
    parser = ArgumentParser()

    # Hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--n-estimators", type=int, default=200)  # Best value from Random Search
    parser.add_argument("--learning-rate", type=float, default=0.2)  # Best value from Random Search
    parser.add_argument("--max-depth", type=int, default=3)  # Best value from Random Search
    parser.add_argument("--subsample", type=float, default=0.8)  # Best value from Random Search
    parser.add_argument("--gamma", type=float, default=0.1)  # Best value from Random Search
    parser.add_argument("--colsample-bytree", type=float, default=1.0)  # Best value from Random Search

    # Data, model, and output directories
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--train-file", type=str, default="train_scaled.csv")
    parser.add_argument("--test-file", type=str, default="test_scaled.csv")
    
    # Local path to save the model in the current directory
    parser.add_argument("--local-model-path", type=str, default="model_local.joblib")  
    args, _ = parser.parse_known_args()

    print("Train channel:", args.train)
    print("Test channel:", args.test)

    # Load training and testing data
    print("Reading data...")
    df_train = pd.read_csv(os.path.join(args.train, args.train_file))
    df_test = pd.read_csv(os.path.join(args.test, args.test_file))

    # Check for NaN values
    if df_train.isnull().any().any():
        raise ValueError("Training data contains NaN values.")

    print("Building training and testing datasets...")
    TARGET_NAME = "price_range"
    all_columns_name = [col for col in df_train.columns if col not in [TARGET_NAME, 'id', 'Unnamed: 0']]
    
    X_train = df_train[all_columns_name]
    y_train = df_train[TARGET_NAME].values

    # Train model using XGBoost
    print("Training model...")
    model = xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
        gamma=args.gamma,
        colsample_bytree=args.colsample_bytree,
        eval_metric="mlogloss",
        use_label_encoder=False,  # Important for newer versions of XGBoost
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Validate model
    print("Validating model...")
    bal_acc_train = balanced_accuracy_score(y_train, model.predict(X_train))
    y_test = df_test[TARGET_NAME].values
    bal_acc_test = balanced_accuracy_score(y_test, model.predict(df_test[all_columns_name]))

    print(f"Train balanced accuracy: {100 * bal_acc_train:.3f} %")
    print(f"Test balanced accuracy: {100 * bal_acc_test:.3f} %")

    # Persist model to SageMaker model directory
    path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, path)
    print("Model persisted at " + path)

    # Save the model locally in the current directory
    local_model_path = os.path.join(os.getcwd(), args.local_model_path)  # Save in the current directory
    joblib.dump(model, local_model_path)
    print(f"Model saved locally at {local_model_path}")
