import pandas as pd
import os
import pickle
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense

from app.utils.config import (
    DATASETS,
    PREDICTION_RESULTS_PATH,
    MODEL_FILENAME,
    TRANSFORMER_FILENAME,
    PREDICTION_CSV_FILENAME
)

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_data(file_num: str):
    """
    Load the dataset file based on selection index (file_num: "1", "2", etc.).
    """
    dataset_path = DATASETS.get(file_num)
    if not dataset_path or not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset for key '{file_num}' not found at {dataset_path}")
    
    logger.info(f"Reading dataset: {dataset_path}")
    return pd.read_csv(dataset_path)


def create_model(data: pd.DataFrame, file_num: str):
    """
    Train the ANN model and save the transformer, model, and predictions.
    """
    predictors = ['Age', 'Gender', 'Experience', 'Education', 'Interview Score', 'Test Score']
    target = ['Salary']

    # Encode categorical variables
    column_transformer = ColumnTransformer(
        transformers=[("onehot", OneHotEncoder(handle_unknown="ignore"), ["Gender", "Education"])],
        remainder="passthrough"
    )

    X = data[predictors]
    y = data[target]

    X_encoded = column_transformer.fit_transform(X)

    # Feature names for encoded dataframe
    new_feature_names = ['Gender_Female', 'Gender_Male', 'Education_Masters', 'Education_PhD',
                         'Age', 'Experience', 'Interview_Score', 'Test_Score']
    X_encoded_df = pd.DataFrame(X_encoded.toarray(), columns=new_feature_names)

    # Train-test split
    X_train, X_test, y_train, y_test, X_real_train, X_real_test = train_test_split(
        X_encoded_df, y, X, test_size=0.3, random_state=42
    )

    # Build the model
    K.clear_session()
    model = Sequential()
    model.add(Dense(units=7, input_dim=8, activation='relu'))
    model.add(Dense(units=7, activation='relu'))
    model.add(Dense(units=1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, batch_size=20, epochs=50, verbose=0)

    # Predict
    predictions = model.predict(X_test)
    testing_df = pd.DataFrame(X_real_test, columns=predictors)
    testing_df['Actual Salary'] = y_test.values
    testing_df['PredictedSalary'] = predictions
    testing_df = testing_df.reset_index(drop=True)

    # Save model and results
    result_dir = Path(PREDICTION_RESULTS_PATH) / f"Datensatz{file_num}_results"
    result_dir.mkdir(parents=True, exist_ok=True)

    with open(result_dir / TRANSFORMER_FILENAME, 'wb') as f:
        pickle.dump(column_transformer, f)
    with open(result_dir / MODEL_FILENAME, 'wb') as f:
        pickle.dump(model, f)

    testing_df.to_csv(result_dir / PREDICTION_CSV_FILENAME, index=False)

    logger.info(f"Model, transformer, and predictions saved in: {result_dir}")
    return X_train, X_test, y_train, model, testing_df


def run_model_generator_for_prediction(file_num: str):
    """
    Orchestrates data loading, model training, and result saving.
    Returns prediction dataframe.
    """
    logger.info(f"Starting model training for dataset {file_num}")
    data = read_data(file_num)
    return create_model(data, file_num)
