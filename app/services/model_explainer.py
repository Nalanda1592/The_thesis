import shap
import pickle
import numpy as np
import pandas as pd
from app.utils.config import *


def run_shap_explanation(file_num: str):
    folder = PREDICTION_RESULTS_PATH / f"Datensatz{file_num}_results"
    data_path = folder / PREDICTION_CSV_FILENAME
    model_path = folder / MODEL_FILENAME
    transformer_path = folder / TRANSFORMER_FILENAME

    df = pd.read_csv(data_path)
    X = df[['Gender', 'Education', 'Age', 'Experience', 'Interview Score', 'Test Score']]
    X_encoded = pickle.load(open(transformer_path, 'rb')).transform(X)
    model = pickle.load(open(model_path, 'rb'))

    X_sample = shap.sample(X_encoded, 100)
    explainer = shap.KernelExplainer(model.predict, X_sample)
    shap_values = explainer.shap_values(X_sample)
    swapped = np.swapaxes(np.array(shap_values), 1, 2)

    shap_df = pd.DataFrame(swapped[0])
    result_df = df.head(100).join(shap_df.reset_index(drop=True))

    output_folder = folder / "explanations"
    output_folder.mkdir(exist_ok=True)
    pickle.dump(explainer, open(output_folder / EXPLAINER_FILENAME, 'wb'))
    result_df.to_csv(output_folder / SHAP_RESULT_FILENAME, index=False)

    return result_df.head(10)