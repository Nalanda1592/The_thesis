import pandas as pd
import pickle
import os
import numpy as np
from app.utils.config import *

def predict_single_instance(values: list, file_num: str):
    folder = PREDICTION_RESULTS_PATH / f"Datensatz{file_num}_results"
    model = pickle.load(open(folder / MODEL_FILENAME, 'rb'))
    transformer = pickle.load(open(folder / TRANSFORMER_FILENAME, 'rb'))
    explainer = pickle.load(open(folder / "explanations" / EXPLAINER_FILENAME, 'rb'))

    columns = ["Age", "Gender", "Experience", "Education", "Interview Score", "Test Score"]
    input_df = pd.DataFrame([values], columns=columns)
    encoded_input = transformer.transform(input_df)
    prediction = model.predict(encoded_input)[0]

    shap_values = explainer.shap_values(encoded_input)
    swapped = np.swapaxes(np.array(shap_values), 1, 2)
    shap_df = pd.DataFrame(swapped[0]).reset_index(drop=True)

    result_df = input_df.copy()
    result_df["predicted salary"] = prediction
    result_df = result_df.join(shap_df)

    out_path = SINGLE_PREDICTION_RESULTS_PATH / f"Datensatz{file_num}_single_results"
    out_path.mkdir(exist_ok=True, parents=True)
    result_df.to_csv(out_path / SINGLE_SHAP_FILENAME, index=False)

    return prediction