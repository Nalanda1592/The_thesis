from pathlib import Path

# === Root Directory of the Project ===
ROOT_DIR = Path(__file__).resolve().parents[2]

# === Dataset Source Files ===
DATASETS = {
    "1": ROOT_DIR / "data" / "Datensatz 1.csv",
    "2": ROOT_DIR / "data" / "Datensatz 2.csv",
    # Add more datasets here as needed
}

# === Folder to Save Prediction Results ===
PREDICTION_RESULTS_PATH = ROOT_DIR / "data" / "total_prediction_results"

# === Folder to Save Single Prediction Results ===
SINGLE_PREDICTION_RESULTS_PATH = ROOT_DIR / "data" / "single_prediction_results"

# === Model File Naming Conventions ===
MODEL_FILENAME = "neural_model.sav"
TRANSFORMER_FILENAME = "column_transformer.pkl"
PREDICTION_CSV_FILENAME = "total_database_prediction.csv"

# === SHAP Explanations ===
SHAP_RESULT_FILENAME = "shap_result.csv"
SINGLE_SHAP_FILENAME = "total_result.csv"
EXPLAINER_FILENAME = "explainer.pkl"

# === Fairlearn Result Files ===
FAIRLEARN_FILENAMES = {
    "overall_metrics": "Fairlearn_results_1.csv",
    "quantile_metrics": "Fairlearn_results_2.csv",
    "augmented_dataset": "Fairlearn_results_3.csv",
    "text_context": "Fairlearn_results_4.csv",
    "metric_frame": "Fairlearn_results_5.csv",
    "mitigation_1": "unfairness_mitigation_result_1.csv",
    "mitigation_2": "unfairness_mitigation_result_2.csv"
}

# === Environment File (used to load API keys) ===
ENV_FILE_PATH = ROOT_DIR / "scripts" / "dev.env"
