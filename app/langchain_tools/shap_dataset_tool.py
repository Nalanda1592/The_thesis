from langchain.tools import BaseTool
import pandas as pd
from app.utils.config import PREDICTION_RESULTS_PATH, SHAP_RESULT_FILENAME

class SHAPDatasetTool(BaseTool):
    name = "SHAP Tool"
    description = "Answer feature importance and explainability questions using SHAP values."

    def _run(self, query: str):
        df = pd.read_csv(PREDICTION_RESULTS_PATH / "Datensatz1_results" / "explanations" / SHAP_RESULT_FILENAME)
        return df.head().to_string()

    def _arun(self, query: str):
        raise NotImplementedError("Async not supported.")
