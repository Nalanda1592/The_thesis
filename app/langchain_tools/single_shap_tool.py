from langchain.tools import BaseTool
import pandas as pd
from app.utils.config import SINGLE_PREDICTION_RESULTS_PATH, SINGLE_SHAP_FILENAME

class SingleSHAPTool(BaseTool):
    name = "Single SHAP Tool"
    description = "Answer SHAP explainability questions based on a single prediction instance."

    def _run(self, query: str):
        df = pd.read_csv(SINGLE_PREDICTION_RESULTS_PATH / "Datensatz1_single_results" / SINGLE_SHAP_FILENAME)
        return df.iloc[0].to_string()

    def _arun(self, query: str):
        raise NotImplementedError("Async not supported.")
