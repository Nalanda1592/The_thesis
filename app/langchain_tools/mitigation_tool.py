from langchain.tools import BaseTool
import pandas as pd
from app.utils.config import PREDICTION_RESULTS_PATH, FAIRLEARN_FILENAMES

class MitigationTool(BaseTool):
    name = "Mitigation Tool"
    description = "Explain fairness mitigation strategies applied to models (e.g., adversarial, SMOTE)."

    def _run(self, query: str):
        df = pd.read_csv(
            PREDICTION_RESULTS_PATH / "Datensatz1_results" / "fairness_measures" / FAIRLEARN_FILENAMES["mitigation_1"]
        )
        return df.describe(include="all").to_string()

    def _arun(self, query: str):
        raise NotImplementedError("Async not supported.")
