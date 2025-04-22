from langchain.tools import BaseTool
import pandas as pd
from app.utils.config import PREDICTION_RESULTS_PATH, FAIRLEARN_FILENAMES

class FairnessTool(BaseTool):
    name = "Fairness Tool"
    description = "Answer fairness evaluation questions using Fairlearn metrics and narrative."

    def _run(self, query: str):
        base_path = PREDICTION_RESULTS_PATH / "Datensatz1_results" / "fairness_measures"
        df = pd.read_csv(base_path / FAIRLEARN_FILENAMES["narrative"])
        return df.iloc[0, 0]

    def _arun(self, query: str):
        raise NotImplementedError("Async not supported.")
