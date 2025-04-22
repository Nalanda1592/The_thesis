from langchain.tools import BaseTool
import pandas as pd
from app.utils.config import PREDICTION_RESULTS_PATH, PREDICTION_CSV_FILENAME

class DatasetTool(BaseTool):
    name = "Dataset Tool"
    description = "Answer questions related to dataset predictions, averages, and columns."

    def _run(self, query: str):
        df = pd.read_csv(PREDICTION_RESULTS_PATH / "Datensatz1_results" / PREDICTION_CSV_FILENAME)
        return df.describe(include='all').to_string()

    def _arun(self, query: str):
        raise NotImplementedError("Async not supported.")
