from langchain.tools import BaseTool
from app.services.single_prediction import predict_single_instance

class SinglePredictionTool(BaseTool):
    name = "Single Prediction Tool"
    description = "Predict salary for a single user input like age, gender, experience, etc."

    def _run(self, query: str):
        # Expected format: 32,f,7,PhD,8,9
        values = query.split(",")
        return predict_single_instance(values, file_num="1")  # default to dataset 1

    def _arun(self, query: str):
        raise NotImplementedError("Async not supported.")
