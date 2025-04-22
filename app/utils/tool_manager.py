from app.langchain_tools.dataset_tool import DatasetTool
from app.langchain_tools.shap_dataset_tool import SHAPDatasetTool
from app.langchain_tools.single_prediction_tool import SinglePredictionTool
from app.langchain_tools.single_shap_tool import SingleSHAPTool
from app.langchain_tools.fairness_tool import FairnessTool
from app.langchain_tools.mitigation_tool import MitigationTool

TOOLS = [
    DatasetTool(),
    SHAPDatasetTool(),
    SinglePredictionTool(),
    SingleSHAPTool(),
    FairnessTool(),
    MitigationTool()
]