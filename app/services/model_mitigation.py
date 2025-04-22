import pandas as pd
from app.utils.config import *
from app.services import model_trainer
from app.services import model_utils as mcutil


def run_fairness_mitigation(file_num: str):
    data = model_trainer.read_data(file_num)
    result = mcutil.run_dataset_specific_mitigation(file_num, data)

    folder = PREDICTION_RESULTS_PATH / f"Datensatz{file_num}_results" / "fairness_measures"
    folder.mkdir(parents=True, exist_ok=True)

    if file_num == "1":
        result_df, dpr, eqo = result
        main_df = pd.read_csv(folder / FAIRLEARN_FILENAMES['overall'])
        side_df = pd.read_csv(folder / FAIRLEARN_FILENAMES['metrics'], index_col=0)
        main_df["Demographic Parity Ratio after Adversarial Mitigation Technique"] = dpr
        main_df["Equalized Odds Ratio after Adversarial Mitigation Technique"] = eqo
        combined = result_df.join(side_df)
        combined.to_csv(folder / FAIRLEARN_FILENAMES['mitigation_1'], index=True)
        main_df.to_csv(folder / FAIRLEARN_FILENAMES['mitigation_2'], index=False)
        return combined

    elif file_num == "2":
        smote_result = result
        smote_result.to_csv(folder / FAIRLEARN_FILENAMES['mitigation_1'], index=False)
        return smote_result

    else:
        raise ValueError("Invalid file number")