import pandas as pd
from app.utils.config import *
from app.services import model_utils as mcutil


def run_fairness_evaluation(file_num: str):
    folder = PREDICTION_RESULTS_PATH / f"Datensatz{file_num}_results"
    df = pd.read_csv(folder / PREDICTION_CSV_FILENAME)

    df['QuantileRank'] = pd.qcut(df['PredictedSalary'], 4, labels=False)
    ranks = [df[df['QuantileRank'] == i] for i in range(4)]

    female_avg = df[df['Gender'] == 'f']['PredictedSalary'].mean()
    male_avg = df[df['Gender'] == 'm']['PredictedSalary'].mean()

    context, quantile_df, overall_df, group_metrics = mcutil.fairness_dataset_gender(df, *ranks)

    fair_dir = folder / "fairness_measures"
    fair_dir.mkdir(exist_ok=True)

    overall_df.to_csv(fair_dir / FAIRLEARN_FILENAMES['overall'], index=False)
    quantile_df.to_csv(fair_dir / FAIRLEARN_FILENAMES['quantile'], index=False)
    df.to_csv(fair_dir / FAIRLEARN_FILENAMES['annotated'], index=False)
    pd.DataFrame([[context]], columns=["main context"]).to_csv(fair_dir / FAIRLEARN_FILENAMES['narrative'], index=False)
    group_metrics.to_csv(fair_dir / FAIRLEARN_FILENAMES['metrics'], index=True)

    return overall_df