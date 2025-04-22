import numpy as np
import pandas as pd
import collections
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector, make_column_transformer, ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from numpy import number
from imblearn.over_sampling import SMOTE

from fairlearn.metrics import (
    equalized_odds_ratio,
    demographic_parity_ratio,
    MetricFrame,
    selection_rate,
    count,
    precision_score,
    recall_score,
    f1_score,
)
from fairlearn.adversarial import AdversarialFairnessClassifier
from keras.models import load_model

from app.utils.config import *


# === USED IN BOTH FILES ===
def fairness_dataset_gender(df, rank_0, rank_1, rank_2, rank_3):
    sens_fe = df['Gender']
    average_gen = df['PredictedSalary'].mean()

    salary_array = np.array(df['Actual Salary'], dtype='float64')
    salary_array_labels = (salary_array > average_gen).astype(int)
    new_pred_labels = (df['PredictedSalary'] > average_gen).astype(int)
    gender_array = np.array(sens_fe, dtype='str')

    m_dpr = demographic_parity_ratio(salary_array_labels, new_pred_labels, sensitive_features=gender_array)
    m_eqo = equalized_odds_ratio(salary_array_labels, new_pred_labels, sensitive_features=gender_array)

    mf = MetricFrame(
        metrics={"precision_score": precision_score, "recall_score": recall_score, "f1_score": f1_score,
                 "selection_rate": selection_rate, "count of instances": count},
        y_true=salary_array_labels,
        y_pred=new_pred_labels,
        sensitive_features=sens_fe
    )

    result_df = mf.by_group
    quantile_metrics = []
    context_parts = []

    for i, rank_df in enumerate([rank_0, rank_1, rank_2, rank_3]):
        avg_rank = rank_df['PredictedSalary'].mean()
        salary_array_r = np.array(rank_df['Actual Salary'], dtype='float64')
        labels_r = (salary_array_r > avg_rank).astype(int)
        preds_r = (rank_df['PredictedSalary'] > avg_rank).astype(int)
        sens_r = np.array(rank_df['Gender'], dtype='str')

        m_dpr_r = demographic_parity_ratio(labels_r, preds_r, sensitive_features=sens_r)
        m_eqo_r = equalized_odds_ratio(labels_r, preds_r, sensitive_features=sens_r)

        quantile_metrics.append([i, m_dpr_r, m_eqo_r])
        context_parts.append(
            f"Range_{i} max={rank_df['PredictedSalary'].max()}, min={rank_df['PredictedSalary'].min()}, "
            f"DPR={m_dpr_r:.2f}, EQO={m_eqo_r:.2f}."
        )

    quantile_df = pd.DataFrame(quantile_metrics, columns=["QuantileRank", "Demographic Parity Ratio", "Equalized Odds Ratio"])
    overall_df = pd.DataFrame([["overall", m_dpr, m_eqo]],
                              columns=["Database", "Demographic Parity Ratio", "Equalized Odds Ratio"])

    female_avg = df[df['Gender'] == 'f']['PredictedSalary'].mean()
    male_avg = df[df['Gender'] == 'm']['PredictedSalary'].mean()
    female_count = (sens_fe == 'f').sum()
    male_count = (sens_fe == 'm').sum()

    long_context = (
        "Fairlearn is used to evaluate fairness. Demographic Parity ensures predictions are independent of group membership. "
        "Equalized Odds ensures equal TP and FP rates across groups. " +
        " ".join(context_parts) +
        f" Overall DPR={m_dpr:.2f}, EQO={m_eqo:.2f}. Female Avg={female_avg:.2f} ({female_count}), "
        f"Male Avg={male_avg:.2f} ({male_count})."
    )

    return long_context, quantile_df, overall_df, result_df


# === FOR DATASET 1 ===
def unfairness_mitigation_fairlearn(data: pd.DataFrame):
    predictors = ['Age', 'Gender', 'Experience', 'Education', 'Interview Score', 'Test Score']
    X = data[predictors]
    y = (data['Salary'] > data['Salary'].mean()).astype(int)
    Z = X["Gender"]

    X_train, X_test, y_train, y_test, Z_train, Z_test = train_test_split(
        X, y, Z, test_size=0.3, stratify=y, random_state=42
    )

    ct = make_column_transformer(
        (
            Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler())
            ]),
            make_column_selector(dtype_include=number),
        ),
        (
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(drop="if_binary", sparse=False)),
            ]),
            make_column_selector(dtype_include="category"),
        )
    )

    X_train_proc = ct.fit_transform(X_train)
    X_test_proc = ct.transform(X_test)

    clf = AdversarialFairnessClassifier(
        backend="tensorflow",
        predictor_model=[50, "leaky_relu"],
        adversary_model=[3, "leaky_relu"],
        batch_size=256,
        progress_updates=0.5,
        random_state=42
    )

    clf.fit(X_train_proc, y_train, sensitive_features=Z_train)
    predictions = clf.predict(X_test_proc)

    m_dpr = demographic_parity_ratio(y_test, predictions, sensitive_features=Z_test)
    m_eqo = equalized_odds_ratio(y_test, predictions, sensitive_features=Z_test)

    pos_label = y_test.max()
    mf = MetricFrame(
        metrics={
            "mitigated precision": precision_score,
            "mitigated recall": recall_score,
            "mitigated f1": f1_score,
            "mitigated selection_rate": selection_rate,
            "count": count,
        },
        y_true=y_test == pos_label,
        y_pred=predictions == pos_label,
        sensitive_features=Z_test
    )

    return mf.by_group, m_dpr, m_eqo


# === FOR DATASET 2 ===
def unfairness_mitigation_SMOTE(data: pd.DataFrame):
    predictors = ['Age', 'Experience', 'Education', 'Interview Score', 'Test Score', 'Salary']
    X = data[predictors]
    y = data['Gender']

    column_transformer = ColumnTransformer(
        transformers=[("hotencoder", OneHotEncoder(handle_unknown="ignore"), ["Education"])],
        remainder="passthrough"
    )
    X_encoded = column_transformer.fit_transform(X)

    y_dummies = pd.get_dummies(y, dtype=float)
    y_binary = y_dummies['f']  # We use 'f' as positive class

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_encoded, y_binary)

    X_df = pd.DataFrame(X_res, columns=[
        'Education_Masters', 'Education_PhD', 'Age', 'Experience', 'Interview Score', 'Test Score', 'Salary'
    ])
    y_df = pd.DataFrame(y_res, columns=['Gender'])

    final_df = pd.concat([y_df, X_df], axis=1)
    final_df['Gender'] = final_df['Gender'].replace({0.0: 'm', 1.0: 'f'})

    return final_df


# === MAIN WRAPPER FUNCTION ===
def run_dataset_specific_mitigation(file_num: str, data: pd.DataFrame):
    if file_num == "1":
        return unfairness_mitigation_fairlearn(data)
    elif file_num == "2":
        return unfairness_mitigation_SMOTE(data)
    else:
        raise ValueError("Unknown dataset number.")
