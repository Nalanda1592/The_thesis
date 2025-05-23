import numpy as np
import pandas as pd
import os
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from fairlearn.metrics import equalized_odds_ratio,demographic_parity_ratio
from fairlearn.adversarial import AdversarialFairnessClassifier
from fairlearn.metrics import MetricFrame,equalized_odds_ratio,demographic_parity_ratio,selection_rate,count
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from numpy import number
from imblearn.over_sampling import SMOTE
import collections
from sklearn.compose import ColumnTransformer
import pickle


pd.set_option('display.max_columns', None)


def fairness_dataset_gender(df,rank_0,rank_1,rank_2,rank_3):

    sens_fe=df['Gender']
    average_gen=df['PredictedSalary'].mean()

    rank_0_sens_fe=rank_0['Gender']
    rank_1_sens_fe=rank_1['Gender']
    rank_2_sens_fe=rank_2['Gender']
    rank_3_sens_fe=rank_3['Gender']

    average_0=rank_0['PredictedSalary'].mean()
    average_1=rank_1['PredictedSalary'].mean()
    average_2=rank_2['PredictedSalary'].mean()
    average_3=rank_3['PredictedSalary'].mean()

    #for overall dataset
    new_pred_labels=(df['PredictedSalary'] > average_gen).astype(int)
    salary_array=np.array(df['Actual Salary'],dtype='float64')
    salary_array_labels=(salary_array > average_gen).astype(int)
    pos_label = salary_array_labels[-1]
    gender_array=np.array(sens_fe,dtype='str')
    print('actual salary is ', salary_array_labels[:10])
    print('predictions are ', new_pred_labels[:10])
    print('sensitive feature is ', gender_array[:10])

    m_dpr = demographic_parity_ratio(salary_array_labels, new_pred_labels, sensitive_features=gender_array)
    m_eqo = equalized_odds_ratio(salary_array_labels, new_pred_labels, sensitive_features=gender_array)
    print(f'Value of demographic parity ratio for overall dataset: {round(m_dpr, 2)}')
    print(f'Value of equal odds ratio: {round(m_eqo, 2)}')

    mf = MetricFrame(metrics={"precision_score": precision_score,"recall_score":recall_score,"f1_score":f1_score, "selection_rate": selection_rate, "count of instances":count},y_true=salary_array_labels==pos_label,y_pred=new_pred_labels==pos_label,sensitive_features=sens_fe)

    result_df=mf.by_group

    #for class 0
    new_pred_labels_0=(rank_0['PredictedSalary'] > average_0).astype(int)
    salary_array_0=np.array(rank_0['Actual Salary'],dtype='float64')
    salary_array_labels_0=(salary_array_0 > average_0).astype(int)
    gender_array_0=np.array(rank_0_sens_fe,dtype='str')
    print('actual salary is ', salary_array_labels_0[:10])
    print('predictions are ', new_pred_labels_0[:10])
    print('sensitive feature is ', gender_array_0[:10])

    m_dpr_0 = demographic_parity_ratio(salary_array_labels_0, new_pred_labels_0, sensitive_features=gender_array_0)
    m_eqo_0 = equalized_odds_ratio(salary_array_labels_0, new_pred_labels_0, sensitive_features=gender_array_0)
    print(f'Value of demographic parity ratio for 0: {round(m_dpr_0, 2)}')
    print(f'Value of equal odds ratio: {round(m_eqo_0, 2)}')
    

    #for class 1
    new_pred_labels_1=(rank_1['PredictedSalary'] > average_1).astype(int)
    salary_array_1=np.array(rank_1['Actual Salary'],dtype='float64')
    salary_array_labels_1=(salary_array_1 > average_1).astype(int)
    gender_array_1=np.array(rank_1_sens_fe,dtype='str')
    print('actual salary is ', salary_array_labels_1[:10])
    print('predictions are ', new_pred_labels_1[:10])
    print('sensitive feature is ', gender_array_1[:10])

    m_dpr_1 = demographic_parity_ratio(salary_array_labels_1, new_pred_labels_1, sensitive_features=gender_array_1)
    m_eqo_1 = equalized_odds_ratio(salary_array_labels_1, new_pred_labels_1, sensitive_features=gender_array_1)
    print(f'Value of demographic parity ratio for 1: {round(m_dpr_1, 2)}')
    print(f'Value of equal odds ratio: {round(m_eqo_1, 2)}')

    #for class 2
    new_pred_labels_2=(rank_2['PredictedSalary'] > average_2).astype(int)
    salary_array_2=np.array(rank_2['Actual Salary'],dtype='float64')
    salary_array_labels_2=(salary_array_2 > average_2).astype(int)
    gender_array_2=np.array(rank_2_sens_fe,dtype='str')
    print('actual salary is ', salary_array_labels_2[:10])
    print('predictions are ', new_pred_labels_2[:10])
    print('sensitive feature is ', gender_array_2[:10])

    m_dpr_2 = demographic_parity_ratio(salary_array_labels_2, new_pred_labels_2, sensitive_features=gender_array_2)
    m_eqo_2 = equalized_odds_ratio(salary_array_labels_2, new_pred_labels_2, sensitive_features=gender_array_2)
    print(f'Value of demographic parity ratio for 2: {round(m_dpr_2, 2)}')
    print(f'Value of equal odds ratio: {round(m_eqo_2, 2)}')

    #for class 3
    new_pred_labels_3=(rank_3['PredictedSalary'] > average_3).astype(int)
    salary_array_3=np.array(rank_3['Actual Salary'],dtype='float64')
    salary_array_labels_3=(salary_array_3 > average_3).astype(int)
    gender_array_3=np.array(rank_3_sens_fe,dtype='str')
    print('actual salary is ', salary_array_labels_3[:10])
    print('predictions are ', new_pred_labels_3[:10])
    print('sensitive feature is ', gender_array_3[:10])

    m_dpr_3 = demographic_parity_ratio(salary_array_labels_3, new_pred_labels_3, sensitive_features=gender_array_3)
    m_eqo_3 = equalized_odds_ratio(salary_array_labels_3, new_pred_labels_3, sensitive_features=gender_array_3)
    print(f'Value of demographic parity ratio for 3: {round(m_dpr_3, 2)}')
    print(f'Value of equal odds ratio: {round(m_eqo_3, 2)}')

    context_subset_0="Range_0 data: max_predicted_salary="+str(rank_0['PredictedSalary'].max())+", min_predicted_salary="+str(rank_0['PredictedSalary'].min())+"demographic parity ratio="+str(m_dpr_0)+", Equalized Odds ratio="+str(m_eqo_0)+"."
    context_subset_1="Range_1 data: max_predicted_salary="+str(rank_1['PredictedSalary'].max())+", min_predicted_salary="+str(rank_1['PredictedSalary'].min())+"demographic parity ratio="+str(m_dpr_1)+", Equalized Odds ratio="+str(m_eqo_1)+"."
    context_subset_2="Range_2 data: max_predicted_salary="+str(rank_2['PredictedSalary'].max())+", min_predicted_salary="+str(rank_2['PredictedSalary'].min())+"demographic parity ratio="+str(m_dpr_2)+", Equalized Odds ratio="+str(m_eqo_2)+"."
    context_subset_3="Range_3 data: max_predicted_salary="+str(rank_3['PredictedSalary'].max())+", min_predicted_salary="+str(rank_3['PredictedSalary'].min())+"demographic parity ratio="+str(m_dpr_3)+", Equalized Odds ratio="+str(m_eqo_3)+"."
    context_subset_overall="Overall database metrices: max_predicted_salary="+str(df['PredictedSalary'].max())+", min_predicted_salary="+str(df['PredictedSalary'].min())+"demographic parity ratio="+str(m_dpr)+", Equalized Odds ratio="+str(m_eqo)+"."


    context="The dataset has been divided into 4 quartile ranges."+context_subset_0+context_subset_1+context_subset_2+context_subset_3+context_subset_overall

    data_quantile = [[0, m_dpr_0,m_eqo_0], [1, m_dpr_1,m_eqo_1], [2, m_dpr_2,m_eqo_2],[3, m_dpr_3,m_eqo_3]]

    data_overall = [['overall database',m_dpr,m_eqo]]


    quantile_metrics_df=pd.DataFrame(data_quantile, columns=['QuantileRank', 'Demographic Parity Ratio','Equalized Odds Ratio'])
    overall_metrics_df=pd.DataFrame(data_overall, columns=['Database', 'Demographic Parity Ratio','Equalized Odds Ratio'])

    return context,quantile_metrics_df,overall_metrics_df,result_df


def unfairness_mitigation_fairlearn(data):


    target=['Salary']
    predictors=['Age', 'Gender', 'Experience', 'Education', 'Interview Score', 'Test Score']

    X=data[predictors]
    y=data[target]

    z = X["Gender"]
    average_gen=data['Salary'].mean()



    #for overall dataset
    salary_array=np.array(data["Salary"],dtype='float64')
    salary_array_labels=(salary_array > average_gen).astype(int)
    pos_label = salary_array_labels[salary_array.argmax()]
    print('actual salary is ', salary_array_labels[:10])

    ct = make_column_transformer(
    (
        Pipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),
                ("normalizer", StandardScaler()),
            ]
        ),
        make_column_selector(dtype_include=number),
    ),
    (
        Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(drop="if_binary", sparse=False)),
            ]
        ),
        make_column_selector(dtype_include="category"),
    ),
)


    #new_feature_names=['Gender_f', 'Gender_m', 'Education_Masters', 'Education_PhD', 'Age', 'Experience', 'Interview_Score', 'Test_Score']
    
    #X_encoded_df=pd.DataFrame(data_input_encoded, columns=new_feature_names)

    X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(X, salary_array_labels, z, test_size=0.3, random_state=12345,stratify=salary_array_labels)

    X_prep_train = ct.fit_transform(X_train) # Only fit on training data!
    X_prep_test = ct.transform(X_test)

    result_df=None
    m_dpr=0.0
    m_eqo=0.0

    mitigator = AdversarialFairnessClassifier(
    backend="tensorflow",
    predictor_model=[50, "leaky_relu"],
    adversary_model=[3, "leaky_relu"],
    batch_size=2 ** 8,
    progress_updates=0.5,
    random_state=123,)

    mitigator.fit(X_prep_train, Y_train, sensitive_features=Z_train)

    predictions = mitigator.predict(X_prep_test)
    print(predictions[:10])

    print("sensitive feature shape:",Z_test.shape)
    print("predictions shape:",predictions.shape)
    print("test data shape:",salary_array_labels.shape)


    m_dpr = demographic_parity_ratio(Y_test, predictions, sensitive_features=Z_test)
    print(f'Value of demographic parity ratio for overall dataset: {round(m_dpr, 2)}')
    m_eqo = equalized_odds_ratio(Y_test, predictions, sensitive_features=Z_test)
    print(f'Value of equalized odds ratio: {round(m_eqo, 2)}')
    {"precision_score": precision_score,"recall_score":recall_score,"f1_score":f1_score, "selection_rate": selection_rate, "count of instances":count}
    mf = MetricFrame(metrics={"mitigated precision_score(Adversarial Mitigation Technique)": precision_score, "mitigated recall_score(Adversarial Mitigation Technique)":recall_score,"mitigated f1_score(Adversarial Mitigation Technique)":f1_score,"mitigated selection_rate(Adversarial Mitigation Technique)": selection_rate,"present count of instances":count},y_true=Y_test==pos_label,y_pred=predictions==pos_label,sensitive_features=Z_test)

    result_df=mf.by_group
    #result_df["mitigated demographic parity ratio"]=m_dpr
    #result_df["mitigated equalized odds ratio"]=m_eqo

    print(result_df)

    
    return result_df,m_dpr,m_eqo

def load_model():

 
    folder_name="Datensatz2_results"

    #current_dir = os.getcwd()

    p_folder="data/total_prediction_results/"+folder_name+"/"

    filename = p_folder +  'neural_model.sav'
    with open(filename, 'rb') as f:
        model = pickle.load(f)

    return model

def unfairness_mitigation_SMOTE(data):

    target=['Gender']
    predictors=['Age', 'Experience', 'Education', 'Interview Score', 'Test Score', 'Salary']
    X=data[predictors]
    y=data[target] 

    # Encode categorical variables
    column_transformer = ColumnTransformer(
        transformers=[

            ("hotencoder", OneHotEncoder(handle_unknown="ignore"), ["Education"])
        ],
        remainder="passthrough"
    )

    X_encoded = column_transformer.fit_transform(X)

    # Data of Gender is converted into Binary Data
    df_one = pd.get_dummies(y,dtype=float)

    #print(df_one.head(5))
 
    # We want Male =0 and Female =1 So we drop Male column here
    df_two = df_one.drop(['Gender_m'], axis=1)
 
    # Rename the Column
    df_two = df_two.rename(columns={'Gender_f': "Gender"})
    y_modified=df_two["Gender"]
   
    sme = SMOTE(random_state=42)
    X_res, y_res = sme.fit_resample(X_encoded, y_modified)

    result=collections.Counter(y_res)
    #print("after resampling: ",result)

    y_res_df=pd.DataFrame(y_res, columns=['Gender'])

    new_feature_column_names=['Education_Masters', 'Education_PhD', 'Age', 'Experience', 'Interview Score', 'Test Score', 'Salary']
    X_res_df=pd.DataFrame(X_res, columns=new_feature_column_names)

    new_y_df=pd.get_dummies(y_res_df['Gender'],dtype=float)
    #print(new_y_df.head(5))

    new_y_df.columns=['Gender_Male', 'Gender_Female']
    #print(new_y_df.head(5))

    resultant_df = pd.concat((new_y_df, X_res_df), axis=1)
    #print(resultant_df.head(5))

    new_predictors=['Gender_Male','Gender_Female','Education_Masters', 'Education_PhD', 'Age', 'Experience', 'Interview Score', 'Test Score']
    new_target=['Salary']

    X_final=resultant_df[new_predictors]
    y_final=resultant_df[new_target]

    mapping = {0.0: 'm', 1.0: 'f'}
    for_gender_df=resultant_df.replace({'Gender_Female': mapping})
    for_gender_df=for_gender_df.rename(columns={'Gender_Female': "Gender"})

    X_train, X_test, y_train, y_test, gender_train, gender_test = train_test_split(X_final, y_final, for_gender_df, test_size=0.3, random_state=42)

    model = load_model()

    model.fit(X_train, y_train ,batch_size = 20, epochs = 50, verbose=1)
    predictions=model.predict(X_test)

    testingData=pd.DataFrame(data=X_test, columns=predictors)
    testingData['Actual Salary']=y_test
    testingData['PredictedSalary']=predictions
    testingData = testingData.reset_index(drop=True)

    sens_fe=gender_test['Gender']
    average_gen=testingData['PredictedSalary'].mean()

     #for overall dataset
    new_pred_labels=(testingData['PredictedSalary'] > average_gen).astype(int)
    salary_array=np.array(testingData['Actual Salary'],dtype='float64')
    salary_array_labels=(salary_array > average_gen).astype(int)
    pos_label = salary_array_labels[salary_array.argmax()]
    gender_array=np.array(sens_fe,dtype='str')
    #print('actual salaries are ', salary_array_labels[:10])
    #print('predictions are ', new_pred_labels[:10])
    #print('sensitive features are ', gender_array[:10])

    m_dpr = demographic_parity_ratio(salary_array_labels, new_pred_labels, sensitive_features=gender_array)
    m_eqo = equalized_odds_ratio(salary_array_labels, new_pred_labels, sensitive_features=gender_array)
    print(f'Value of demographic parity ratio for overall dataset: {round(m_dpr, 2)}')
    print(f'Value of equal odds ratio: {round(m_eqo, 2)}')

    mf = MetricFrame(metrics={"precision_score": precision_score,"recall_score":recall_score,"f1_score":f1_score, "selection_rate": selection_rate, "count of instances":count},y_true=salary_array_labels==pos_label,y_pred=new_pred_labels==pos_label,sensitive_features=sens_fe)

    result_df=mf.by_group

    print(result_df)


