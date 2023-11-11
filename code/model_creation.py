import pandas as pd
import os
from pyparsing import Path
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import pickle
import numpy as np
import functools
import shap
from langchain.agents import create_csv_agent 
from langchain.chat_models import ChatOpenAI
import dotenv
from IPython.display import display,Markdown
import dalex as dx
from io import StringIO 
import sys
from fairlearn.metrics import equalized_odds_ratio,demographic_parity_ratio,MetricFrame



# This class handles everything for the model creation and multiple predictions on test data


def read_data(file_num):

    data = None

    current_dir = os.getcwd()
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

    if file_num == "1":
        data = pd.read_csv(parent_dir + "/data/" + "Datensatz 1.csv")
    elif file_num == "2":
        data = pd.read_csv(parent_dir + "/data/" + "Datensatz 2.csv")
    elif file_num == "3":
        data = pd.read_csv(parent_dir + "/data/" + "Datensatz 3.csv")
    #else:
    #    data = pd.read_csv(parent_dir + "/data/" + "Datensatz 1.csv")

    return data, parent_dir

def convert_data_to_text_for_explanations(the_data,path):
    filename='shap_result.txt' 
    the_data.to_csv(Path(path+filename), header=None, index=None, sep=' ', mode='a')                    


def create_model(data, parent_dir,file_num):

    # Encode categorical variables
    column_transformer = ColumnTransformer(
        transformers=[

            ("hotencoder", OneHotEncoder(handle_unknown="ignore"), ["Gender","Education"])
        ],
        remainder="passthrough"
    )

    target=['Salary']
    predictors=['Age', 'Gender', 'Experience', 'Education', 'Interview Score', 'Test Score']

    X=data[predictors]
    y=data[target]

    X_encoded = column_transformer.fit_transform(X)


    new_feature_names=['Gender_Female', 'Gender_Male', 'Education_Masters', 'Education_PhD', 'Age', 'Experience', 'Interview_Score', 'Test_Score']
    
    X_encoded_df=pd.DataFrame(X_encoded, columns=new_feature_names)

    # Split the data into training and testing set
    X_train, X_test, y_train, y_test, X_real_train, X_real_test = train_test_split(X_encoded_df, y, X, test_size=0.3, random_state=42)
    

    # create ANN model
    model = Sequential()
 
    # Defining the Input layer and FIRST hidden layer, both are same!
    model.add(Dense(units=7, input_dim=8, kernel_initializer='normal', activation='relu'))
 
    # Defining the Second layer of the model
    model.add(Dense(units=7, kernel_initializer='normal', activation='relu'))
 
    # The output neuron is a single fully connected node 
    # Since we will be predicting a single number
    model.add(Dense(1, kernel_initializer='normal'))
 
    # Compiling the model
    model.compile(loss='mean_squared_error', optimizer='adam')
 
    # Fitting the ANN to the Training set
    model.fit(X_train, y_train ,batch_size = 20, epochs = 50, verbose=1)


    # Generating Predictions on testing data
    predictions=model.predict(X_test)
    print('prediction shape', predictions.shape)

    testingData=pd.DataFrame(data=X_real_test, columns=predictors)
    testingData['Actual Salary']=y_test
    testingData['PredictedSalary']=predictions
    testingData = testingData.reset_index(drop=True)
    tData=testingData
    testingData=testingData.sort_values('PredictedSalary')
    print(testingData.head())


    # save the model

    pickle.dump(model, open(parent_dir + "/data/" + 'neural_model.sav', 'wb'))
        

    # save the transformer

    pickle.dump(column_transformer, open(parent_dir + "/data/" + 'column_transformer.pkl', 'wb'))
        
    # save the results

    if(file_num=="1"):
         folder_name="Datensatz1_results"
    elif(file_num=="2"):
         folder_name="Datensatz2_results"
    else: folder_name="Datensatz3_results"

    p_folder=parent_dir + "/data/total_prediction_results/"+folder_name+"/"
    filename='total_database_prediction.csv' 
    testingData.to_csv(Path(p_folder+filename), index=False)

    return X_train,X_test,y_train,model,tData


def fairness_model_creation(file_num):

    current_dir = os.getcwd()
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

    p=None
    if(file_num=="1"):
        p=parent_dir + "/data/total_prediction_results/Datensatz1_results/"
    elif(file_num=="2"):
        p=parent_dir + "/data/total_prediction_results/Datensatz2_results/"
    else:
        p=parent_dir + "/data/total_prediction_results/Datensatz3_results/"

    df=pd.read_csv(p+'total_database_prediction.csv')

    #pred_sal=df['PredictedSalary']
    #actual_sal=df['Actual Salary']
    

    df['QuantileRank']= pd.qcut(df['PredictedSalary'], q = 4, labels = False)

    rank=df['QuantileRank']
    

    rank_0=df[rank==0]
    rank_1=df[rank==1]
    rank_2=df[rank==2]
    rank_3=df[rank==3]

    long_context_subset="The interpretation of the values depend on the definitions as follows. Demographic Parity: It measures the ML model's ability to make prediction such that they are independent of the influence by sensitive groups. Equalized odds: It also ensures that ML model's predictions are independent of sensitive groups. It's more strict than Demographic parity by ensuring all groups in the dataset have same true positive rates and false positive rates. Equal Opportunity: It's similar to equalized odds but applies only to positive instances, i.e. Y=1. Demographic parity ratio: Ratio of selection rates between smallest and largest groups. Return type is a decimal value. A ratio of 1 means all groups have same selection rate. Equalized odds ratio: The equalized odds ratio of 1 means that all groups have the same true positive, true negative, false positive, and false negative rates."

    main_context=None

    if(file_num=='1'):

        gender_female=df[df['Gender']=='f']
        gender_male=df[df['Gender']=='m']

        avg_female_sal=gender_female['PredictedSalary'].mean()
        avg_male_sal=gender_male['PredictedSalary'].mean()

        context_gender,metrics_df,df=fairness_dataset_gender(df,rank_0,rank_1,rank_2,rank_3)
        long_context="Fairlearn is a libarary to compute fairness of ML models. It has various fairness metrics."+ context_gender +long_context_subset+"The sensitive feature considered here is the gender(male,female). Average salary for a feamle in the databse is "+str(avg_female_sal)+" and average salary for a male is "+str(avg_male_sal)+"."

        with open(p+'/fairness_measures/fairlearn_doc.csv', 'w') as f:
            f.write(long_context)

        filename='Fairlearn_results_1.csv' 
        df.to_csv(Path(p+'/fairness_measures/'+filename), index=False)

        filename='Fairlearn_results_2.csv' 
        metrics_df.to_csv(Path(p+'/fairness_measures/'+filename), index=False)

    return 0

def show_results(testingData):
     show_results(testingData)


def run_model_generator_for_prediction(file_num):

    data, parent_dir = read_data(file_num)
    X_train,X_test,y_train,model,testingData = create_model(data, parent_dir,file_num)
    #path_for_text_file=convert_data_to_text_for_tables(testingData,file_num,parent_dir)
    
    return testingData


def run_model_generator_for_explanation(file_num):
     
    data, parent_dir = read_data(file_num)
    X_train,X_test,y_train,model,testingData = create_model(data, parent_dir,file_num)

    print("initial data", testingData.head(10))

        
    
    shap_feature_names=['Shap_value_of_Gender_female', 'Shap_value_of_Gender_male', 'Shap_value_of_Education_Masters', 'Shap_value_of_Education_PhD', 'Shap_value_of_Age', 'Shap_value_of_Experience', 'Shap_value_of_Interview Score', 'Shap_value_of_Test Score']

    
    # get shap values
    X_train_summary=shap.sample(X_train, 1000)
    X_test_summary=shap.sample(X_test, 1000)
    explainer = shap.KernelExplainer(model.predict, X_train_summary)
    shap_values = explainer.shap_values(X_test_summary)
    shap_value_array = np.array(shap_values)
    print('shape of shapley value array', shap_value_array.shape)
    swapped_array = np.swapaxes(shap_value_array,1,2)
    print('shape of swapped shapley value array', swapped_array.shape)
    shap_value_dataframe = pd.DataFrame(swapped_array[0], index = shap_feature_names)
    shap_val_axes_change = shap_value_dataframe.T
    shap_val_axes_change = shap_val_axes_change.reset_index(drop=True)
    td=testingData.head(1000)
    final_result = td.join(shap_val_axes_change)
    print(final_result.head(10))

    #final_result.to_csv(r'/Users/nalanda1592/VSProjects/onlypyton/Result.csv')

    p=None
    if(file_num=="1"):
        p=parent_dir + "/data/total_prediction_results/Datensatz1_results/explanations/"
    elif(file_num=="2"):
        p=parent_dir + "/data/total_prediction_results/Datensatz2_results/explanations/"
    else:
        p=parent_dir + "/data/total_prediction_results/Datensatz3_results/explanations/"
    
    filename='shap_result.csv' 
    final_result.to_csv(Path(p+filename), index=False)

    #convert_data_to_text_for_explanations(final_result,p)
        
    return final_result.head(10)

    #shap.summary_plot(shap_values, X_test_summary)

    #GPT-4 Interpretations

    #dotenv.load_dotenv('dev.env', override=True)

    #agent  =  create_csv_agent ( ChatOpenAI(temperature = 0, model_name='gpt-4' ) ,   'Result.csv' ,  verbose = True ) 
    #display(Markdown(agent.agent.llm_chain.prompt.template))

    #agent.run('How many rows and columns the data has?')
    #agent.run('m=male,f=female,Ma=Masters.Based on Shapley value concept and summary plot,what is the most important feature in the data?')
    #agent.run('Which applicant has the most experience?')
    #agent.run("Which combination of 2 features yields the highest prediction?")
        

class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout

def run_model_generator_for_fairness_dalex(file_num):
    
    data1, parent_dir1 = read_data(file_num)
    X_train1,X_test1,y_train,model,testingData1 = create_model(data1, parent_dir1,file_num)
    
    current_dir = os.getcwd()
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    
    # Generating fairness

    exp = dx.Explainer(model, X_train1, y_train)

    print(exp.model_performance().result)

    interim_result=exp.model_performance().result
    
    #interim_result.to_csv(r'/Users/nalanda1592/VSProjects/onlypyton/ModelPerformance.csv')
    p=None
    if(file_num=="1"):
        p=parent_dir1 + "/data/total_prediction_results/Datensatz1_results/fairness_measures/"
    elif(file_num=="2"):
        p=parent_dir1 + "/data/total_prediction_results/Datensatz2_results/fairness_measures/"
    else:
        p=parent_dir1 + "/data/total_prediction_results/Datensatz3_results/fairness_measures/"

    filename='model_performance_for_fairness_using_dalex.txt' 
    interim_result.to_csv(Path(p+filename), header=None, index=None, sep=' ', mode='a')

    # array with values like male_old, female_young, etc.
    first=np.where(X_train1['Gender_Female'] ==1.0, 'female', 'male')
    print(first)
    second=np.where(X_train1['Age'] < 25.0, 'young', 'old')
    print(second)

    intermediate=np.add(first.astype(object), '_') 
    protected = np.add(intermediate, second.astype(object))

    privileged = 'male_young'

    fobject = exp.model_fairness(protected = protected, privileged=privileged)

    with Capturing() as output: fobject.fairness_check(epsilon = 0.8) # default epsilon
    df= pd.DataFrame(output, columns=['output of fairness_check'])
    df['details about dalex']="dalex is a fairness library with various metrices.The fairness check method measures bias from different perspectives so that no bias model can go through.0.8 is default epsilon for it.FPR is False Positive Rate.Lower FPR means that the privileged subgroup is getting False Positives more frequently than the unprivileged. Here the priviledged group chosen is 'young male'.There are metrics TPR (True Positive Rate), ACC (Accuracy),PPV (Positive Predictive Value), FPR (False Positive Rate), STP(Statistical parity). The metrics are derived from a confusion matrix for each unprivileged subgroup and then divided by metric values based on the privileged subgroup. Some metrics will not be equal but they will not necessarily exceed the user's threshold.Generally, the score for each subgroup should be close to the score of the privileged subgroup. To put it in a more mathematical perspective the ratios between scores of privileged and unprivileged metrics should be close to 1.The 0.8 threshold has been chosen as our default epsilon as it is the only known tangible threshold for the acceptable amount of discrimination."
    #df.to_csv(r'/Users/nalanda1592/VSProjects/onlypyton/Result_of_function_fairness_check.csv')
    
    filename='result_of_function_fairness_check_using_dalex.txt'
    df.to_csv(Path(p + filename ), header=None, index=None, sep=' ', mode='a')

    # to see all scaled metric values you can run
    interim2=fobject.result
    #interim2.to_csv(r'/Users/nalanda1592/VSProjects/onlypyton/GroupFairnessRegressionResult.csv')

    filename='group_fairness_regression_result_using_dalex.txt'
    interim2.to_csv(Path(p + filename ), header=None, index=None, sep=' ', mode='a')

    #GPT-4 Interpretations

    #dotenv.load_dotenv('dev.env', override=True)

    #agent  =  create_csv_agent ( ChatOpenAI(temperature = 0, model_name='gpt-4' ) ,   ['ModelPerformance.csv' , 'Result_of_function_fairness_check.csv','GroupFairnessRegressionResult.csv'] , verbose = True ) 
    #display(Markdown(agent.agent.llm_chain.prompt.template))

    #agent.run('How many rows and columns the data has?')
    #agent.run('m=male,f=female,Ma=Masters.Based on dalex explanation and fairness calculation library, what does the fairness_check method for GroupFairnessRegression signify?')
    #agent.run('m=male,f=female,Ma=Masters.Based on dalex explanation and fairness calculation library,is this model biased on sensitive features from the data given?')
    return 0

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
    gender_array=np.array(sens_fe,dtype='str')
    print('actual salary is ', salary_array_labels[:10])
    print('predictions are ', new_pred_labels[:10])
    print('sensitive feature is ', gender_array[:10])

    m_dpr = demographic_parity_ratio(salary_array_labels, new_pred_labels, sensitive_features=gender_array)
    m_eqo = equalized_odds_ratio(salary_array_labels, new_pred_labels, sensitive_features=gender_array)
    print(f'Value of demographic parity ratio for overall dataset: {round(m_dpr, 2)}')
    print(f'Value of equal odds ratio: {round(m_eqo, 2)}')

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

    data = [[0, m_dpr_0,m_eqo_0], [1, m_dpr_1,m_eqo_1], [2, m_dpr_2,m_eqo_2],[3, m_dpr_3,m_eqo_3]]



    metrics_df=pd.DataFrame(data, columns=['QuantileRank', 'Demographic Parity Ratio','Equalized Odds Ratio'])
    df['Demographic_Parity_Ratio_Overall']=m_dpr
    df['Equalized_Odds_ratio_Overall']=m_eqo

    return context,metrics_df,df