import pandas as pd
import os
import model_creation_utils as mcutil
from pyparsing import Path
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import pickle
import numpy as np
import shap
from fairlearn.metrics import equalized_odds_ratio,demographic_parity_ratio



# This class handles everything for the model creation and multiple predictions on test data


def read_data(file_num):

    data = None

    current_dir = os.getcwd()
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

    if file_num == "1":
        data = pd.read_csv("/data/" + "Datensatz 1.csv")
    elif file_num == "2":
        data = pd.read_csv("/data/" + "Datensatz 2.csv")
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
        

    # save the transformer

    pickle.dump(column_transformer, open("/data/" + 'column_transformer.pkl', 'wb'))
        
    # save the results

    if(file_num=="1"):
         folder_name="Datensatz1_results"
    elif(file_num=="2"):
         folder_name="Datensatz2_results"

    p_folder="/data/total_prediction_results/"+folder_name+"/"
    filename='total_database_prediction.csv' 
    testingData.to_csv(Path(p_folder+filename), index=False)


    # save the model

    pickle.dump(model, open(p_folder + 'neural_model.sav', 'wb'))

    return X_train,X_test,y_train,model,tData


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
    X_train_summary=shap.sample(X_train, 100)
    X_test_summary=shap.sample(X_test, 100)
    explainer = shap.KernelExplainer(model.predict, X_train_summary)
    shap_values = explainer.shap_values(X_test_summary)
    shap_value_array = np.array(shap_values)
    print('shape of shapley value array', shap_value_array.shape)
    swapped_array = np.swapaxes(shap_value_array,1,2)
    print('shape of swapped shapley value array', swapped_array.shape)
    shap_value_dataframe = pd.DataFrame(swapped_array[0], index = shap_feature_names)
    shap_val_axes_change = shap_value_dataframe.T
    shap_val_axes_change = shap_val_axes_change.reset_index(drop=True)
    td=testingData.head(100)
    final_result = td.join(shap_val_axes_change)
    print(final_result.head(10))

    #final_result.to_csv(r'/Users/nalanda1592/VSProjects/onlypyton/Result.csv')

    
    p=None
    if(file_num=="1"):
        p="/data/total_prediction_results/Datensatz1_results/explanations/"
    elif(file_num=="2"):
        p="/data/total_prediction_results/Datensatz2_results/explanations/"
    
    #filename='shap_result.csv' 
    #final_result.to_csv(Path(p+filename), index=False)

    # save the explainer

    pickle.dump(explainer, open(p + 'explainer.pkl', 'wb'))


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

def fairness_model_creation(file_num):

    current_dir = os.getcwd()
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

    p=None
    if(file_num=="1"):
        p="/data/total_prediction_results/Datensatz1_results/"
    elif(file_num=="2"):
        p="/data/total_prediction_results/Datensatz2_results/"

    df=pd.read_csv(p+'total_database_prediction.csv')

    #pred_sal=df['PredictedSalary']
    #actual_sal=df['Actual Salary']
    

    df['QuantileRank']= pd.qcut(df['PredictedSalary'], q = 4, labels = False)

    rank=df['QuantileRank']
    

    rank_0=df[rank==0]
    rank_1=df[rank==1]
    rank_2=df[rank==2]
    rank_3=df[rank==3]

    long_context_subset=" The interpretation of the values depend on the definitions as follows. Demographic Parity: It measures the ML model's ability to make prediction such that they are independent of the influence by sensitive groups. Equalized odds: It also ensures that ML model's predictions are independent of sensitive groups. It's more strict than Demographic parity by ensuring all groups in the dataset have same true positive rates and false positive rates. Equal Opportunity: It's similar to equalized odds but applies only to positive instances, i.e. Y=1. Demographic parity ratio: Ratio of selection rates between smallest and largest groups. Return type is a decimal value. A ratio of 1 means all groups have same selection rate. Equalized odds ratio: The equalized odds ratio of 1 means that all groups have the same true positive, true negative, false positive, and false negative rates. "

    gender_female=df[df['Gender']=='f']
    gender_female_length=len(gender_female.axes[0])
    gender_male=df[df['Gender']=='m']
    gender_male_length=len(gender_male.axes[0])

    avg_female_sal=gender_female['PredictedSalary'].mean()
    avg_male_sal=gender_male['PredictedSalary'].mean()

    context_gender,quantile_metrics_df,overall_metrics_df,result_df=mcutil.fairness_dataset_gender(df,rank_0,rank_1,rank_2,rank_3)
    long_context="Fairlearn is a libarary to compute fairness of ML models. It has various fairness metrics. "+ context_gender +long_context_subset+" The sensitive feature considered here is the gender(male,female). Average salary for a feamle in the database is "+str(avg_female_sal)+" and for a male is "+str(avg_male_sal)+". Number of females in the database is "+str(gender_female_length)+" and number of males in the database is "+str(gender_male_length)+"."

    long_input=[['context',long_context]]

    main_df=pd.DataFrame(long_input,columns=['overall context','main context'])

    #with open(p+'/fairness_measures/fairlearn_doc.csv', 'w') as f:
    #    f.write(long_context)

    filename='Fairlearn_results_1.csv' 
    overall_metrics_df.to_csv(Path(p+'/fairness_measures/'+filename), index=False)

    filename='Fairlearn_results_2.csv' 
    quantile_metrics_df.to_csv(Path(p+'/fairness_measures/'+filename), index=False)

    filename='Fairlearn_results_3.csv' 
    df.to_csv(Path(p+'/fairness_measures/'+filename), index=False)

    filename='Fairlearn_results_4.csv' 
    main_df.to_csv(Path(p+'/fairness_measures/'+filename), index=False)

    filename='Fairlearn_results_5.csv' 
    result_df.to_csv(Path(p+'/fairness_measures/'+filename), index=True)

    return overall_metrics_df

def run_model_generator_for_mitigation(file_num):

    data, parent_dir = read_data(file_num)
    result,m_dpr,m_eqo = mcutil.unfairness_mitigation_fairlearn(data)

    p=None
    main_df=None
    if(file_num=="1"):
        p="/data/total_prediction_results/Datensatz1_results/fairness_measures/"
        main_df=pd.read_csv("/data/total_prediction_results/Datensatz1_results/fairness_measures/Fairlearn_results_1.csv")
        side_df=pd.read_csv("/data/total_prediction_results/Datensatz1_results/fairness_measures/Fairlearn_results_5.csv",index_col=0)
    elif(file_num=="2"):
        p="/data/total_prediction_results/Datensatz2_results/fairness_measures/"
        main_df=pd.read_csv("/data/total_prediction_results/Datensatz2_results/fairness_measures/Fairlearn_results_1.csv")
        side_df=pd.read_csv("/data/total_prediction_results/Datensatz2_results/fairness_measures/Fairlearn_results_5.csv",index_col=0)

    main_df["Demographic Parity Ratio after mitigation"]=m_dpr
    main_df["Equalized Odds Ratio after mitigation"]=m_eqo

    #selec_accu=pd.concat([result.set_index('Gender'),side_df.set_index('Gender')], axis=1, join='inner')
    selec_accu=result.join(side_df)

    filename='unfairness_mitigation_result_1.csv'
    selec_accu.to_csv(Path(p+filename), index=False)

    filename='unfairness_mitigation_result_2.csv'
    main_df.to_csv(Path(p+filename), index=False)
    
    return selec_accu
        
