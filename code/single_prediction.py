import pickle
import pandas as pd
import os
import numpy as np
from pyparsing import Path



#This class handles everything for single prediction


def load_model(parent_dir,file_num):

    if(file_num=="1"):
         folder_name="Datensatz1_results"
    elif(file_num=="2"):
         folder_name="Datensatz2_results"

    p_folder="data/total_prediction_results/"+folder_name+"/"

    filename = p_folder +  'neural_model.sav'
    with open(filename, 'rb') as f:
        model = pickle.load(f)

    return model

def load_column_transformer(parent_dir):

    filename = "data/" + 'column_transformer.pkl'
    with open(filename, 'rb') as f:
        column_transformer = pickle.load(f)

    return column_transformer

def load_explainer(parent_dir,file_num):

    p=None
    if(file_num=="1"):
        p="data/total_prediction_results/Datensatz1_results/explanations/"
    elif(file_num=="2"):
        p="data/total_prediction_results/Datensatz2_results/explanations/"

    filename = p + 'explainer.pkl'
    with open(filename, 'rb') as f:
        explainer = pickle.load(f)

    return explainer

def predict_salary(age, gender, experience, education, interview_score, test_score,file_num):

    current_dir = os.getcwd()
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

    model = load_model(parent_dir,file_num)
    column_transformer = load_column_transformer(parent_dir)
    explainer = load_explainer(parent_dir,file_num)


    try:
        data_input = pd.DataFrame([[int(age), gender, int(experience), education, int(interview_score), int(test_score)]], columns=["Age", "Gender", "Experience", "Education", "Interview Score", "Test Score"])
        data_input_encoded = column_transformer.transform(data_input)
        salary_pred = model.predict(data_input_encoded)[0]
        data_input["predicted salary"]=salary_pred
        result=model(data_input_encoded)
        print("result prediction:",result)
        print(salary_pred)

        shap_feature_names=['Shap_value_of_Gender_female', 'Shap_value_of_Gender_male', 'Shap_value_of_Education_Masters', 'Shap_value_of_Education_PhD', 'Shap_value_of_Age', 'Shap_value_of_Experience', 'Shap_value_of_Interview Score', 'Shap_value_of_Test Score']

        shap_values = explainer.shap_values(data_input_encoded)
        shap_value_array = np.array(shap_values)
        print('shape of shapley value array', shap_value_array.shape)
        swapped_array = np.swapaxes(shap_value_array,1,2)
        print('shape of swapped shapley value array', swapped_array.shape)
        shap_value_dataframe = pd.DataFrame(swapped_array[0], index = shap_feature_names)
        shap_val_axes_change = shap_value_dataframe.T
        shap_val_axes_change = shap_val_axes_change.reset_index(drop=True)
        final_result = data_input.join(shap_val_axes_change)
        print(final_result)

        p=None
        if(file_num=="1"):
            p="data/single_prediction_results/Datensatz1_single_results/"
        elif(file_num=="2"):
            p="data/single_prediction_results/Datensatz2_single_results/"

        filename='total_result.csv' 
        final_result.to_csv(Path(p+filename), index=False)

        return salary_pred
    
    except Exception as e:
        print(e)
        return 0
    
