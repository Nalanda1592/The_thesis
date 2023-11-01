import pickle
import pandas as pd
import os
import numpy as np
import model_creation as mc
import shap
from langchain.agents import create_csv_agent 
from langchain.chat_models import ChatOpenAI
import dotenv
from IPython.display import display,Markdown


#This class handles everything for single prediction


def load_model(parent_dir):

    filename = parent_dir + "/data/" +  'neural_model.sav'
    with open(filename, 'rb') as f:
        model = pickle.load(f)

    return model

def load_column_transformer(parent_dir):

    filename = parent_dir + "/data/" + 'column_transformer.pkl'
    with open(filename, 'rb') as f:
        column_transformer = pickle.load(f)

    return column_transformer

def predict_salary(age, gender, experience, education, interview_score, test_score):

    current_dir = os.getcwd()
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

    model = load_model(parent_dir)
    column_transformer = load_column_transformer(parent_dir)


    try:
        data_input = pd.DataFrame([[int(age), gender, int(experience), education, int(interview_score), int(test_score)]], columns=["Age", "Gender", "Experience", "Education", "Interview Score", "Test Score"])
        data_input_encoded = column_transformer.transform(data_input)
        salary_pred = model.predict(data_input_encoded)[0]
        print(int(salary_pred))
        return int(salary_pred)
    
    except Exception as e:
        print(e)
        return 0
    

def explanation_of_the_prediction():

    return 0

def fairness_of_the_prediction():
     return 0