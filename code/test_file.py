import numpy as np
import pandas as pd
import os

current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

#find max values of age, experience, interview_score, test_score, salary
def max_vals(data,data_exp):

    max_age=data['Age'].max()
    print("Max Age",max_age)
    max_experience=data['Experience'].max()
    print("Max Experience",max_experience)
    max_interview_score=data['Interview Score'].max()
    print("Max Interview Score",max_interview_score)
    max_test_score=data['Test Score'].max()
    print("Max Test Score",max_test_score)
    max_salary=data['Salary'].max()
    print("Max Salary",max_salary)

    avg_data=data_exp.abs().mean(axis=0,numeric_only=True)
    print(avg_data)

def main():
    #for gender bias data 
    data_1 = pd.read_csv(current_dir + "/data/" + "Datensatz 1.csv")
    data_2=pd.read_csv(current_dir + "/data/total_prediction_results/Datensatz1_results/explanations/shap_result.csv")
    max_vals(data_1,data_2)

    #for uneven gender distribution data 
    data_2 = pd.read_csv(current_dir + "/data/" + "Datensatz 2.csv")

    #for gender bias data 
    data_3 = pd.read_csv(current_dir + "/data/" + "Datensatz 3.csv")
   

if __name__ == "__main__":
    main()


    
