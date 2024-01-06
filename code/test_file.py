import numpy as np
import pandas as pd
import os
import model_creation_utils as mu

current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

#find max values of age, experience, interview_score, test_score, salary
def max_vals(data,data_exp):

    no_col=len(data.axes[1])
    print("No. of columns",no_col)
    print("numeric max: ",data.max(numeric_only=True))
    print("avg act sal: ",data['Actual Salary'].mean())
    print("sal based on gen: ",data.groupby('Gender')['Actual Salary'].mean())
    print("no. of female : ",len(data[data['Gender'] == 'f']))
    print("avg act sal for 30 yr olds : ",data[data['Age'] == 30]['Actual Salary'].mean())
    print("avg sal for 32 yr old phd holders : ",data[(data['Age'] == 32) & (data['Education'] == 'PhD')]['PredictedSalary'].mean())

    shap_res=[i for i in data_exp if i.startswith('Shap')]
    #print(data_exp[shap_res].head(5))
    avg_data=data_exp[shap_res].abs().mean()
    print("global Shap values for features : ",avg_data)
    print("shap for 10th observation : ",data_exp[shap_res].iloc[9])

def main():
    #for gender bias data 
    #data_1 = pd.read_csv(current_dir + "/data/total_prediction_results/Datensatz2_results/" + "total_database_prediction.csv")
    #data_2=pd.read_csv(current_dir + "/data/total_prediction_results/Datensatz2_results/explanations/shap_result.csv")
    unfai_df_1= pd.read_csv(current_dir + "/data/total_prediction_results/Datensatz2_results/fairness_measures/unfairness_mitigation_result_1.csv")
    unfai_df_2= pd.read_csv(current_dir + "/data/total_prediction_results/Datensatz2_results/fairness_measures/unfairness_mitigation_result_2.csv")

    print(unfai_df_2)


    #max_vals(data_1,data_2)

    #for uneven gender distribution data 
    #data_2 = pd.read_csv(current_dir + "/data/" + "Datensatz 2.csv")
    #mu.unfairness_mitigation_SMOTE(data_2)

   

if __name__ == "__main__":
    main()


    
