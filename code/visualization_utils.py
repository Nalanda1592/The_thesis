import pinecone
import openai
import dotenv
import os
import streamlit as st
import pandas as pd
import model_creation
from langchain.chains import ConversationChain
from langchain.embeddings import SentenceTransformerEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain.agents import Tool
from langchain.tools import BaseTool
from langchain.agents import load_tools
from langchain.agents import create_pandas_dataframe_agent,create_csv_agent
from langchain.agents import initialize_agent
from langchain.memory import ReadOnlySharedMemory


dotenv.load_dotenv('dev.env', override=True)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai.api_key=OPENAI_API_KEY


pinecone.init(api_key="636de315-26de-4439-bdf1-5603788c6963", environment="gcp-starter")
index_name= "explainer-index"

index = pinecone.Index(index_name)

dotenv.load_dotenv('dev.env', override=True)

embeddings = SentenceTransformerEmbeddings(model_name="deepset/all-mpnet-base-v2-table")

model = SentenceTransformer("deepset/all-mpnet-base-v2-table", device='cpu')
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))


def query_redirecting(input :str,llm: ChatOpenAI,filenum: str):
    response=None
    readonlymemory = ReadOnlySharedMemory(memory=st.session_state.buffer_memory)
    system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question as a helpful assistant.""")

    human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

    prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="chat_history"), human_msg_template])
    conversation = ConversationChain(memory=readonlymemory, prompt=prompt_template, llm=llm, verbose=True)
    #context = find_match(input)
    #response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{input}")

    p_folder=None
    exp_folder=None
    if(filenum=='1'):
        original_file="Datensatz 1.csv"
        folder_name="Datensatz1_results"
        p_folder=parent_dir + "/data/total_prediction_results/"+folder_name+"/total_database_prediction.csv"
        if(st.session_state['exp_select']=='1'):
            exp_folder=parent_dir + "/data/total_prediction_results/"+folder_name+"/explanations/shap_result.csv"
        if(st.session_state['dalex_select'] == '1'):
            dalex_folder_1=parent_dir + "/data/total_prediction_results/"+folder_name+"/fairness_measures/model_performance_for_fairness_using_dalex.txt"
            dalex_folder_2=parent_dir + "/data/total_prediction_results/"+folder_name+"/fairness_measures/result_of_function_fairness_check_using_dalex.txt"
            dalex_folder_3=parent_dir + "/data/total_prediction_results/"+folder_name+"/fairness_measures/group_fairness_regression_result_using_dalex.txt"
        if(st.session_state['fairlearn_select'] == '1'):
            fairlearn_folder_1=parent_dir + "/data/total_prediction_results/"+folder_name+"/fairness_measures/Fairlearn_results_1.csv"
            fairlearn_folder_2=parent_dir + "/data/total_prediction_results/"+folder_name+"/fairness_measures/Fairlearn_results_2.csv"
       

    elif(filenum=='2'):
        original_file="Datensatz 2.csv"
        folder_name="Datensatz2_results"
        p_folder=parent_dir + "/data/total_prediction_results/"+folder_name+"/total_database_prediction.csv"
        if(st.session_state['exp_select']=='1'):
            exp_folder=parent_dir + "/data/total_prediction_results/"+folder_name+"/explanations/shap_result.csv"
        if(st.session_state['dalex_select'] == '1'):
            dalex_folder_1=parent_dir + "/data/total_prediction_results/"+folder_name+"/fairness_measures/model_performance_for_fairness_using_dalex.txt"
            dalex_folder_2=parent_dir + "/data/total_prediction_results/"+folder_name+"/fairness_measures/result_of_function_fairness_check_using_dalex.txt"
            dalex_folder_3=parent_dir + "/data/total_prediction_results/"+folder_name+"/fairness_measures/group_fairness_regression_result_using_dalex.txt"
        if(st.session_state['fairlearn_select'] == '1'):
            fairlearn_folder_1=parent_dir + "/data/total_prediction_results/"+folder_name+"/fairness_measures/Fairlearn_results_1.csv"
            fairlearn_folder_2=parent_dir + "/data/total_prediction_results/"+folder_name+"/fairness_measures/Fairlearn_results_2.csv"
       

    else:
        original_file="Datensatz 3.csv"
        folder_name="Datensatz3_results"
        p_folder=parent_dir + "/data/total_prediction_results/"+folder_name+"/total_database_prediction.csv"
        if(st.session_state['exp_select']=='1'):
            exp_folder=parent_dir + "/data/total_prediction_results/"+folder_name+"/explanations/shap_result.csv"
        if(st.session_state['dalex_select'] == '1'):
            dalex_folder_1=parent_dir + "/data/total_prediction_results/"+folder_name+"/fairness_measures/model_performance_for_fairness_using_dalex.txt"
            dalex_folder_2=parent_dir + "/data/total_prediction_results/"+folder_name+"/fairness_measures/result_of_function_fairness_check_using_dalex.txt"
            dalex_folder_3=parent_dir + "/data/total_prediction_results/"+folder_name+"/fairness_measures/group_fairness_regression_result_using_dalex.txt"  
        if(st.session_state['fairlearn_select'] == '1'):
            fairlearn_folder_1=parent_dir + "/data/total_prediction_results/"+folder_name+"/fairness_measures/Fairlearn_results_1.csv"
            fairlearn_folder_2=parent_dir + "/data/total_prediction_results/"+folder_name+"/fairness_measures/Fairlearn_results_2.csv"
       

    class PredictedDatabaseTool(BaseTool):
      name = "Database QA"
      description = "use this tool when you need to answer any questions regarding the database"

      def _run(self, query: str):
            df=None
            print("entered")
            df=pd.read_csv(p_folder)

            pd_agent=create_pandas_dataframe_agent(llm, df, verbose=True)
            return pd_agent.run(query)

      def _arun(self):
            raise NotImplementedError("This tool does not support async")

    class ShapExplainPredictedResultTool(BaseTool):
      name = "SHAP Explainability QA"
      description = "use this tool when you need to answer any SHAP explanation questions like which feature/column depends on which other feature, or what is the main feature/column. Use Shap explainability library concepts to answer the questions.show visualizations using shap plot functions if requested for."

      def _run(self, query: str):
            df=None
            print("entered")
            df=pd.read_csv(exp_folder)

            pd_agent=create_pandas_dataframe_agent(llm, df, verbose=True)
            return pd_agent.run(query)

      def _arun(self):
            raise NotImplementedError("This tool does not support async")
      
    class FairlearnFairnessTool(BaseTool):
        name = "Fairlearn Fairness QA"
        description = "use this tool to answer questions on the given context based on the concept of fairness library Fairlearn(search the internet and understand its fairness metrices in detail) and explain the given context accordingly to a layman. Also explain Demographic Parity Ratio and Equalized Odds ratio of Fairlearn fairness library in very easy way to a layman at first, as these are the metrics being used to calculate the fairness for the context.The interpretation of the values depend on the definitions as follows. Demographic Parity: It measures the ML model's ability to make prediction such that they are independent of the influence by sensitive groups. Equalized odds: It also ensures that ML model's predictions are independent of sensitive groups. It's more strict than Demographic parity by ensuring all groups in the dataset have same true positive rates and false positive rates. Equal Opportunity: It's similar to equalized odds but applies only to positive instances, i.e. Y=1. Demographic parity ratio: Ratio of selection rates between smallest and largest groups. Return type is a decimal value. A ratio of 1 means all groups have same selection rate. Equalized odds ratio: The equalized odds ratio of 1 means that all groups have the same true positive, true negative, false positive, and false negative rates."

        def _run(self,query: str):
            print("entered")

            df_1=pd.read_csv(fairlearn_folder_1)
            df_2=pd.read_csv(fairlearn_folder_2)

            pd_agent=create_pandas_dataframe_agent(llm,[df_1,df_2],verbose=True)

            return pd_agent.run(query+"use tool")

        def _arun(self):
            raise NotImplementedError("This tool does not support async")
        
    class DalexFairnessTool(BaseTool):
        name = "Dalex Fairness QA"
        description = "use this tool to answer questions on the given context based on the concept of fairness library dalex(search the internet and understand its fairness metrices like TPR (True Positive Rate), ACC (Accuracy),  PPV (Positive Predictive Value), FPR (False Positive Rate), STP(Statistical parity) in detail) and explain the given context accordingly to a layman. There are 3 types of possible conclusions: # not fair Conclusion: your model is not fair because 2 or more metric scores exceeded acceptable limits set by epsilon. # neither fair or not Conclusion: your model cannot be called fair because 1 metric score exceeded acceptable limits set by epsilon.It does not mean that your model is unfair but it cannot be automatically approved based on these metrics. # fair Conclusion: your model is fair in terms of checked fairness metrics.The context here is for male and female subgroups."


        def _run(self,query: str):
            df_1=pd.read_csv(dalex_folder_1)
            df_2=pd.read_csv(dalex_folder_2)
            df_3=pd.read_csv(dalex_folder_3)

            pd_agent=create_pandas_dataframe_agent(llm, [df_1,df_2,df_3,], verbose=True)

            return pd_agent.run(query)

        def _arun(self):
            raise NotImplementedError("This tool does not support async")

    tools = [PredictedDatabaseTool(),ShapExplainPredictedResultTool(),FairlearnFairnessTool(),DalexFairnessTool()]
    
    # initialize agent with tools
    main_agent = initialize_agent(
        agent='chat-conversational-react-description',
        tools=tools,
        llm=llm,
        llm_chain=conversation,
        verbose=True,
        max_iterations=3,
        early_stopping_method='generate',
        memory=st.session_state.buffer_memory
    )

    response=main_agent(input)
    print("the response is"+ response.get('output'))


    return response.get('output')

def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(input_em, top_k=1, includeMetadata=True)
    return result['matches'][0]['id']

def query_refiner(conversation, query):

    response = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0.7,
        max_tokens=100,
        n=1,
        messages=[{"role": "system", "content":f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.Use tools first.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:"}]
    )
    return response['choices'][0]['message']['content']

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string