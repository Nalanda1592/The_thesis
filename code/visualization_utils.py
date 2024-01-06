import openai
import dotenv
import os
import streamlit as st
import pandas as pd
import model_creation as mc
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain.tools import BaseTool
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain.agents import initialize_agent
from langchain.memory import ReadOnlySharedMemory
import single_prediction as sp


dotenv.load_dotenv('dev.env', override=True)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai.api_key=OPENAI_API_KEY


#pinecone.init(api_key="636de315-26de-4439-bdf1-5603788c6963", environment="gcp-starter")
#index_name= "explainer-index"

#index = pinecone.Index(index_name)

dotenv.load_dotenv('dev.env', override=True)

#embeddings = SentenceTransformerEmbeddings(model_name="deepset/all-mpnet-base-v2-table")

#model = SentenceTransformer("deepset/all-mpnet-base-v2-table", device='cpu')

current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
p_folder=None
exp_folder=None
single_explainability=None
fairlearn_folder_1=None
fairlearn_folder_2=None
fairlearn_folder_3=None
fairlearn_folder_4=None
fairlearn_folder_5=None
mitigation_folder_1=None
mitigation_folder_2=None
demo_filenum=None

def input_parsing(query):
    global demo_filenum
    age, gender, experience, education, interview_score, test_score=query.split(",")
    prediction=sp.predict_salary(age, gender, experience, education, interview_score, test_score,demo_filenum)
    return prediction


def query_redirecting(input :str,llm: ChatOpenAI,filenum: str):
    global demo_filenum
    demo_filenum=filenum
    response=None
    readonlymemory = ReadOnlySharedMemory(memory=st.session_state.buffer_memory)
    system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question as a helpful assistant.Use tool in most cases.""")

    human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

    prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="chat_history"), human_msg_template])
    conversation = ConversationChain(memory=readonlymemory, prompt=prompt_template, llm=llm, verbose=True)
    #context = find_match(input)
    #response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{input}")


    if(filenum=='1'):
        folder_name="Datensatz1_results"
        single_folder_name="Datensatz1_single_results"
        tool_context_selection(folder_name,single_folder_name)

    elif(filenum=='2'):
        folder_name="Datensatz2_results"
        single_folder_name="Datensatz2_single_results"
        tool_context_selection(folder_name,single_folder_name)
       

    class PredictedDatasetTool(BaseTool):
      name = "Dataset QA"
      description = "use this tool when you need to answer any questions regarding the dataset"

      def _run(self, query: str):
            df=None
            print("entered")
            df=pd.read_csv(p_folder)

            pd_agent=create_pandas_dataframe_agent(llm, df, verbose=True,agent_type=AgentType.OPENAI_FUNCTIONS,handle_parsing_errors=True)
            return pd_agent.run(query)

      def _arun(self):
            raise NotImplementedError("This tool does not support async")

    class ShapExplainPredictedResultTool(BaseTool):
      name = "SHAP Explainability QA"
      description = "use this tool when you need to answer any SHAP explanation questions like which feature/column depends on which other feature, or what is the main feature/column. Use Shap explainability library concepts to answer the questions.Use the whole query as action input. Use absolute mean of shap values to find out feature importance."

      def _run(self, query: str):
            df=None
            print("entered")
            df=pd.read_csv(exp_folder)

            pd_agent=create_pandas_dataframe_agent(llm, df, verbose=True,agent_type=AgentType.OPENAI_FUNCTIONS,handle_parsing_errors=True)
            return pd_agent.run(query)

      def _arun(self):
            raise NotImplementedError("This tool does not support async")
      
    class FairlearnFairnessTool(BaseTool):
        name = "Fairlearn Fairness QA"
        description = "use this tool to answer questions on the given context based on the concept of fairness library Fairlearn(search the internet and understand its fairness metrices in detail) and explain the given context accordingly to a layman. Also explain strategies to improve the results according to Fairlearn library, if asked so. Discuss mitigation techniques and compare them if asked so."

        def _run(self,query: str):
            print("entered")

            df_1=pd.read_csv(fairlearn_folder_1)
            df_2=pd.read_csv(fairlearn_folder_2)
            df_3=pd.read_csv(fairlearn_folder_3)
            df_4=pd.read_csv(fairlearn_folder_4)
            df_5=pd.read_csv(fairlearn_folder_5)

            pd_agent=create_pandas_dataframe_agent(llm,[df_1,df_2,df_3,df_4,df_5],verbose=True,agent_type=AgentType.OPENAI_FUNCTIONS,handle_parsing_errors=True)

            result=pd_agent.run(query)

            return result

        def _arun(self):
            raise NotImplementedError("This tool does not support async")
        
    class UnfairnessMitigationTool(BaseTool):
        name = "Unfairness Mitigation QA"
        description = "use this tool to answer unfairness mitigation questions on the given context based on the concept of Adversarial Mitigation Technique in fairness library Fairlearn and explain the technique and given context accordingly to a layman. Answer any question about Adversarial Mitigation Technique like its pros over other mitigation techniques, how is it better etc. Use the internet to answer such questions. f=female and m=male gender."

        def _run(self,query: str):
            print("entered")

            df_1=pd.read_csv(mitigation_folder_1) 
            df_2=pd.read_csv(mitigation_folder_2)      

            pd_agent=create_pandas_dataframe_agent(llm,[df_1,df_2],verbose=True, agent_type=AgentType.OPENAI_FUNCTIONS, handle_parsing_errors=True)

            result=pd_agent.run(query)

            return result

        def _arun(self):
            raise NotImplementedError("This tool does not support async")
        
        
    class SinglePredictionTool(BaseTool):
        name = "Single Prediction QA"
        description="Use this tool to calculate single prediction. The input to this tool should be a comma separated list of numbers and string values, of length six, representing age, gender, experience, education, interview_score, test_score needed to get a single prediction. For example, `32,f,7,Ma,8,9` would be the input if you wanted to take values of age, gender, experience, education, interview_score, test_score consecutively."

        def _run(self,query):
            result=input_parsing(query)
            return result

        def _arun(self):
            raise NotImplementedError("This tool does not support async")
        
    class SingleExplanationTool(BaseTool):
        name = "Single SHAP Explainability QA"
        description = "use this tool when you need to answer the single salary prediction's SHAP explanation questions like which feature/column depends on which other feature, or what is the main feature/column. Use Shap explainability library concepts to answer the questions.Use the whole query as action input. Use absolute mean of shap values to find out feature importance. Always work as a follow up question after a single prediction. Do not consider the queries like feature importance of nth observation. Such queries will go to ShapExplainPredictedResultTool tool"

        def _run(self, query: str):
            df=None
            print("entered")
            df=pd.read_csv(single_explainability)

            pd_agent=create_pandas_dataframe_agent(llm, df, verbose=True,agent_type=AgentType.OPENAI_FUNCTIONS,handle_parsing_errors=True)
            return pd_agent.run(query)

        def _arun(self):
            raise NotImplementedError("This tool does not support async")

    tools = [PredictedDatasetTool(),ShapExplainPredictedResultTool(),FairlearnFairnessTool(),UnfairnessMitigationTool(),SinglePredictionTool(),SingleExplanationTool()]
    
    # initialize agent with tools
    main_agent = initialize_agent(
        agent='chat-conversational-react-description',
        tools=tools,
        llm=llm,
        llm_chain=conversation,
        verbose=True,
        max_iterations=3,
        early_stopping_method='generate',
        memory=st.session_state.buffer_memory,
        handle_parsing_errors=True
    )
    final_word=None
    try:
        response=main_agent(input+"use tool")
        print("the response is"+ response.get('output'))
        final_word=response.get('output')
    except Exception as e:
        result = str(e)
        if result.startswith("Could not parse LLM output: `"):
            result = result.removeprefix("Could not parse LLM output: `").removesuffix("`")
        if result.startswith("An output parsing error occurred.`"):
            result = result.removeprefix("Could not parse LLM output: `").removesuffix("`")
        print(result)
        final_word=result


    return final_word


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

def dataset_selection_func(dataset_selection):
    if st.button('Total Salary Predictions'):
        #testingData=mc.run_model_generator_for_prediction(dataset_selection)
        st.write("You can now ask questions related to the dataset being used,to the chatbot. Please use the keyword 'dataset' in your questions for better results. The columns of the dataset are Age, Gender, Experience, Education, Interview Score, Test Score, Actual Salary, PredictedSalary")

    if st.button('Explanation of Predictions(SHAP)'):
        st.session_state['exp_select'] = '1'
        #output=mc.run_model_generator_for_explanation(dataset_selection)
        st.write("You can now ask questions related to the SHAP values of features/columns in the dataset being used,to the chatbot. Please use the keyword 'dataset' and 'shap values' in your questions for appropriate results. SHAP is a library for explainability. It helps explain the AI model and its predicted results and gives significant insights on the results.")

    if st.button('Fairness of Predictions(Fairlearn)'):
        st.session_state['fairlearn_select'] = '1'
        fair_output=mc.fairness_model_creation(dataset_selection)
        st.write(fair_output)

    if st.button('Unfairness Mitigation'):
        st.session_state['mitigation_select'] = '1'

        result_metrices=mc.run_model_generator_for_mitigation(dataset_selection)
        st.write(result_metrices)

def tool_context_selection(folder_name,single_folder_name):

    global parent_dir,p_folder,exp_folder,single_explainability,fairlearn_folder_1,fairlearn_folder_2,fairlearn_folder_3,fairlearn_folder_4,fairlearn_folder_5,mitigation_folder_1,mitigation_folder_2

    p_folder="data/total_prediction_results/"+folder_name+"/total_database_prediction.csv"
    single_explainability="data/single_prediction_results/"+single_folder_name+"/total_result.csv"
    if(st.session_state['exp_select']=='1'):
        exp_folder="data/total_prediction_results/"+folder_name+"/explanations/shap_result.csv"
    if(st.session_state['fairlearn_select'] == '1'):
        fairlearn_folder_1="data/total_prediction_results/"+folder_name+"/fairness_measures/Fairlearn_results_1.csv"
        fairlearn_folder_2="data/total_prediction_results/"+folder_name+"/fairness_measures/Fairlearn_results_2.csv"
        fairlearn_folder_3="data/total_prediction_results/"+folder_name+"/fairness_measures/Fairlearn_results_3.csv"
        fairlearn_folder_4="data/total_prediction_results/"+folder_name+"/fairness_measures/Fairlearn_results_4.csv"
        fairlearn_folder_5="data/total_prediction_results/"+folder_name+"/fairness_measures/Fairlearn_results_5.csv"
    if(st.session_state['mitigation_select'] == '1'):
        mitigation_folder_1="data/total_prediction_results/"+folder_name+"/fairness_measures/unfairness_mitigation_result_1.csv"
        mitigation_folder_2="data/total_prediction_results/"+folder_name+"/fairness_measures/unfairness_mitigation_result_2.csv"


