import streamlit as st
from streamlit_chat import message
import dotenv
import os
import model_creation as mc
from langchain.chat_models import ChatOpenAI
from visualization_utils import *
from langchain.memory import ConversationBufferWindowMemory

def main():

    # Setting page title and header
    st.set_page_config(page_title="LLM Explainer ", page_icon="👩🏻‍🏫")
    #st.header("Custom ChatGPT")
    st.markdown("<h1 style='text-align: center;'>Chat Explainer</h1>", unsafe_allow_html=True)

    file_num=None

    if 'exp_select' not in st.session_state:
        st.session_state['exp_select'] = '0'


    with st.sidebar:
        database_selection=st.radio(label='Choose the type of dataset you want to use and wait for a minute',options=('Gender Bias','Uneven Gender Distribution','Nepotism'))   

        if database_selection=='Gender Bias':
            database_selection='1'
            file_num='1'
            if st.button('Total Salary Predictions'):
                testingData=mc.run_model_generator_for_prediction(database_selection)
                st.write(testingData.head())

            if st.button('Explanation of Predictions(SHAP)'):
                st.session_state['exp_select'] = '1'
                output=mc.run_model_generator_for_explanation(database_selection)
                st.write(output.head())

            if st.button('Fairness of Predictions(dalex)'):
                st.session_state['dalex_select'] = '1'
                mc.run_model_generator_for_fairness_dalex(database_selection)

            if st.button('Fairness of Predictions(Fairlearn)'):
                st.session_state['fairlearn_select'] = '1'
                mc.fairness_model_creation(database_selection)

        elif database_selection=='Uneven Gender Distribution':
            database_selection='2'
            file_num='2'
            if st.button('Total Salary Predictions'):
                testingData=mc.run_model_generator_for_prediction(database_selection)
                st.write(testingData.head())   

            if st.button('Explanation of Predictions(SHAP)'):
                st.session_state['exp_select'] = '1'
                output=mc.run_model_generator_for_explanation(database_selection)
                st.write(output.head()) 

            if st.button('Fairness of Predictions(dalex)'):
                st.session_state['dalex_select'] = '1'
                mc.run_model_generator_for_fairness_dalex(database_selection)  

            if st.button('Fairness of Predictions(Fairlearn)'):
                st.session_state['fairlearn_select'] = '1'
                mc.fairness_model_creation(database_selection)  

        elif database_selection=='Nepotism':
            database_selection='3'
            file_num='3'
            if st.button('Total Salary Predictions'):
                testingData=mc.run_model_generator_for_prediction(database_selection)
                st.write(testingData.head())  

            if st.button('Explanation of Predictions(SHAP)'):
                st.session_state['exp_select'] = '1'
                output=mc.run_model_generator_for_explanation(database_selection)
                st.write(output.head())   

            if st.button('Fairness of Predictions(dalex)'):
                st.session_state['dalex_select'] = '1'
                mc.run_model_generator_for_fairness_dalex(database_selection)   

            if st.button('Fairness of Predictions(Fairlearn)'):
                st.session_state['fairlearn_select'] = '1'
                fair_output=mc.fairness_model_creation(database_selection)
                st.write(fair_output)
        

    if 'responses' not in st.session_state:
        st.session_state['responses'] = ["How can I assist you?"]

    if 'requests' not in st.session_state:
        st.session_state['requests'] = []

    dotenv.load_dotenv('dev.env', override=True)
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    llm = ChatOpenAI(temperature = 0, model_name='gpt-4',openai_api_key= OPENAI_API_KEY)

    if 'buffer_memory' not in st.session_state:

        st.session_state.buffer_memory=ConversationBufferWindowMemory(memory_key='chat_history',k=5,return_messages=True)

    #container for chat history
    response_container = st.container()
    #container for text box
    textcontainer = st.container()

    with textcontainer:
        query = st.text_input("Query: ", key="input")
        if query:
            with st.spinner("typing..."):
                conversation_string = get_conversation_string()
                refined_query = query_refiner(conversation_string, query)
                st.subheader("Refined Query:")
                st.write(refined_query)
                response=query_redirecting(refined_query,llm,file_num)
            st.session_state.requests.append(query)
            st.session_state.responses.append(response) 

    with response_container:
        if st.session_state['responses']:

            for i in range(len(st.session_state['responses'])):
                message(st.session_state['responses'][i],key=str(i))
                if i < len(st.session_state['requests']):
                    message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')

 
if __name__=='__main__':
    main()