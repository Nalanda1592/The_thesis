import streamlit as st
from streamlit_chat import message
import dotenv
import os
import model_creation as mc
from langchain.chat_models import ChatOpenAI
from visualization_utils import *
from langchain.memory import ConversationBufferWindowMemory
from trubrics import Trubrics
from langchain.callbacks import TrubricsCallbackHandler
import uuid
from trubrics.integrations.streamlit import FeedbackCollector


def main():

    # Setting page title and header
    st.set_page_config(page_title="LLM Explainer ", page_icon="👩🏻‍🏫")
    #st.header("Custom ChatGPT")
    st.markdown("<h1 style='text-align: center;'>Chat Explainer</h1>", unsafe_allow_html=True)

    file_num=None
    user_feedback=None

    if 'exp_select' not in st.session_state:
        st.session_state['exp_select'] = '0'
    if 'fairlearn_select' not in st.session_state:
        st.session_state['fairlearn_select'] = '0'
    if 'mitigation_select' not in st.session_state:
        st.session_state['mitigation_select'] = '0'

    if "prompt_id" not in st.session_state:
        st.session_state.prompt_id = None
    if "logged_prompt" not in st.session_state:
        st.session_state.logged_prompt = {}
    if "feedback_key" not in st.session_state:
        st.session_state.feedback_key = None
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())


    with st.sidebar:
        database_selection=st.radio(label='Choose the type of dataset you want to use',options=('Gender Bias','Uneven Gender Distribution'))  

        if database_selection=='Gender Bias':
            database_selection='1'
            file_num='1'
            database_selection_func(database_selection)
            

        elif database_selection=='Uneven Gender Distribution':
            database_selection='2'
            file_num='2'
            database_selection_func(database_selection)

    if 'responses' not in st.session_state:
        st.session_state['responses'] = ["How can I assist you?"]

    if 'requests' not in st.session_state:
        st.session_state['requests'] = []


    tb = FeedbackCollector(project="thesis",email=os.environ["TRUBRICS_EMAIL"],password=os.environ["TRUBRICS_PASSWORD"],)

    #user_prompt = tb.log_prompt(config_model={"model": "gpt-4"},prompt="How may I help you?",)

    dotenv.load_dotenv('dev.env', override=True)
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

    llm = ChatOpenAI(temperature = 0, model_name='gpt-4',openai_api_key= OPENAI_API_KEY,callbacks=[
                TrubricsCallbackHandler(
                    project="thesis",
                    tags=["chat model"],
                    user_id="user-id-1234",
                    some_metadata={"hello": [1, 2]},
                    #prompt_id=st.session_state.prompt_id,
                    session_id=st.session_state.session_id,
                )])
    

    if 'buffer_memory' not in st.session_state:

        st.session_state.buffer_memory=ConversationBufferWindowMemory(memory_key='chat_history',k=5,return_messages=True)

    #container for chat history
    response_container = st.container()
    #container for text box
    textcontainer = st.container()

    def clear_text():
        st.session_state.my_text = st.session_state.input
        st.session_state.input = ""


    with textcontainer:
        st.text_input("Query: ", key="input", on_change=clear_text)
        query = st.session_state.get('my_text', '')
        st.session_state.my_text=""

        response=None

        if query:
            #st.session_state.prompt_id=str(uuid.uuid4())
            if "how to predict salary" in query:
                response="For a single prediction: max age=35,gender={m,f},max experience=15,education={PhD,Ma},max interview score=10,max test score=10. Example question template='Get single prediction for 32,'f',7,'Ma',8,9. dont change the these values'"
            else:
                with st.spinner("typing..."):
                    conversation_string = get_conversation_string()
                    refined_query = query_refiner(conversation_string, query)
                    st.subheader("Refined Query:")
                    st.write(refined_query)
                    response=query_redirecting(refined_query,llm,file_num)
            if(response):
                st.session_state.logged_prompt={"answer":response,"id":st.session_state.session_id}
                st.session_state.feedback_key =str(uuid.uuid4())

            st.session_state.requests.append(query)
            st.session_state.responses.append(response) 

    with response_container:
 
        if st.session_state['responses']:

            for i in range(len(st.session_state['responses'])):
                message(st.session_state['responses'][i],key=str(i))
                if i < len(st.session_state['requests']):
                    message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')

        if st.session_state.logged_prompt:
            user_feedback = tb.st_feedback(
                component="thesis",
                feedback_type="thumbs",
                open_feedback_label="[Optional] Provide additional feedback",
                model="gpt-4",
                prompt_id=st.session_state.session_id,
                key=st.session_state.feedback_key,
                align="flex-end",
            )  
    
 
if __name__=='__main__':
    main()