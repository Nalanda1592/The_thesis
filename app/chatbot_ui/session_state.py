import streamlit as st

def init_session_state():
    if "requests" not in st.session_state:
        st.session_state["requests"] = []
    if "responses" not in st.session_state:
        st.session_state["responses"] = []
