import streamlit as st
from dotenv import load_dotenv
import os
import uuid
import pandas as pd
from datetime import datetime

from app.utils.tool_manager import TOOLS
from app.chatbot_ui.layout import render_layout
from app.chatbot_ui.session_state import init_session_state
from app.utils.config import ENV_FILE_PATH, ROOT_DIR

from langchain.agents import initialize_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.llms.openai import OpenAIChat

# === Load environment variables ===
load_dotenv(ENV_FILE_PATH)
print("✅ ENV loaded from:", ENV_FILE_PATH)
print("🔑 OPENAI_API_KEY found?", os.getenv("OPENAI_API_KEY") is not None)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# === Init Session ID ===
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

# === Initialize LangChain LLM ===
llm = OpenAIChat(
    temperature=0,
    model="gpt-4-turbo",
    openai_api_key=OPENAI_API_KEY,
)

print("LLM Class:", llm.__class__.__name__)


# === Render UI & Initialize State ===
render_layout()
init_session_state()

# === Setup memory and agent ===
if "buffer_memory" not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=5, return_messages=True)

agent_executor = initialize_agent(
    tools=TOOLS,
    llm=llm,
    agent="chat-conversational-react-description",
    memory=st.session_state.buffer_memory,
    verbose=True,
    handle_parsing_errors=True,
)

# === Chat Input / Output ===
query = st.chat_input("Ask about predictions, fairness, SHAP, etc...")
if query:
    with st.spinner("Thinking..."):
        try:
            result = agent_executor.run(query + " Use tool")
        except Exception as e:
            result = f"⚠️ Error: {str(e)}"
        st.session_state["requests"].append(query)
        st.session_state["responses"].append(result)

# === Chat History ===
for i, res in enumerate(st.session_state["responses"]):
    st.chat_message("user").write(st.session_state["requests"][i])
    st.chat_message("assistant").write(res)

# === Emoji Mood Rating Feedback ===
if st.session_state["responses"]:
    last_response = st.session_state["responses"][-1]
    last_prompt = st.session_state["requests"][-1]

    with st.expander("💬 Rate this answer"):
        mood_rating = st.radio(
            "How did this response make you feel?",
            ["😡", "😕", "😐", "🙂", "🤩"],
            horizontal=True,
            key="emoji_rating"
        )

        comment = st.text_area("Optional comment", placeholder="What did you like or dislike?", key="feedback_comment")

        if st.button("Submit Feedback"):
            feedback_data = {
                "timestamp": datetime.now().isoformat(),
                "session_id": st.session_state["session_id"],
                "prompt": last_prompt,
                "response": last_response,
                "emoji_rating": mood_rating,
                "comment": comment,
            }

            feedback_df = pd.DataFrame([feedback_data])
            feedback_path = ROOT_DIR / "data" / "feedback.csv"

            if feedback_path.exists():
                feedback_df.to_csv(feedback_path, mode="a", header=False, index=False)
            else:
                feedback_df.to_csv(feedback_path, index=False)

            st.success("✅ Feedback submitted. Thank you!")
