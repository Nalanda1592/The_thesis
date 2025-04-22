import streamlit as st

def render_layout():
    st.set_page_config(page_title="AI Model Chat", page_icon="🤖")
    
    st.markdown(
        "<h1 style='text-align: center;'>📊 AI Fairness & SHAP Chatbot</h1>",
        unsafe_allow_html=True
    )

    with st.sidebar:
        st.header("💡 How To Use")

        st.markdown("""
        ### 📂 Dataset Queries
        - "How many columns are in the dataset?"
        - "What is the average salary by gender?"
        - "Max value of interview score?"

        ### 🔮 SHAP Explainability
        - "What are the most important features?"
        - "Explain 10th observation using SHAP"

        ### 🧪 Single Prediction
        - Ask: `"Predict salary for 32,f,7,PhD,8,9"`
        - Then: "What features mattered in this prediction?"

        ### ⚖️ Fairness & Mitigation
        - "How fair is the model?"
        - "Explain the Fairlearn results"
        - "What mitigation technique was used?"

        ---

        ⚠️ *If an answer seems off, try rephrasing or ask again!*

        💬 *You can give feedback using 👍 / 👎 below the chatbot.*
        """)
