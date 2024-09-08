import streamlit as st
from module import LLMAgent

llm_agent = LLMAgent(model_name="gpt2")

st.title("Student Management System: Book Query Agent")

query = st.text_input("Ask about a book or student-related topic:")

if st.button("Submit"):
    if query:
        with st.spinner("Processing..."):
            try:
                response = llm_agent.get_response(query)
                
                recommendations = llm_agent.get_recommendations(query)
                
                st.markdown("""
                    <div style="font-size: 20px; font-weight: bold; color: #4CAF50;">
                        Recommendations:
                    </div>
                    <ul style="font-size: 16px; color: #555;">
                        {}
                    </ul>
                """.format(''.join(f"<li>{rec}</li>" for rec in recommendations)), unsafe_allow_html=True)
                
                llm_agent.log_query(query)
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a query.")
