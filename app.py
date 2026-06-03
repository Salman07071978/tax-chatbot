import streamlit as st

st.title("Pakistan Tax Chatbot")

st.write("Ask Income Tax or Sales Tax related questions")

question = st.text_input("Your Question")

if question:
    st.write("You asked:", question)
    st.info("Next step: we will connect tax knowledge base")
