import streamlit as st
from pypdf import PdfReader

st.title("🇵🇰 Pakistan Tax AI Chatbot")

# Load PDF
def load_pdf():
    text = ""
    reader = PdfReader("taxlaw.pdf")  # optional file
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

try:
    pdf_text = load_pdf()
except:
    pdf_text = ""

st.write("Ask Income Tax or Sales Tax questions")

question = st.text_input("Your Question")

if question:
    question_lower = question.lower()

    if pdf_text and question_lower in pdf_text.lower():
        st.success("Found in Tax Law:")
        st.write(pdf_text[:1500])
    else:
        st.info("Basic response (AI mode):")
        st.write(f"""
        You asked: {question}

        This is a tax-related query. In future upgrade, this bot will
        answer using Income Tax Ordinance 2001 and Sales Tax Act 1990
        with proper section references.
        """)
