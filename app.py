import streamlit as st
from pypdf import PdfReader

st.title("🇵🇰 Pakistan Tax Smart Assistant")

# Load PDF
def get_text():
    try:
        reader = PdfReader("taxlaw.pdf")
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except:
        return ""

pdf_text = get_text()

st.write("Ask Income Tax or Sales Tax question")

question = st.text_input("Your Question")

if question:
    q = question.lower()

    # Simple smart search (keyword match)
    if pdf_text:
        if q in pdf_text.lower():
            st.success("Relevant Tax Law Found:")
            
            # show small relevant chunk
            index = pdf_text.lower().find(q)
            start = max(0, index - 200)
            end = index + 800

            st.write(pdf_text[start:end])
        else:
            st.info("AI Explanation Mode:")
            st.write(f"""
**Question:** {question}

This is a tax-related query under Pakistan Income Tax / Sales Tax laws.

👉 In real system, this will:
- Search Income Tax Ordinance 2001
- Search Sales Tax Act 1990
- Provide section-wise explanation
- Give legal references

(Current system is base version)
""")
