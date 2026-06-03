import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np

st.title("🇵🇰 Pakistan Tax AI Assistant")

# Load PDF
def load_pdf():
    text = ""
    try:
        reader = PdfReader("taxlaw.pdf")
        for page in reader.pages:
            text += page.extract_text() or ""
    except:
        text = ""
    return text

pdf_text = load_pdf()

# Split text into chunks
def split_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

chunks = split_text(pdf_text)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

chunk_embeddings = model.encode(chunks)

def search(query):
    q_emb = model.encode([query])
    scores = np.dot(chunk_embeddings, q_emb.T)
    idx = np.argmax(scores)
    return chunks[idx]

st.write("Ask Income Tax or Sales Tax question")

question = st.text_input("Your Question")

if question:
    result = search(question)

    st.success("AI Found Relevant Tax Info:")
    st.write(result)

    st.info("Answer Summary:")
    st.write(f"""
Based on Pakistan tax law:

👉 {question}

Relevant legal explanation is extracted from Income Tax / Sales Tax documents.

(This is AI-powered retrieval system - no manual matching)
""")
