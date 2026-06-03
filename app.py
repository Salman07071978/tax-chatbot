import streamlit as st
import requests
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np

st.set_page_config(page_title="Pakistan Tax AI Brain", layout="wide")

st.title("🇵🇰 Pakistan Tax AI Brain")

# ---------------- PDF LOAD ----------------
def load_pdf():
    reader = PdfReader("taxlaw.pdf")
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

pdf_text = load_pdf()

# ---------------- SPLIT TEXT ----------------
def split_text(text, size=900):
    return [text[i:i+size] for i in range(0, len(text), size)]

chunks = split_text(pdf_text)

# ---------------- EMBEDDING MODEL ----------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

chunk_vectors = model.encode(chunks)

# ---------------- SEARCH FUNCTION ----------------
def get_context(query):
    q_vec = model.encode([query])
    scores = np.dot(chunk_vectors, q_vec.T)
    idx = np.argmax(scores)
    return chunks[idx]

# ---------------- AI (HUGGINGFACE BRAIN) ----------------
HF_API_KEY = st.secrets["HF_API_KEY"]

def ai_answer(question, context):
    prompt = f"""
You are a Pakistan Tax Expert.

Use the context below to answer.

Context:
{context}

Question:
{question}

Give a simple legal explanation with section reference.
"""

    response = requests.post(
        "https://api-inference.huggingface.co/models/google/flan-t5-large",
        headers={"Authorization": f"Bearer {HF_API_KEY}"},
        json={"inputs": prompt}
    )

    try:
        return response.json()[0]["generated_text"]
    except:
        return "AI is busy. Try again."

# ---------------- UI ----------------
question = st.text_input("Ask Income Tax or Sales Tax Question")

if question:

    with st.spinner("Reading Tax Laws..."):
        context = get_context(question)

    st.success("Relevant Law Found")

    st.markdown("### 📚 Legal Context")
    st.write(context)

    st.markdown("### 🧠 AI Answer")

    answer = ai_answer(question, context)
    st.write(answer)
