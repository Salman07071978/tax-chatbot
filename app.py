import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np

st.title("🇵🇰 Pakistan Tax AI Pro Assistant")

# Load PDF
def load_pdf():
    reader = PdfReader("taxlaw.pdf")
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

pdf_text = load_pdf()

# Split into chunks
def split_text(text, size=800):
    return [text[i:i+size] for i in range(0, len(text), size)]

chunks = split_text(pdf_text)

# Embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

chunk_vectors = model.encode(chunks)

# Search function
def get_best_chunk(query):
    q_vec = model.encode([query])
    scores = np.dot(chunk_vectors, q_vec.T)
    idx = np.argmax(scores)
    return chunks[idx]

st.write("Ask Income Tax or Sales Tax question")

question = st.text_input("Your Question")

if question:
    context = get_best_chunk(question)

    st.success("📚 Relevant Tax Law Found")

    st.write(context)

    st.markdown("### 🤖 AI Answer (Pro Mode)")

    st.write(f"""
Based on Pakistan Income Tax Ordinance 2001 / Sales Tax Act 1990:

**Question:** {question}

**Explanation:**
The answer is derived from official tax law documents. In production version, this will be enhanced with:
- Section-wise citation
- Legal interpretation
- FBR circular references
- Case-based reasoning

**Reference Context:**
{context[:500]}
""")
