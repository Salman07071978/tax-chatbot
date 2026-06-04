import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
from huggingface_hub import InferenceClient

st.set_page_config(page_title="Pakistan Tax AI", layout="wide")

st.title("🇵🇰 Pakistan Tax AI Assistant (Fast Mode)")

# ---------------- SECRET ----------------
HF_API_KEY = st.secrets["HF_API_KEY"]

client = InferenceClient(
    provider="hf-inference",
    api_key=HF_API_KEY,
)

# ---------------- LOAD PDF (FAST LIMITED) ----------------
@st.cache_data
def load_pdf():
    reader = PdfReader("taxlaw.pdf")
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text[:20000]   # IMPORTANT: limit size for speed

pdf_text = load_pdf()

# ---------------- SPLIT TEXT ----------------
def split_text(text, size=800):
    return [text[i:i+size] for i in range(0, len(text), size)]

chunks = split_text(pdf_text)

# ---------------- MODEL (CACHED) ----------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ---------------- EMBEDDINGS (CACHED ONCE) ----------------
@st.cache_resource
def build_embeddings(chunks):
    return model.encode(chunks)

chunk_vectors = build_embeddings(chunks)

# ---------------- SEARCH FUNCTION ----------------
def get_context(query):
    q_vec = model.encode([query])
    scores = np.dot(chunk_vectors, q_vec.T)
    idx = np.argmax(scores)
    return chunks[idx]

# ---------------- AI ANSWER ----------------
def ai_answer(question, context):

    prompt = f"""
You are a Pakistan Tax Expert AI.

Use ONLY the context below.

Context:
{context}

Question:
{question}

Give a simple legal explanation.
Mention tax law reasoning if possible.
"""

    try:
        completion = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
        )

        return completion.choices[0].message.content

    except Exception as e:
        return f"AI Error (try again later): {str(e)}"

# ---------------- UI ----------------
question = st.text_input("Ask Income Tax or Sales Tax Question")

if question:

    with st.spinner("Searching Tax Laws..."):
        context = get_context(question)

    st.success("Relevant Law Found")

    st.markdown("### 📚 Context")
    st.write(context)

    st.markdown("### 🧠 AI Answer")

    answer = ai_answer(question, context)
    st.write(answer)
