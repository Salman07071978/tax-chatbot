import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

st.set_page_config(page_title="Pakistan Tax AI", layout="wide")

st.title("🇵🇰 Pakistan Tax AI Assistant")

# ---------------- SECRET ----------------
HF_API_KEY = st.secrets["HF_API_KEY"]

client = InferenceClient(api_key=HF_API_KEY)

# ---------------- LOAD DATA ----------------
data = np.load("tax_data.npz", allow_pickle=True)

chunks = data["chunks"]
chunk_vectors = data["embeddings"]

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ---------------- SEARCH ----------------
def get_context(query):
    q_vec = model.encode([query])
    scores = np.dot(chunk_vectors, q_vec.T)
    idx = np.argmax(scores)
    return chunks[idx]

# ---------------- AI ----------------
def ai_answer(question, context):

    prompt = f"""
You are a Pakistan Tax Law Expert.

Context:
{context}

Question:
{question}

Answer simply and clearly.
"""

    try:
        response = client.text_generation(
            model="google/flan-t5-large",
            prompt=prompt,
            max_new_tokens=300,
        )

        return response

    except Exception as e:
        return f"AI Error: {str(e)}"

# ---------------- UI ----------------
question = st.text_input("Ask Income Tax or Sales Tax Question")

if question:

    with st.spinner("Searching Tax Law..."):
        context = get_context(question)

    st.success("Relevant Law Found")

    st.markdown("### 📚 Context")
    st.write(context)

    st.markdown("### 🧠 AI Answer")

    answer = ai_answer(question, context)
    st.write(answer)
