import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Pakistan Tax AI", layout="wide")

st.title("🇵🇰 Pakistan Tax Assistant (Stable Final Version)")

# ---------------- LOAD DATA ----------------
data = np.load("tax_data.npz", allow_pickle=True)

chunks = data["chunks"]
chunk_vectors = data["embeddings"]

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ---------------- SEARCH FUNCTION ----------------
def get_context(query):
    q_vec = model.encode([query])
    scores = np.dot(chunk_vectors, q_vec.T)
    idx = np.argmax(scores)
    return chunks[idx]

# ---------------- SIMPLE ANSWER ENGINE ----------------
def answer_engine(question, context):

    return f"""
📚 Relevant Tax Law Section:

{context}

🧠 Explanation:
This section is relevant to your question. It should be interpreted according to the Income Tax / Sales Tax provisions mentioned above.

💡 Tip:
For accurate legal interpretation, always cross-check with FBR official rules or updated amendments.
"""

# ---------------- UI ----------------
question = st.text_input("Ask Income Tax or Sales Tax Question")

if question:

    with st.spinner("Searching tax laws..."):
        context = get_context(question)

    st.success("Relevant section found")

    st.markdown("### 📚 Context")
    st.write(context)

    st.markdown("### 🧠 Answer")
    st.write(answer_engine(question, context))
