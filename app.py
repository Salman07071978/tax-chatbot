import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Pakistan Tax AI", layout="wide")

st.title("🇵🇰 Pakistan Tax Assistant (Accurate RAG Version)")

# ---------------- LOAD DATA ----------------
data = np.load("tax_data.npz", allow_pickle=True)

chunks = data["chunks"]
chunk_vectors = data["embeddings"]

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ---------------- BETTER SEARCH (TOP 3 RESULTS) ----------------
def get_context(query):

    # encode query
    q_vec = model.encode([query], normalize_embeddings=True)

    # normalize chunk vectors
    chunk_vecs = model.encode(chunks, normalize_embeddings=True)

    # similarity scores
    scores = np.dot(chunk_vecs, q_vec.T).flatten()

    # top 3 results
    top_idx = np.argsort(scores)[-3:][::-1]

    context = ""
    for i in top_idx:
        context += chunks[i] + "\n\n"

    return context

# ---------------- SMART ANSWER ENGINE ----------------
def answer_engine(question, context):

    return f"""
📚 RELEVANT TAX LAW CONTEXT
--------------------------------
{context}

🧠 SIMPLE EXPLANATION
--------------------------------
Based on the above legal provisions:

✔ This section defines tax conditions, eligibility, or restrictions.
✔ If any condition (such as ownership, production, or time limit) is violated,
  then the tax benefit or exemption becomes invalid.
✔ All clauses must be read together for correct interpretation.

💡 KEY UNDERSTANDING POINTS:
- Tax laws depend on strict conditions (a), (b), (c)
- Time limits are legally binding (e.g. 30 June 2026 rules)
- Ownership changes can cancel eligibility
- Asset disposal can invalidate exemptions

📌 FINAL NOTE:
This is a simplified explanation for understanding.
For legal accuracy, always verify with updated FBR rules or a tax advisor.
"""

# ---------------- UI ----------------
question = st.text_input("Ask Income Tax or Sales Tax Question")

if question:

    with st.spinner("Searching tax laws..."):
        context = get_context(question)

    st.success("Relevant sections found")

    st.markdown("### 📚 Context (Top 3 Matches)")
    st.write(context)

    st.markdown("### 🧠 Answer")
    st.write(answer_engine(question, context))
