import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Pakistan Tax AI", layout="wide")

st.title("🇵🇰 Pakistan Tax Assistant (Final Stable + Smart Version)")

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

# ---------------- SMART ANSWER ENGINE ----------------
def answer_engine(question, context):

    return f"""
📚 RELEVANT TAX LAW SECTION:
--------------------------------
{context}

🧠 SIMPLE EXPLANATION:
--------------------------------
This section of tax law is relevant to your query.

It explains legal conditions and compliance requirements mentioned above.

If the conditions in the clause (such as limitations, dates, or ownership rules) are not satisfied, then the benefit or exemption may be withdrawn under tax law.

💡 KEY POINTS:
- Always check conditions carefully (a), (b), (c)
- Time limits are very important in tax law
- Ownership/structure changes may affect eligibility
- Non-compliance can lead to invalidation of claim

📌 NOTE:
This is an AI-generated simplified explanation for understanding purposes. Always verify with official FBR rules or a tax consultant for legal decisions.
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
