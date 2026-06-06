import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

st.set_page_config(page_title="Pakistan Tax AI V3", layout="wide")

st.title("🇵🇰 Pakistan Tax AI Assistant (V3 - Fixed Search + AI)")

# ---------------- LOAD DATA ----------------
@st.cache_resource
def load_data():
    data = np.load("tax_data.npz", allow_pickle=True)
    return data["chunks"], data["embeddings"]

chunks, chunk_vectors = load_data()

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ---------------- GROQ CLIENT ----------------
client = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
)

# ---------------- HYBRID SEARCH (FIXED) ----------------
def get_context(query, top_k=5):

    query_lower = query.lower()

    # ---- semantic search ----
    q_vec = model.encode([query], normalize_embeddings=True)
    sem_scores = np.dot(chunk_vectors, q_vec.T).flatten()

    # ---- keyword boost (IMPORTANT FIX) ----
    keyword_scores = []

    for chunk in chunks:
        score = 0
        chunk_lower = chunk.lower()

        # direct match boost
        if query_lower in chunk_lower:
            score += 15

        # section number boost
        words = query_lower.split()
        for w in words:
            if w.isdigit() and f"section {w}" in chunk_lower:
                score += 25

        keyword_scores.append(score)

    keyword_scores = np.array(keyword_scores)

    # ---- combine ----
    final_scores = sem_scores + keyword_scores

    top_idx = np.argsort(final_scores)[-top_k:][::-1]

    context = "\n\n".join([chunks[i] for i in top_idx])

    return context

# ---------------- AI ANSWER ----------------
def ai_answer(question, context):

    prompt = f"""
You are a senior Pakistan Tax Law expert.

Instructions:
- Use ONLY provided legal text
- Explain in simple and clear points
- Mention conditions, exceptions, consequences
- If answer not found, say: "Not clearly available in provided tax law"

LEGAL TEXT:
{context}

QUESTION:
{question}
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=800
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"AI Error: {str(e)}"

# ---------------- UI ----------------
question = st.text_input("Ask Income Tax or Sales Tax Question")

if st.button("Get Answer"):

    if question.strip():

        with st.spinner("Searching tax law..."):
            context = get_context(question)

        with st.spinner("Generating AI answer..."):
            answer = ai_answer(question, context)

        st.markdown("## 🧠 AI Answer")
        st.write(answer)

        with st.expander("📚 Retrieved Legal Context"):
            st.write(context)
