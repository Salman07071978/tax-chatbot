import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

st.set_page_config(page_title="Pakistan Tax AI V2", layout="wide")

st.title("🇵🇰 Pakistan Tax AI Assistant (V2 - Smart)")

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

# ---------------- RETRIEVAL (TOP 5) ----------------
def get_context(query, top_k=5):

    q_vec = model.encode([query], normalize_embeddings=True)

    scores = np.dot(chunk_vectors, q_vec.T).flatten()

    top_idx = np.argsort(scores)[-top_k:][::-1]

    context = "\n\n".join([chunks[i] for i in top_idx])

    return context

# ---------------- AI ANSWER ----------------
def ai_answer(question, context):

    prompt = f"""
You are a Pakistan Tax Law Expert AI.

Your job:
- Read the legal text carefully
- Explain in simple words
- Give point-wise answer
- Mention conditions clearly

If answer is not in text, say:
"Not found in provided tax law."

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
            max_tokens=700
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"AI Error: {str(e)}"

# ---------------- UI ----------------
question = st.text_area("Ask Income Tax or Sales Tax Question")

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
