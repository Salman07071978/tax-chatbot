import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

# ---------------- APP CONFIG ----------------
st.set_page_config(page_title="Pakistan Tax AI V4", layout="wide")

st.title("🇵🇰 Pakistan Tax AI Assistant (V4 - FAST & STABLE)")

# ---------------- LOAD DATA (OPTIMIZED) ----------------
@st.cache_resource
def load_data():
    data = np.load("tax_data.npz", allow_pickle=True)

    chunks = data["chunks"]
    vectors = data["embeddings"]

    # ✅ normalize ONCE (FAST SEARCH)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    return chunks, vectors

chunks, chunk_vectors = load_data()

# ---------------- MODEL (CACHE) ----------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ---------------- GROQ CLIENT ----------------
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ---------------- FAST SEARCH ENGINE ----------------
def get_context(query, top_k=5):

    # normalize query embedding
    q_vec = model.encode([query])
    q_vec = q_vec / np.linalg.norm(q_vec)

    # cosine similarity (fast dot product)
    scores = np.dot(chunk_vectors, q_vec.T).flatten()

    top_idx = np.argsort(scores)[-top_k:][::-1]

    context = "\n\n".join([chunks[i] for i in top_idx])

    return context

# ---------------- AI ENGINE ----------------
def ai_answer(question, context):

    if not context:
        return "No relevant tax section found."

    prompt = f"""
You are a Senior Pakistan Tax Law Expert.

RULES:
- Use ONLY given legal text
- DO NOT repeat question
- DO NOT give generic answers
- Be precise and detailed

FORMAT:

📌 SECTION SUMMARY
📌 APPLICABILITY
📌 CONDITIONS
📌 EXCEPTIONS
📌 CONSEQUENCES
📌 FINAL EXPLANATION

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
            temperature=0.1,
            max_tokens=900
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"AI Error: {str(e)}"

# ---------------- UI ----------------
question = st.text_input("Ask Income Tax or Sales Tax Question")

if st.button("Get Answer"):

    if question.strip():

        with st.spinner("Searching tax database..."):
            context = get_context(question)

        with st.spinner("Generating AI answer..."):
            answer = ai_answer(question, context)

        st.markdown("## 🧠 AI Answer")
        st.write(answer)

        with st.expander("📚 Retrieved Context"):
            st.write(context)
