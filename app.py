import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

# ---------------- APP CONFIG ----------------
st.set_page_config(page_title="Pakistan Tax AI V3", layout="wide")

st.title("🇵🇰 Pakistan Tax AI Assistant (V3 - Stable & Accurate)")

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
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ---------------- HYBRID SEARCH + CONFIDENCE ----------------
def get_context(query, top_k=5):

    query_lower = query.lower()

    q_vec = model.encode([query], normalize_embeddings=True)
    sem_scores = np.dot(chunk_vectors, q_vec.T).flatten()

    keyword_scores = []

    for chunk in chunks:
        score = 0
        chunk_lower = chunk.lower()

        # direct match boost
        if query_lower in chunk_lower:
            score += 30

        # section number boost
        for word in query_lower.split():
            if word.isdigit() and f"section {word}" in chunk_lower:
                score += 50

        keyword_scores.append(score)

    keyword_scores = np.array(keyword_scores)

    final_scores = sem_scores + keyword_scores

    top_idx = np.argsort(final_scores)[-top_k:][::-1]

    best_score = final_scores[top_idx[0]]

    # ---------------- CONFIDENCE FILTER ----------------
    if best_score < 0.40:
        return None

    context = "\n\n".join([chunks[i] for i in top_idx])

    return context

# ---------------- AI ANSWER ENGINE ----------------
def ai_answer(question, context):

    # if no context found
    if context is None:
        return """
📌 RESULT:
No relevant section found in the uploaded tax law database.

💡 TRY:
- Use correct section number (e.g. Section 49)
- Try full wording
- Or check spelling
"""

    prompt = f"""
You are a SENIOR Pakistan Tax Law Expert.

RULES:
- Use ONLY given legal text
- Do NOT repeat questions
- Do NOT say "not available in general law"
- Give detailed structured explanation

FORMAT:

📌 SECTION SUMMARY:
📌 APPLICABILITY:
📌 CONDITIONS:
📌 EXCEPTIONS:
📌 CONSEQUENCES:
📌 FINAL EXPLANATION:

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

        with st.spinner("Searching tax laws..."):
            context = get_context(question)

        with st.spinner("Generating answer..."):
            answer = ai_answer(question, context)

        st.markdown("## 🧠 AI Answer")
        st.write(answer)

        with st.expander("📚 Retrieved Context"):
            st.write(context)
