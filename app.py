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

# ---------------- PROFESSIONAL UI ----------------

st.markdown("""
<style>

.main-header{
    background:#0E4D92;
    color:white;
    padding:20px;
    border-radius:12px;
    text-align:center;
    font-size:32px;
    font-weight:bold;
    margin-bottom:10px;
}

.sub-header{
    text-align:center;
    color:#666;
    margin-bottom:25px;
}

.result-box{
    border:1px solid #dcdcdc;
    border-radius:10px;
    padding:20px;
    background-color:#f8f9fa;
    margin-top:15px;
}

.result-title{
    font-size:20px;
    font-weight:bold;
    color:#0E4D92;
    margin-bottom:15px;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
Pakistan Tax AI Assistant
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="sub-header">
AI-Powered Search and Interpretation of Pakistan Tax Laws
</div>
""", unsafe_allow_html=True)

question = st.text_area(
    "Enter Your Tax Question",
    height=120,
    placeholder="Example: Explain Section 49 exemption..."
)

col1, col2, col3 = st.columns([1,2,1])

with col2:
    search_btn = st.button(
        "Generate Legal Opinion",
        use_container_width=True
    )

if search_btn:

    if question.strip():

        with st.spinner("Searching legal database..."):
            context = get_context(question)

        with st.spinner("Generating legal analysis..."):
            answer = ai_answer(question, context)

        st.markdown("""
        <div class="result-box">
            <div class="result-title">
                Legal Analysis
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(answer)

        with st.expander("View Legal Source Text"):
            st.text(context)
