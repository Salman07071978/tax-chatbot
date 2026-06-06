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
