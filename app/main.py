import streamlit as st
from app.evaluator import load_prompts, evaluate_single
from app.langsmith_logger import log_evaluation_run

st.title("Multi-Model LLM Evaluation Dashboard")
st.write("Compare GPT-4o, Claude Sonnet, and Gemini Flash on the same prompt using RAGAS metrics.")

prompts = load_prompts("data/prompts/qa_prompts.json")

questions = [p["question"] for p in prompts]
selected_question = st.selectbox("Choose a prompt to evaluate:", questions)

selected_prompt = next(p for p in prompts if p["question"]==selected_question)

if st.button("Run Evaluation"):
    with st.spinner("Calling all three models and scoring with RAGAS..."):
        results = evaluate_single(selected_prompt)
        log_evaluation_run(selected_prompt, results)

    st.success("Evaluation Complete!")
    st.dataframe(results)