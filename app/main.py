import streamlit as st
from app.evaluator import load_prompts, evaluate_single, run_evaluation
from app.visualizations import aggregate_results, create_cost_quality_chart
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

st.divider()
st.subheader("Full Benchmark: Cost vs. Quality")
st.caption("Runs all 10 prompts across all 3 models — takes 1-2 minutes and calls all three APIs.")

if st.button("Run Full Benchmark"):
    with st.spinner("Running all 10 prompts across all 3 models..."):
        full_results_df = run_evaluation("data/prompts/qa_prompts.json")
        aggregated = aggregate_results(full_results_df)
        fig = create_cost_quality_chart(aggregated)

    st.success("Benchmark complete!")
    st.plotly_chart(fig)