import streamlit as st
from app.evaluator import load_prompts, evaluate_single, run_evaluation
from app.visualizations import aggregate_results, create_cost_quality_chart, create_latency_chart
from app.langsmith_logger import log_evaluation_run
from app.llm_clients import MODELS


def main():
    st.title("Multi-Model LLM Evaluation Dashboard")
    st.write(f"Compare {', '.join(MODELS.values())} on the same prompt using RAGAS metrics.")

    prompts = load_prompts("data/prompts/qa_prompts.json")

    questions = [p["question"] for p in prompts]
    selected_question = st.selectbox("Choose a prompt to evaluate:", questions)

    selected_prompt = next(p for p in prompts if p["question"] == selected_question)

    if st.button("Run Evaluation"):
        with st.spinner("Calling all three models and scoring with RAGAS..."):
            results = evaluate_single(selected_prompt)
            log_evaluation_run(selected_prompt, results)

        st.success("Evaluation Complete!")
        st.dataframe(results)

    st.divider()
    st.subheader("Full Benchmark: Cost vs. Quality")
    st.caption(
        "Runs all 10 prompts across all 3 models — roughly 10 minutes and 30 API calls. "
        "Memory-intensive; if it fails on the hosted app, run `python run_benchmark.py` locally."
    )

    if st.button("Run Full Benchmark"):
        with st.spinner("Running all 10 prompts across all 3 models..."):
            full_results_df = run_evaluation("data/prompts/qa_prompts.json")
            aggregated = aggregate_results(full_results_df)
            cost_quality_fig = create_cost_quality_chart(aggregated)
            latency_fig = create_latency_chart(aggregated)

        st.success("Benchmark complete!")
        st.plotly_chart(cost_quality_fig)
        st.plotly_chart(latency_fig)
        st.dataframe(full_results_df)


if __name__ == "__main__":
    main()