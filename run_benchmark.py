"""Run the full benchmark and write results to CSV. Run locally, not on Streamlit Cloud."""

from app.evaluator import run_evaluation

OUTPUT_PATH = "benchmark_results.csv"

df = run_evaluation("data/prompts/qa_prompts.json")
df.to_csv(OUTPUT_PATH, index=False)

print(f"\nWrote {len(df)} rows to {OUTPUT_PATH}")
print(f"\nCost by model:\n{df.groupby('model')['cost_usd'].agg(['sum', 'mean'])}")
print(f"\nLatency by model:\n{df.groupby('model')['latency'].agg(['mean', 'median', 'min', 'max'])}")
print(f"\nFaithfulness by model:\n{df.groupby('model')['faithfulness'].agg(['mean', 'min', 'max'])}")