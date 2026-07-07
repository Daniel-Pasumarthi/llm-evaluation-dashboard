def aggregate_results(results_df):
    df = results_df.copy()

    df["quality_score"] = (
        df["faithfulness"] +
        df["answer_relevancy"] +
        df["llm_context_precision_with_reference"]
    ) / 3

    aggregated = df.groupby("model").agg(
        avg_cost=("cost_usd", "mean"),
        avg_quality=("quality_score", "mean")
    ).reset_index()

    return aggregated