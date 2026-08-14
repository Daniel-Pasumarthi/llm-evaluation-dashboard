import pandas as pd
import plotly.express as px


def aggregate_results(results_df):
    df = results_df.copy()
    df["quality_score"] = (
        df["faithfulness"] +
        df["answer_relevancy"] +
        df["llm_context_precision_with_reference"]
    ) / 3
    aggregated = df.groupby("model").agg(
        avg_cost=("cost_usd", "mean"),
        avg_quality=("quality_score", "mean"),
        median_latency=("latency", "median")
    ).reset_index()
    return aggregated


def create_cost_quality_chart(aggregated_df):
    fig = px.scatter(
        aggregated_df,
        x="avg_cost",
        y="avg_quality",
        color="model",
        text="model",
        labels={
            "avg_cost": "Average Cost per Response (USD)",
            "avg_quality": "Average Quality Score (RAGAS composite)"
        },
        title="Model Cost vs. Quality Tradeoff"
    )
    fig.update_traces(marker=dict(size=14), textposition="top center")

    # Pinned to the metric's full range. Autoscaling a 0.05 spread across the plot
    # height makes near-identical models look dramatically different.
    fig.update_yaxes(range=[0, 1])
    fig.update_xaxes(rangemode="tozero")

    return fig


def create_latency_chart(aggregated_df):
    fig = px.bar(
        aggregated_df.sort_values("median_latency"),
        x="model",
        y="median_latency",
        color="model",
        labels={"median_latency": "Median Latency (seconds)", "model": ""},
        title="Median Response Latency by Model"
    )
    fig.update_layout(showlegend=False)
    return fig