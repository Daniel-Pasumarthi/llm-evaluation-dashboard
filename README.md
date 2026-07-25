# Multi-Model LLM Evaluation Dashboard

A Streamlit dashboard that benchmarks **GPT-4o, Claude Sonnet, and Gemini Flash 2.5** across identical prompts using **RAGAS evaluation metrics** and full **LangSmith** observability — built to answer a simple question with real data: *which model should you actually use?*

## What it does

Running the same prompt set against three commercial LLM APIs and eyeballing the outputs doesn't scale — and it doesn't give you numbers you can defend. This dashboard standardizes that comparison:

- Benchmarks **GPT-4o**, **Claude Sonnet**, and **Gemini Flash 2.5** on identical prompts
- Scores every response on **RAGAS metrics**: faithfulness, answer relevancy, and context precision
- Surfaces full results in an interactive Streamlit UI so quality-to-cost tradeoffs are visible at a glance, not buried in a notebook

## Architecture

- **Standardized multi-provider client layer** — one consistent interface across three different vendor SDKs, so adding a fourth model later is a small diff, not a rewrite
- **LangSmith observability** — every benchmark run is logged and reproducible, not a one-off script execution
- **RAGAS evaluation pipeline** — faithfulness, answer relevancy, and context precision computed per response, per model

## A debugging note worth mentioning

While building this, I diagnosed a scoring artifact in the RAGAS pipeline that traced back to model-specific response patterns (differences in how each model formats or hedges answers were quietly skewing scores). I corrected the benchmark methodology before drawing any conclusions from it — a reminder that evaluation frameworks need to be evaluated too.

## Stack

`Python` · `OpenAI API` · `Anthropic API` · `Gemini Pro` · `Streamlit` · `LangSmith` · `RAGAS` · `Docker`

## Running locally

```bash
git clone https://github.com/Daniel-Pasumarthi/llm-evaluation-dashboard.git
cd llm-evaluation-dashboard
cp .env.example .env   # add your API keys
docker-compose up --build
```
