# Multi-Model LLM Evaluation Dashboard

A Streamlit dashboard that benchmarks **GPT-4o, Claude Sonnet, and Gemini Flash 2.5** on identical prompts using **RAGAS metrics** — built to answer *which model should you actually use?* with numbers instead of eyeballing outputs.

## How it's built

- **Standardized multi-provider client layer** — one consistent interface across three vendor SDKs, so adding a fourth model is a small diff, not a rewrite
- **RAGAS evaluation pipeline** scoring faithfulness, answer relevancy, and context precision per response, per model
- **Full LangSmith observability** — every benchmark run is logged and reproducible, not a one-off script execution
- **Interactive results UI** so quality-to-cost tradeoffs are visible at a glance, not buried in a notebook

## A debugging note worth mentioning

While building this, I diagnosed a scoring artifact in the RAGAS pipeline traced to model-specific response patterns — differences in how each model formats or hedges answers were quietly skewing scores. I corrected the methodology before drawing conclusions from it. Evaluation frameworks need to be evaluated too.

## Stack

`Python` · `OpenAI API` · `Anthropic API` · `Gemini Pro` · `Streamlit` · `LangSmith` · `RAGAS` · `Docker`

## Running locally

```bash
git clone https://github.com/Daniel-Pasumarthi/llm-evaluation-dashboard.git
cd llm-evaluation-dashboard
cp .env.example .env   # add your API keys
docker-compose up --build
```
