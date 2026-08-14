# Multi-Model LLM Evaluation Dashboard

Picking an LLM for production shouldn't come down to vibes. This dashboard sends identical prompts to three current-generation models, scores the responses with RAGAS, and reports what each one actually costs and how long it takes — so the choice can be made on numbers.

**[Live demo →](https://multi-llm-evaluation-dashboard.streamlit.app)**

Models compared: `gpt-5.6-terra` · `claude-sonnet-5` · `gemini-3.6-flash` — deliberately tier-matched (all mid-tier, $1.50–$2.00 per million input tokens) so the comparison is fair.

---

## What I found

Ten prompts × three models × three RAGAS metrics = 30 scored responses per run. Two independent runs.

| | Cost / response | Median latency | Faithfulness |
|---|---|---|---|
| gpt-5.6-terra | $0.00113 | 1.63s | 1.000 |
| claude-sonnet-5 | $0.00247 | 2.56s | 0.946 |
| gemini-3.6-flash | $0.00413 | 2.96s | 0.981 |

**Cost varies 3.7x between models priced almost identically.** Gemini has the lowest list price of the three and the highest real cost. The reason is thinking tokens: they bill at the output rate but are reported in `thoughts_token_count`, not `candidates_token_count`. On one short prompt Gemini used 36 candidate tokens and 452 thinking tokens. Counting only the visible output understated its cost by roughly 13x.

**Quality doesn't separate them.** All three land within 0.05 on every metric and context precision is a flat 1.0 across the board. For grounded short-context Q&A, these models are interchangeable — cost and latency are the real decision criteria.

Claude's 0.933–0.946 faithfulness replicated across both runs, but reading the responses shows it isn't hallucination. It writes more interpretive prose ("this makes it an absolute limit"), and RAGAS's claim decomposer extracts those as unsupported claims. A style difference meeting a metric quirk, not a quality gap.

## Two bugs worth documenting

My first numbers were wrong, and finding out why was the most useful part of building this.

**The models never saw the context.** I was passing only the question to each model, then scoring the response against a context it had never received. Faithfulness was measuring accidental overlap with parametric knowledge — and punishing correct, well-informed elaboration. The most thorough answer scored worst. Templating the context into the prompt fixed it and moved mean faithfulness from 0.68 to 0.98.

**Answer relevancy was returning hard zeros.** RAGAS computes it by reverse-generating questions from the response and comparing them to the original. When a model ended with a conversational follow-up question, the mechanism collapsed to zero. Adding one instruction to the prompt eliminated it.

Both are the same category of failure: the metric ran fine and returned a plausible number that meant something other than its name suggested.

## Architecture

```
Prompt + context
      │
      ├──→ OpenAI ─┐
      ├──→ Anthropic ─┼──→ RAGAS scoring ──→ Streamlit UI
      └──→ Google ────┘         │
                                └──→ LangSmith traces
```

| File | Role |
|---|---|
| `app/llm_clients.py` | Provider wrappers. Model IDs and pricing live here as constants — single source of truth. |
| `app/evaluator.py` | Prompt construction and RAGAS scoring. Checkpoints per prompt so one failure doesn't discard a whole run. |
| `app/visualizations.py` | Plotly charts, y-axis pinned to [0,1] so a 0.05 spread can't be drawn as a chasm. |
| `app/langsmith_logger.py` | Run tracing. |
| `streamlit_app.py` | Root entry point — keeps `app` importable across local, Docker, and Streamlit Cloud. |

Stack: Python 3.11 · Streamlit · RAGAS · LangSmith · Plotly · Docker · tenacity

## Running it

```bash
git clone https://github.com/Daniel-Pasumarthi/llm-evaluation-dashboard.git
cd llm-evaluation-dashboard
cp .env.example .env          # add your three API keys
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Or with Docker:

```bash
docker-compose up --build
```

Full benchmark (~10 minutes, ~$0.25 in API calls):

```bash
python run_benchmark.py
```

Results land in `benchmark_results.csv`. A completed run is committed to this repo so the numbers above can be checked without spending anything.

## Known limitations

- **n=10.** Enough to establish cost and latency, not enough to claim a quality ranking. Differences under ~0.05 are noise here.
- **No temperature control.** GPT-5.6 rejects non-default values and Sonnet 5 has deprecated the parameter. I removed it from all three rather than controlling only one, which trades reproducibility for comparability. Tighter reproducibility would mean N runs per prompt with error bars.
- **No retrieval.** Contexts are hand-authored, so `context_precision` has nothing to evaluate and sits at 1.0. It stays in as a demonstration of the metric, not as a result.
- **Single-session state.** Results live in memory. Scaling this would mean an async queue for the LLM calls and Postgres for results.

## What I'd do next

Adversarial contexts — incomplete, contradictory, or partially irrelevant — to make faithfulness actually discriminate. That's where these models would start to differ, and where the evaluation would stop measuring style.