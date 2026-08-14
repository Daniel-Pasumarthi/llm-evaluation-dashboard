import json
import pandas as pd

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import Faithfulness, AnswerRelevancy, LLMContextPrecisionWithReference

from app.llm_clients import call_openai, call_anthropic, call_gemini

# The models must see the same context RAGAS scores them against. Without this,
# faithfulness measures accidental overlap with parametric knowledge rather than
# grounding, and penalizes correct-but-unsupported elaboration.
PROMPT_TEMPLATE = """Context:
{context}

Question: {question}

Answer the question using only the information in the context above. Do not end your response with a follow-up question."""


def load_prompts(path):
    with open(path, "r") as f:
        return json.load(f)


def build_prompt(question, context):
    return PROMPT_TEMPLATE.format(context=context, question=question)


def evaluate_single(prompt_entry):
    question = prompt_entry["question"]
    context = prompt_entry["context"]
    ground_truth = prompt_entry["ground_truth"]

    prompt = build_prompt(question, context)

    openai_result = call_openai(prompt)
    anthropic_result = call_anthropic(prompt)
    gemini_result = call_gemini(prompt)

    model_results = [openai_result, anthropic_result, gemini_result]

    metrics = [Faithfulness(), AnswerRelevancy(), LLMContextPrecisionWithReference()]

    results = []
    for result in model_results:
        sample = SingleTurnSample(
            user_input=question,  # the bare question, not the templated prompt
            response=result["response"],
            retrieved_contexts=[context],  # RAGAS expects a list — wrap single string in brackets
            reference=ground_truth
        )

        dataset = EvaluationDataset(samples=[sample])
        scores = evaluate(dataset=dataset, metrics=metrics)  # RAGAS calls OpenAI internally as judge here

        scores_dict = scores.to_pandas().iloc[0].to_dict()  # convert RAGAS result to plain dict
        scores_dict["model"] = result["model"]
        scores_dict["latency"] = result["latency"]
        scores_dict["cost_usd"] = result["cost_usd"]

        results.append(scores_dict)

    return results


def run_evaluation(prompts_path, checkpoint_path="benchmark_partial.csv"):
    prompts = load_prompts(prompts_path)

    all_results = []
    failures = []

    for i, prompt_entry in enumerate(prompts):
        try:
            results = evaluate_single(prompt_entry)
            all_results.extend(results)  # extend flattens the list and append would nest it
            pd.DataFrame(all_results).to_csv(checkpoint_path, index=False)
            print(f"[{i + 1}/{len(prompts)}] ok: {prompt_entry['question'][:50]}")
        except Exception as e:
            failures.append({"question": prompt_entry["question"], "error": f"{type(e).__name__}: {e}"})
            print(f"[{i + 1}/{len(prompts)}] FAILED: {type(e).__name__}: {e}")

    if failures:
        print(f"\n{len(failures)} of {len(prompts)} prompts failed. Completed results are in {checkpoint_path}")

    return pd.DataFrame(all_results)