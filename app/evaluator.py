import json
import pandas as pd

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import Faithfulness, AnswerRelevancy, LLMContextPrecisionWithReference

from app.llm_clients import call_openai, call_anthropic, call_gemini

def load_prompts(path):
    with open(path, "r") as f:
        return json.load(f)
    
def evaluate_single(prompt_entry):
    question = prompt_entry["question"]
    context = prompt_entry["context"]
    ground_truth = prompt_entry["ground_truth"]

    openai_result = call_openai(question)
    anthropic_result = call_anthropic(question)
    gemini_result = call_gemini(question)

    samples = [
        SingleTurnSample(
            user_input=question,
            response=openai_result["response"],
            retrieved_contexts=[context], # RAGAS expects a list — wrap single string in brackets
            reference=ground_truth
        ),
        SingleTurnSample(
            user_input=question,
            response=anthropic_result["response"],
            retrieved_contexts=[context],
            reference=ground_truth
        ),
        SingleTurnSample(
            user_input=question,
            response=gemini_result["response"],
            retrieved_contexts=[context],
            reference=ground_truth
        ),
    ]

    metrics = [Faithfulness(), AnswerRelevancy(), LLMContextPrecisionWithReference()]

    results = []
    for i, result in enumerate([openai_result, anthropic_result, gemini_result]):
        dataset = EvaluationDataset(samples=[samples[i]])

        scores = evaluate(dataset=dataset, metrics=metrics) # RAGAS calls OpenAI internally as judge here

        scores_dict = scores.to_pandas().iloc[0].to_dict() # convert RAGAS result to plain dict
        scores_dict["model"] = result["model"]
        scores_dict["latency"] = result["latency"]
        scores_dict["cost_usd"] = result["cost_usd"]

        results.append(scores_dict)

    return results

def run_evaluation(prompts_path):
    prompts = load_prompts(prompts_path)

    all_results = []
    for prompt_entry in prompts:
        results = evaluate_single(prompt_entry)
        all_results.extend(results)  # extend flattens the list and append would nest it

    return pd.DataFrame(all_results)