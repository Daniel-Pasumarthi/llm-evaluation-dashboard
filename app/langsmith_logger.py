import os
import uuid
from langsmith import Client
from dotenv import load_dotenv

load_dotenv()

client = Client()

def log_evaluation_run(prompt_entry: dict, results: list) -> None:
    """
    Logs one evaluation run to LangSmith.

    Args:
        prompt_entry: dict with question, context, and ground_truth
        results: list of 3 dicts, one per model, with LLM output and RAGAS scores

    Returns:
        None
    """

    inputs = {
        "id": prompt_entry["id"],
        "question": prompt_entry["question"],
        "context": prompt_entry["context"],
    }

    outputs = {"results": results}

    run_id = uuid.uuid4()

    try:
        client.create_run(
            id=run_id,
            name="llm_evaluation_run",
            run_type="chain",
            inputs=inputs,
            project_name=os.getenv("LANGCHAIN_PROJECT"),
        )

        client.update_run(
            run_id,
            outputs=outputs,
        )
    except Exception as e:
        print(f"LangSmith logging failed: {e}")