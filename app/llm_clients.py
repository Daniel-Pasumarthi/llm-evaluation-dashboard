import os
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def call_openai(prompt: str) -> dict:
    start_time = time.time()

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )

    latency = time.time() - start_time

    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    cost = (input_tokens * 2.50 / 1_000_000) + (output_tokens * 10.00 / 1_000_000)

    return {
        "model": "gpt-4o",
        "response": response.choices[0].message.content,
        "latency": round(latency, 2),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost, 6)
    }