import os
import time
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
from google import genai
from google.genai import types

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def call_openai(prompt: str) -> dict:
    start_time = time.time()

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=2048
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

def call_anthropic(prompt: str) -> dict:
    start_time = time.time()

    response = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )

    latency = time.time() - start_time

    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    cost = (input_tokens * 3.00 / 1_000_000) + (output_tokens * 15.00 / 1_000_000)

    return {
        "model": "claude-sonnet-4-6",
        "response": response.content[0].text,
        "latency": round(latency, 2),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost, 6)
    }

def call_gemini(prompt: str) -> dict:
    start_time = time.time()

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=2048
        )
    )

    latency = time.time() - start_time

    input_tokens = response.usage_metadata.prompt_token_count
    output_tokens = response.usage_metadata.candidates_token_count
    cost = (input_tokens * 0.30 / 1_000_000) + (output_tokens * 2.50 / 1_000_000)

    return {
        "model": "gemini-2.5-flash",
        "response": response.text,
        "latency": round(latency, 2),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost, 6)
    }
