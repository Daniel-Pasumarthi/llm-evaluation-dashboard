import os
import time
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

MODELS = {
    "openai": "gpt-5.6-terra",
    "anthropic": "claude-sonnet-5",
    "gemini": "gemini-3.6-flash",
}

# USD per million tokens, (input, output). Verified against provider pricing pages 2026-08-13.
PRICING = {
    "openai": (2.00, 12.00),
    "anthropic": (2.00, 10.00),
    "gemini": (1.50, 7.50),
}

# Temperature is deliberately not set. GPT-5.6 rejects any non-default value and
# Sonnet 5 has deprecated the parameter, so pinning it on Gemini alone would give
# one model greedy decoding and the other two sampling — a confound, not a control.
MAX_TOKENS = 4096

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def _cost(provider: str, input_tokens: int, output_tokens: int) -> float:
    in_rate, out_rate = PRICING[provider]
    return (input_tokens * in_rate / 1_000_000) + (output_tokens * out_rate / 1_000_000)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_openai(prompt: str) -> dict:
    start_time = time.time()

    response = openai_client.chat.completions.create(
        model=MODELS["openai"],
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=MAX_TOKENS
    )

    latency = time.time() - start_time

    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens

    return {
        "model": MODELS["openai"],
        "response": response.choices[0].message.content,
        "latency": round(latency, 2),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(_cost("openai", input_tokens, output_tokens), 6)
    }


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_anthropic(prompt: str) -> dict:
    start_time = time.time()

    response = anthropic_client.messages.create(
        model=MODELS["anthropic"],
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}]
    )

    latency = time.time() - start_time

    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens

    return {
        "model": MODELS["anthropic"],
        "response": response.content[0].text,
        "latency": round(latency, 2),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(_cost("anthropic", input_tokens, output_tokens), 6)
    }


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_gemini(prompt: str) -> dict:
    start_time = time.time()

    response = gemini_client.models.generate_content(
        model=MODELS["gemini"],
        contents=prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=MAX_TOKENS
        )
    )

    latency = time.time() - start_time

    usage = response.usage_metadata
    input_tokens = usage.prompt_token_count
    # Gemini 3.x are thinking models; thinking tokens bill at the output rate but are
    # reported separately from candidates_token_count.
    output_tokens = (usage.candidates_token_count or 0) + (getattr(usage, "thoughts_token_count", 0) or 0)

    return {
        "model": MODELS["gemini"],
        "response": response.text,
        "latency": round(latency, 2),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(_cost("gemini", input_tokens, output_tokens), 6)
    }