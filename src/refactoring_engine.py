"""
Refactoring Engine
==================
Calls the OpenRouter API to refactor Python functions using the recommended LLM.

Supports all 5 models from the recommender:
  - Claude Opus, Claude Sonnet 4.6, Gemini 3.1 Pro, Gemini Flash, GPT-OSS

The refactoring prompt instructs the LLM to:
  - Fix code smells (long method, missing docstrings, missing type hints, etc.)
  - Improve code quality while maintaining identical behavior
  - Return ONLY the refactored code (no explanations, no markdown fences)
"""

import os
import sys
import json
import time
import requests
from pathlib import Path

# ── OpenRouter model slugs (mapped from our internal keys) ───────────────────
OPENROUTER_MODEL_MAP = {
    "claude_opus":       "anthropic/claude-opus-4.5",
    "claude_sonnet_4_6": "anthropic/claude-sonnet-4.5",
    "gemini_3_1_pro":    "google/gemini-3.1-pro-preview",
    "gemini_flash":      "google/gemini-3.5-flash",
    "gpt_oss":           "openai/gpt-oss-120b",
}

DEFAULT_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Refactoring prompt template
REFACTOR_PROMPT_TEMPLATE = """You are an expert Python code refactoring assistant. Your task is to refactor the following Python function to improve its quality.

## Requirements:
1. **Maintain identical behavior** - The function must produce the same outputs for the same inputs
2. **Fix these code smells** if present:
   - Long method (break into smaller helper functions if > 20 lines)
   - Long parameter list (use *args, **kwargs, or data classes if > 4 params)
   - Deep nesting (flatten with early returns / guard clauses)
   - Magic numbers (replace with named constants)
   - Missing docstring (add Google-style docstring)
   - Missing type hints (add type annotations for params and return)
   - Long lines (> 79 chars)
   - Commented-out code (remove it)
   - Poor naming (rename single-letter variables to meaningful names)
3. **Add a proper docstring** with Args/Returns/Raises sections
4. **Add type hints** to all parameters and return value
5. **Keep imports** that the function depends on

## Output Format:
Return ONLY the refactored Python code. No explanations, no markdown code fences (no ```), no introductory text. Just the raw Python code.

## Code to Refactor:

```python
{code}
```"""


import time
from typing import Any, Dict

DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT = 120
RATE_LIMIT_BACKOFF_BASE = 5
RATE_LIMIT_MAX_WAIT = 60
RETRY_WAIT = 2
TIMEOUT_RETRY_WAIT = 3
CONNECTION_RETRY_WAIT = 5
EXCEPTION_RETRY_WAIT = 2


def _build_headers(api_key: str) -> Dict[str, str]:
    """Create request headers for OpenRouter API."""
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/coderefactor-ai",
        "X-Title": "CodeRefactor AI Pipeline",
    }


def _build_payload(model_slug: str, code: str) -> Dict[str, Any]:
    """Create JSON payload for the refactoring request."""
    prompt = REFACTOR_PROMPT_TEMPLATE.format(code=code)
    return {
        "model": model_slug,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert Python code refactoring assistant. "
                    "Return ONLY clean Python code with no explanations or markdown fences."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 4000,
    }


def _handle_rate_limit(attempt: int) -> None:
    """Sleep with exponential backoff when rate limited."""
    wait = min(2 ** attempt * RATE_LIMIT_BACKOFF_BASE, RATE_LIMIT_MAX_WAIT)
    print(f"    Rate limited. Waiting {wait}s...")
    time.sleep(wait)


def _process_success(
    data: Dict[str, Any], model_key: str, model_slug: str
) -> Dict[str, Any]:
    """Extract refactored code and usage from a successful response."""
    choices = data.get("choices", [])
    if not choices:
        raise ValueError("No choices in response")

    raw_code = choices[0].get("message", {}).get("content", "").strip()
    refactored = _clean_code_output(raw_code)
    usage = data.get("usage")

    return {
        "success": True,
        "refactored_code": refactored,
        "model_key": model_key,
        "model_slug": model_slug,
        "error": None,
        "usage": usage,
    }


def refactor_function(
    code: str,
    model_key: str,
    api_key: str,
    api_url: str = DEFAULT_API_URL,
    max_retries: int = DEFAULT_MAX_RETRIES,
    timeout: int = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """Refactor Python code using an LLM via the OpenRouter API.

    Args:
        code: Source code to refactor.
        model_key: Identifier for the desired model (e.g., "claude_opus").
        api_key: OpenRouter API key.
        api_url: Endpoint for the OpenRouter API.
        max_retries: Number of retry attempts on failure.
        timeout: Request timeout in seconds.

    Returns:
        A dictionary containing success status, refactored code, model info,
        any error message, and token usage.

    Raises:
        ValueError: If the API response lacks expected fields.
    """
    model_slug = OPENROUTER_MODEL_MAP.get(model_key, model_key)
    headers = _build_headers(api_key)
    payload = _build_payload(model_slug, code)

    last_error: str | None = None

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                api_url, headers=headers, json=payload, timeout=timeout
            )

            if response.status_code == 429:
                _handle_rate_limit(attempt)
                continue

            if response.status_code != 200:
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                if attempt < max_retries:
                    time.sleep(RETRY_WAIT)
                continue

            data = response.json()
            return _process_success(data, model_key, model_slug)

        except requests.exceptions.Timeout:
            last_error = f"Request timeout (attempt {attempt}/{max_retries})"
            if attempt < max_retries:
                time.sleep(TIMEOUT_RETRY_WAIT)

        except requests.exceptions.ConnectionError as exc:
            last_error = f"Connection error: {exc}"
            if attempt < max_retries:
                time.sleep(CONNECTION_RETRY_WAIT)

        except Exception as exc:
            last_error = f"Unexpected error: {exc}"
            if attempt < max_retries:
                time.sleep(EXCEPTION_RETRY_WAIT)

    return {
        "success": False,
        "refactored_code": None,
        "model_key": model_key,
        "model_slug": model_slug,
        "error": last_error,
        "usage": None,
    }


def _clean_code_output(text: str) -> str:
    """Remove markdown code fences if present."""
    text = text.strip()
    # Remove leading ```python or ```py
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line if it starts with ```
        if lines[0].startswith("```"):
            lines = lines[1:]
        # Remove last line if it ends with ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def verify_refactored_code(original: str, refactored: str) -> bool:
    """
    Basic sanity check: ensure the refactored code is valid Python
    and contains a function definition.
    """
    if not refactored or len(refactored.strip()) < 10:
        return False

    # Must contain at least one function definition
    if "def " not in refactored:
        return False

    # Try to compile as valid Python
    try:
        compile(refactored, "<refactored>", "exec")
        return True
    except SyntaxError:
        return False
