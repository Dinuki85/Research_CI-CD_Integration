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
from typing import Any, Dict, Optional

DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 4000
BACKOFF_BASE_SECONDS = 5
MAX_BACKOFF_WAIT = 60
RETRY_SLEEP_SECONDS = 2
TIMEOUT_SLEEP_SECONDS = 3
CONNECTION_SLEEP_SECONDS = 5
UNEXPECTED_SLEEP_SECONDS = 2


def _build_headers(api_key: str) -> Dict[str, str]:
    """Create request headers for OpenRouter API."""
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/coderefactor-ai",
        "X-Title": "CodeRefactor AI Pipeline",
    }


def _build_payload(model_slug: str, prompt: str) -> Dict[str, Any]:
    """Create JSON payload for the OpenRouter request."""
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
        "temperature": DEFAULT_TEMPERATURE,
        "max_tokens": DEFAULT_MAX_TOKENS,
    }


def _extract_refactored_code(data: Dict[str, Any]) -> Optional[str]:
    """Extract and clean the refactored code from API response."""
    choices = data.get("choices", [])
    if not choices:
        return None
    raw_code = choices[0].get("message", {}).get("content", "").strip()
    return _clean_code_output(raw_code)


def _handle_success(
    refactored_code: str,
    model_key: str,
    model_slug: str,
    usage: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Return a success result dictionary."""
    return {
        "success": True,
        "refactored_code": refactored_code,
        "model_key": model_key,
        "model_slug": model_slug,
        "error": None,
        "usage": usage,
    }


def _handle_failure(
    model_key: str,
    model_slug: str,
    error_message: Optional[str],
) -> Dict[str, Any]:
    """Return a failure result dictionary."""
    return {
        "success": False,
        "refactored_code": None,
        "model_key": model_key,
        "model_slug": model_slug,
        "error": error_message,
        "usage": None,
    }


def refactor_function(
    code: str,
    model_key: str,
    api_key: str,
    api_url: str = DEFAULT_API_URL,
    max_retries: int = 3,
    timeout: int = 120,
) -> Dict[str, Any]:
    """Refactor a Python function using an LLM via the OpenRouter API.

    Args:
        code: Source code of the Python function to refactor.
        model_key: Identifier for the desired model (e.g., "claude_opus").
        api_key: OpenRouter API key.
        api_url: Endpoint URL for the OpenRouter API.
        max_retries: Maximum number of retry attempts on failure.
        timeout: Request timeout in seconds.

    Returns:
        A dictionary containing:
            success (bool): Whether the refactoring succeeded.
            refactored_code (str | None): The cleaned refactored code.
            model_key (str): The original model key.
            model_slug (str): Resolved model slug used for the request.
            error (str | None): Error message if the request failed.
            usage (dict | None): Token usage information from the API.

    Raises:
        None.
    """
    model_slug = OPENROUTER_MODEL_MAP.get(model_key, model_key)
    prompt = REFACTOR_PROMPT_TEMPLATE.format(code=code)

    headers = _build_headers(api_key)
    payload = _build_payload(model_slug, prompt)

    last_error: Optional[str] = None

    for attempt_num in range(1, max_retries + 1):
        try:
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
        except requests.exceptions.Timeout:
            last_error = f"Request timeout (attempt {attempt_num}/{max_retries})"
            if attempt_num < max_retries:
                time.sleep(TIMEOUT_SLEEP_SECONDS)
            continue
        except requests.exceptions.ConnectionError as exc:
            last_error = f"Connection error: {exc}"
            if attempt_num < max_retries:
                time.sleep(CONNECTION_SLEEP_SECONDS)
            continue
        except Exception as exc:
            last_error = f"Unexpected error: {exc}"
            if attempt_num < max_retries:
                time.sleep(UNEXPECTED_SLEEP_SECONDS)
            continue

        if response.status_code == 429:
            wait = min(2 ** attempt_num * BACKOFF_BASE_SECONDS, MAX_BACKOFF_WAIT)
            print(f"    Rate limited. Waiting {wait}s...")
            time.sleep(wait)
            continue

        if response.status_code != 200:
            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
            if attempt_num < max_retries:
                time.sleep(RETRY_SLEEP_SECONDS)
            continue

        data = response.json()
        refactored = _extract_refactored_code(data)

        if not refactored:
            last_error = "No refactored code in response"
            if attempt_num < max_retries:
                time.sleep(RETRY_SLEEP_SECONDS)
            continue

        usage = data.get("usage")
        return _handle_success(refactored, model_key, model_slug, usage)

    return _handle_failure(model_key, model_slug, last_error)


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
