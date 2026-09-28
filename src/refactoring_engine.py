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
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

# Constants
DEFAULT_TEMPERATURE: float = 0.3
DEFAULT_MAX_TOKENS: int = 4000
RATE_LIMIT_STATUS: int = 429
MAX_BACKOFF_SECONDS: int = 60
BACKOFF_BASE_SECONDS: int = 5
BACKOFF_FACTOR: int = 2
RETRY_WAIT_SECONDS: int = 2
TIMEOUT_WAIT_SECONDS: int = 3
CONNECTION_WAIT_SECONDS: int = 5
HTTP_ERROR_TRUNCATE: int = 200
SYSTEM_MESSAGE: str = (
    "You are an expert Python code refactoring assistant. Return ONLY clean Python code "
    "with no explanations or markdown fences."
)


@dataclass
class RefactorRequest:
    """Configuration for a refactoring request."""

    code: str
    model_key: str
    api_key: str
    api_url: str = DEFAULT_API_URL
    max_retries: int = 3
    timeout: int = 120


def _build_headers(api_key: str) -> Dict[str, str]:
    """Create request headers for the OpenRouter API."""
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
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
        ],
        "temperature": DEFAULT_TEMPERATURE,
        "max_tokens": DEFAULT_MAX_TOKENS,
    }


def _handle_rate_limit(attempt: int) -> None:
    """Wait using exponential backoff when rate limited."""
    wait = min((BACKOFF_FACTOR ** attempt) * BACKOFF_BASE_SECONDS, MAX_BACKOFF_SECONDS)
    print(f"    Rate limited. Waiting {wait}s...")
    time.sleep(wait)


def _extract_refactored_code(data: Dict[str, Any]) -> Optional[str]:
    """Extract and clean the refactored code from the API response."""
    choices = data.get("choices", [])
    if not choices:
        return None
    raw = choices[0].get("message", {}).get("content", "").strip()
    return _clean_code_output(raw)


def _make_request(
    api_url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout: int,
) -> requests.Response:
    """Send a POST request to the OpenRouter API."""
    return requests.post(api_url, headers=headers, json=payload, timeout=timeout)


def refactor_function(request: RefactorRequest) -> Dict[str, Any]:
    """Refactor Python code using a specified LLM via the OpenRouter API.

    Args:
        request: RefactorRequest object containing code, model information,
            authentication, and optional settings.

    Returns:
        A dictionary with keys:
            - success (bool): Whether the refactoring succeeded.
            - refactored_code (str | None): The cleaned refactored code.
            - model_key (str): Original model identifier.
            - model_slug (str): Resolved model slug used for the request.
            - error (str | None): Error message if unsuccessful.
            - usage (dict | None): Token usage information from the API.

    Raises:
        None. All errors are captured and returned in the result dictionary.
    """
    model_slug = OPENROUTER_MODEL_MAP.get(request.model_key, request.model_key)
    prompt = REFACTOR_PROMPT_TEMPLATE.format(code=request.code)

    headers = _build_headers(request.api_key)
    payload = _build_payload(model_slug, prompt)

    last_error: Optional[str] = None

    for attempt in range(1, request.max_retries + 1):
        try:
            response = _make_request(request.api_url, headers, payload, request.timeout)

            if response.status_code == RATE_LIMIT_STATUS:
                _handle_rate_limit(attempt)
                continue

            if response.status_code != 200:
                last_error = f"HTTP {response.status_code}: {response.text[:HTTP_ERROR_TRUNCATE]}"
                if attempt < request.max_retries:
                    time.sleep(RETRY_WAIT_SECONDS)
                continue

            data = response.json()
            refactored = _extract_refactored_code(data)

            if refactored is None:
                last_error = "No choices in response"
                if attempt < request.max_retries:
                    time.sleep(RETRY_WAIT_SECONDS)
                continue

            usage = data.get("usage")
            return {
                "success": True,
                "refactored_code": refactored,
                "model_key": request.model_key,
                "model_slug": model_slug,
                "error": None,
                "usage": usage,
            }

        except requests.exceptions.Timeout:
            last_error = f"Request timeout (attempt {attempt}/{request.max_retries})"
            if attempt < request.max_retries:
                time.sleep(TIMEOUT_WAIT_SECONDS)

        except requests.exceptions.ConnectionError as exc:
            last_error = f"Connection error: {exc}"
            if attempt < request.max_retries:
                time.sleep(CONNECTION_WAIT_SECONDS)

        except Exception as exc:
            last_error = f"Unexpected error: {exc}"
            if attempt < request.max_retries:
                time.sleep(RETRY_WAIT_SECONDS)

    return {
        "success": False,
        "refactored_code": None,
        "model_key": request.model_key,
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
