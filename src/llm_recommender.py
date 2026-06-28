"""
LLM Recommender
===============
Predicts the best LLM for refactoring a given Python function using trained
ML models (quality regression + metrics regression).

Reuses the feature engineering and model loading logic from
codeworks4/refactor_cli.py. The quality model predicts a continuous 0-10
refactoring quality score, and the metrics model predicts generation time
and token usage for each of the 5 supported LLMs.

Composite score (0-100): 50% quality + 20% cost + 20% time + 10% tokens.
"""

import re
import os
import sys
import csv
import warnings
import numpy as np
import joblib
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Constants ────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent.parent
MODELS_DIR   = SCRIPT_DIR / "models"

MODEL_ORDER = ["claude_opus", "claude_sonnet_4_6", "gemini_3_1_pro", "gemini_flash", "gpt_oss"]
MODEL_DISPLAY = {
    "claude_opus":        "Claude Opus",
    "claude_sonnet_4_6":  "Claude Sonnet 4.6",
    "gemini_3_1_pro":     "Gemini 3.1 Pro",
    "gemini_flash":       "Gemini Flash",
    "gpt_oss":            "GPT-OSS 120B",
}

# OpenRouter model slugs
OPENROUTER_MODEL_MAP = {
    "claude_opus":       "anthropic/claude-opus-4.5",
    "claude_sonnet_4_6": "anthropic/claude-sonnet-4.5",
    "gemini_3_1_pro":    "google/gemini-3.1-pro-preview",
    "gemini_flash":      "google/gemini-3.5-flash",
    "gpt_oss":           "openai/gpt-oss-120b",
}

# ── Smell names (must match refactoring_quality_ml.ipynb) ───────────────────
SMELL_NAMES = [
    "long_method", "long_param_list", "deep_nesting", "magic_numbers",
    "no_docstring", "no_type_hints", "long_lines", "commented_code", "poor_naming",
]

# ── Helper functions ─────────────────────────────────────────────────────────


def count_lines(s: str) -> int:
    if not isinstance(s, str):
        return 0
    return sum(1 for l in s.replace("\\n", "\n").split("\n") if l.strip())


def count_keywords(s: str) -> int:
    if not isinstance(s, str):
        return 0
    return sum(s.count(k) for k in ("if ", "for ", "while ", "try:", "except", "with "))


def has_docstring(s: str) -> int:
    return int(isinstance(s, str) and ('"""' in s or "'''" in s))


def has_type_hints(s: str) -> int:
    return int(isinstance(s, str) and bool(
        re.search(r"->\s*(str|int|float|bool|None|list|dict|tuple|Any|Optional|List|Dict|Tuple)", s)
    ))


def has_future_import(s: str) -> int:
    return int(isinstance(s, str) and "from __future__ import annotations" in s)


def char_token_est(s: str) -> int:
    if not isinstance(s, str):
        return 0
    return max(0, len(s.replace("\\n", "\n")) // 4)


# ── Code smell detectors ─────────────────────────────────────────────────────


import re
from typing import Dict

MAX_METHOD_LENGTH = 20
MAX_PARAMS = 4
MAX_INDENT = 16
MAX_LINE_LENGTH = 79
ALLOWED_NUMBERS = {"0", "1", "2", "3", "10", "100", "-1", "0.0", "1.0"}
ALLOWED_VARIABLES = {
    "i",
    "j",
    "k",
    "n",
    "x",
    "y",
    "z",
    "f",
    "e",
    "v",
    "_",
    "s",
    "c",
    "p",
    "q",
}


def _detect_long_method(nonempty_lines: list[str]) -> int:
    """Return 1 if the method exceeds the maximum allowed length."""
    return int(len(nonempty_lines) > MAX_METHOD_LENGTH)


def _detect_long_param_list(code: str) -> int:
    """Return 1 if a function definition has more than the allowed parameters."""
    signature_match = re.search(r"def\s+\w+\s*\(([^)]*)\)", code)
    if not signature_match:
        return 0
    params = [
        p.strip()
        for p in signature_match.group(1).split(",")
        if p.strip() and p.strip() not in ("self", "cls")
    ]
    return int(len(params) > MAX_PARAMS)


def _detect_deep_nesting(nonempty_lines: list[str]) -> int:
    """Return 1 if any line is indented deeper than the allowed threshold."""
    if not nonempty_lines:
        return 0
    max_indent = max(len(line) - len(line.lstrip()) for line in nonempty_lines)
    return int(max_indent >= MAX_INDENT)


def _detect_magic_numbers(code: str) -> int:
    """Return 1 if more than two magic numbers are present."""
    numbers = re.findall(r"(?<!\w)(\d+\.?\d*)(?!\w)", code)
    magic_numbers = [n for n in numbers if n not in ALLOWED_NUMBERS]
    return int(len(magic_numbers) > 2)


def _detect_no_docstring(code: str) -> int:
    """Return 1 if the code lacks a triple‑quoted docstring."""
    return int('"""' not in code and "'''" not in code)


def _detect_no_type_hints(code: str) -> int:
    """Return 1 if no type hints are found in function signatures or return annotations."""
    type_hint_pattern = r"->\s*\w|:\s*(str|int|float|bool|None|list|dict|tuple|Any|Optional|List|Dict|Tuple)"
    return int(not bool(re.search(type_hint_pattern, code)))


def _detect_long_lines(lines: list[str]) -> int:
    """Return 1 if any line exceeds the maximum line length."""
    return int(any(len(line) > MAX_LINE_LENGTH for line in lines))


def _detect_commented_code(lines: list[str]) -> int:
    """Return 1 if there are lines that appear to be commented‑out code."""
    commented = [
        line
        for line in lines
        if re.match(
            r"\s*#\s*(if |for |while |return |self\.|import |def |class |print\()", line
        )
    ]
    return int(len(commented) >= 1)


def _detect_poor_naming(code: str) -> int:
    """Return 1 if more than one single‑letter variable name is used outside the allowed set."""
    poor_vars = [
        var
        for var in re.findall(r"\b([a-zA-Z])\s*(?:\+|-|\*|\/)?=(?!=)", code)
        if var.lower() not in ALLOWED_VARIABLES
    ]
    return int(len(poor_vars) > 1)


def detect_code_smells(code_str: str) -> Dict[str, int]:
    """Detect nine common code smells in a Python source string.

    Args:
        code_str: The source code to analyze.

    Returns:
        A dictionary mapping each smell name to 0 (absent) or 1 (present).

    Raises:
        TypeError: If ``code_str`` is not a string.
    """
    if not isinstance(code_str, str):
        raise TypeError("code_str must be a string")

    if len(code_str.strip()) < 5:
        return {smell: 0 for smell in SMELL_NAMES}

    normalized_code = code_str.replace("\\n", "\n")
    lines = normalized_code.split("\n")
    nonempty_lines = [line for line in lines if line.strip()]

    smells: Dict[str, int] = {
        "long_method": _detect_long_method(nonempty_lines),
        "long_param_list": _detect_long_param_list(normalized_code),
        "deep_nesting": _detect_deep_nesting(nonempty_lines),
        "magic_numbers": _detect_magic_numbers(normalized_code),
        "no_docstring": _detect_no_docstring(normalized_code),
        "no_type_hints": _detect_no_type_hints(normalized_code),
        "long_lines": _detect_long_lines(lines),
        "commented_code": _detect_commented_code(lines),
        "poor_naming": _detect_poor_naming(normalized_code),
    }

    return smells


def count_smells(code_str: str) -> int:
    return sum(detect_code_smells(code_str).values())


# ── Model loading ────────────────────────────────────────────────────────────

_models_loaded = False
_q_regressors = {}
_q_scaler = None
_q_le = None
_q_stats = None
_q_meta = None
_m_time = {}
_m_tokens = {}
_m_scaler = None
_m_le = None
_m_stats = None
_m_meta = None
_prices = {}


def _safe_joblib_load(path):
    """Try to load a joblib pickle. Return None on failure (version mismatch)."""
    try:
        return joblib.load(str(path))
    except Exception as e:
        print(f"  [llm_recommender] ⚠ Could not load {path.name}: {e}", file=sys.stderr)
        print(f"  [llm_recommender]   Using fallback defaults.", file=sys.stderr)
        return None


from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

# Constants
QUALITY_MODEL_NAMES: List[str] = [
    "ridge_regression",
    "decision_tree",
    "random_forest",
    "gradient_boosting",
    "xgboost",
    "svr",
]
METRICS_MODEL_NAMES: List[str] = ["ridge", "random_forest", "gradient_boosting", "xgboost"]
METRICS_PREFIXES: List[tuple[Dict[str, Any], str]] = [
    (_m_time, "metrics_time"),
    (_m_tokens, "metrics_tokens"),
]
PRICE_CSV_HEADERS: List[str] = ["Model Family", "Model", "model", "Input", "input"]
NUMERIC_PATTERN = re.compile(r"[0-9]+\.?[0-9]*(?:e[+-]?\d+)?", re.IGNORECASE)


def _load_quality_regressors() -> None:
    """Load quality regressors and related artifacts."""
    for name in QUALITY_MODEL_NAMES:
        model_path = MODELS_DIR / f"quality_{name}.pkl"
        if model_path.exists():
            loaded = _safe_joblib_load(model_path)
            if loaded is not None:
                _q_regressors[name] = loaded

    _q_scaler = _safe_joblib_load(MODELS_DIR / "quality_scaler.pkl")
    _q_le = _safe_joblib_load(MODELS_DIR / "quality_label_encoder.pkl")
    _q_stats = _safe_joblib_load(MODELS_DIR / "quality_stats.pkl")
    _q_meta = _safe_joblib_load(MODELS_DIR / "quality_meta.pkl")

    globals().update(
        {
            "_q_scaler": _q_scaler,
            "_q_le": _q_le,
            "_q_stats": _q_stats,
            "_q_meta": _q_meta,
        }
    )


def _load_metrics_regressors() -> None:
    """Load metrics regressors and related artifacts."""
    for name in METRICS_MODEL_NAMES:
        for container, prefix in METRICS_PREFIXES:
            model_path = MODELS_DIR / f"{prefix}_{name}.pkl"
            if model_path.exists():
                loaded = _safe_joblib_load(model_path)
                if loaded is not None:
                    container[name] = loaded

    _m_scaler = _safe_joblib_load(MODELS_DIR / "metrics_scaler.pkl")
    _m_le = _safe_joblib_load(MODELS_DIR / "metrics_label_encoder.pkl")
    _m_stats = _safe_joblib_load(MODELS_DIR / "metrics_stats.pkl")
    _m_meta = _safe_joblib_load(MODELS_DIR / "metrics_meta.pkl")

    globals().update(
        {
            "_m_scaler": _m_scaler,
            "_m_le": _m_le,
            "_m_stats": _m_stats,
            "_m_meta": _m_meta,
        }
    )


def _extract_price(row: Dict[str, str]) -> Optional[float]:
    """Extract average numeric price from a CSV row.

    Args:
        row: Mapping of column names to string values.

    Returns:
        The average of numeric values found in the input field, or None if
        extraction fails.
    """
    key = row.get("Model Family") or row.get("Model") or row.get("model")
    inp = row.get("Input") or row.get("input")
    if not key or not inp:
        return None

    numbers = NUMERIC_PATTERN.findall(str(inp).strip())
    try:
        values = [float(num) for num in numbers]
        return sum(values) / len(values) if values else None
    except Exception:
        return None


def _load_prices() -> None:
    """Load pricing information from the CSV file into the global _prices dict."""
    prices_path = SCRIPT_DIR / "price.csv"
    if not prices_path.exists():
        return

    with open(str(prices_path), newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            price = _extract_price(row)
            if price is None:
                continue
            key = (row.get("Model Family") or row.get("Model") or row.get("model")).strip()
            _prices[key] = price


def _load_models() -> None:
    """Load all model artifacts and pricing data if not already loaded.

    This function populates several global dictionaries and flags:
    - Quality regressors and associated scaler, label encoder, stats, and metadata.
    - Metrics regressors for time and token predictions, plus their scaler,
      label encoder, stats, and metadata.
    - Pricing information from a CSV file.

    The function is idempotent; subsequent calls have no effect after the first
    successful load.

    Raises:
        Any exception raised by underlying I/O or deserialization functions is
        propagated to the caller.
    """
    global _models_loaded

    if _models_loaded:
        return

    _load_quality_regressors()
    _load_metrics_regressors()
    _load_prices()

    _models_loaded = True


# ── Feature engineering ──────────────────────────────────────────────────────


def quality_features(orig_code: str, refact_code: str, func_name: str,
                     model_enc: int) -> np.ndarray:
    """Build 50 features (19 structural + 31 smell) for quality prediction."""
    q_stats = _q_stats

    orig_line_count = float(count_lines(orig_code))
    long_method_flag = int(orig_line_count > 15)
    refact_line_count = float(count_lines(refact_code))
    line_delta = orig_line_count - refact_line_count
    line_delta_ratio = line_delta / (orig_line_count + 1)
    hfi = has_future_import(refact_code)
    hds = has_docstring(refact_code)
    hth = has_type_hints(refact_code)
    complexity_proxy = count_keywords(refact_code)
    orig_complexity = count_keywords(orig_code)
    complexity_delta = orig_complexity - complexity_proxy
    func_name_len = len(str(func_name))
    is_private = int(str(func_name).startswith("_") and not str(func_name).startswith("__"))
    is_dunder = int(str(func_name).startswith("__") and str(func_name).endswith("__"))
    is_test_func = int(str(func_name).startswith("test"))
    repo_avg_lines = float(q_stats.get("global_repo_mean", orig_line_count))
    file_avg_lines = float(q_stats.get("global_file_mean", orig_line_count))
    global_max = float(q_stats.get("global_max_lines", max(orig_line_count, 1)))
    norm_line_count = orig_line_count / (global_max + 1)

    structural = [
        orig_line_count, long_method_flag, refact_line_count, line_delta,
        line_delta_ratio, hfi, hds, hth,
        complexity_proxy, orig_complexity, complexity_delta,
        func_name_len, is_private, is_dunder, is_test_func,
        repo_avg_lines, file_avg_lines, norm_line_count,
        model_enc,
    ]

    # Smell features (31)
    orig_smells = detect_code_smells(orig_code)
    refact_smells = detect_code_smells(refact_code)

    orig_smell_count = sum(orig_smells.values())
    refact_smell_count = sum(refact_smells.values())
    smell_reduction = max(0, orig_smell_count - refact_smell_count)
    smell_reduction_ratio = smell_reduction / (orig_smell_count + 1)

    smell_agg = [orig_smell_count, refact_smell_count, smell_reduction, smell_reduction_ratio]
    orig_per = [orig_smells[s] for s in SMELL_NAMES]
    refact_per = [refact_smells[s] for s in SMELL_NAMES]
    fixed_per = [max(0, orig_smells[s] - refact_smells[s]) for s in SMELL_NAMES]

    smell_features = smell_agg + orig_per + refact_per + fixed_per

    return np.array([structural + smell_features], dtype=float)


def metrics_features(orig_code: str, refact_code: str, func_name: str,
                     model_enc: int) -> np.ndarray:
    """Build 23 features for metrics (time/tokens) prediction."""
    m_stats = _m_stats

    orig_lines = float(count_lines(orig_code))
    long_method = int(orig_lines > 15)
    orig_complexity = count_keywords(orig_code)
    orig_char_tokens = char_token_est(orig_code)
    refact_lines = float(count_lines(refact_code))
    refact_complexity = count_keywords(refact_code)
    refact_char_tokens = char_token_est(refact_code)
    hds = has_docstring(refact_code)
    hth = has_type_hints(refact_code)
    hfi = has_future_import(refact_code)
    line_delta = orig_lines - refact_lines
    line_delta_ratio = line_delta / (orig_lines + 1)
    complexity_delta = orig_complexity - refact_complexity
    token_delta = orig_char_tokens - refact_char_tokens
    est_input_tokens = orig_char_tokens + 250
    est_output_tokens = refact_char_tokens
    func_name_len = len(str(func_name))
    is_private = int(str(func_name).startswith("_") and not str(func_name).startswith("__"))
    is_dunder = int(str(func_name).startswith("__") and str(func_name).endswith("__"))
    is_test_func = int(str(func_name).startswith("test"))
    repo_avg_orig_lines = float(m_stats.get("global_repo_mean", orig_lines))
    file_avg_orig_lines = float(m_stats.get("global_file_mean", orig_lines))

    return np.array([[
        orig_lines, long_method, orig_complexity, orig_char_tokens,
        refact_lines, refact_complexity, refact_char_tokens,
        hds, hth, hfi,
        line_delta, line_delta_ratio, complexity_delta, token_delta,
        est_input_tokens, est_output_tokens,
        func_name_len, is_private, is_dunder, is_test_func,
        repo_avg_orig_lines, file_avg_orig_lines,
        model_enc,
    ]], dtype=float)


# ── Prediction ───────────────────────────────────────────────────────────────


DEFAULT_QUALITY = 5.0
DEFAULT_TIME_MS = 2000
DEFAULT_TOKENS = 500


def _encode_label(
    encoder: Any, llm: str, fallback_index: int
) -> int:
    """Encode LLM name using a label encoder or fallback index."""
    try:
        return int(encoder.transform([llm])[0]) if encoder is not None else fallback_index
    except Exception:
        return fallback_index


def _predict_quality(
    orig_code: str,
    refact_proxy: str,
    func_name: str,
    q_enc: int,
    best_q_name: str,
    best_q_safe: str,
    models_available: bool,
    q_scaled_models: List[str],
) -> float:
    """Return quality score in the range [0, 10]."""
    if not (models_available and best_q_safe in _q_regressors):
        return DEFAULT_QUALITY

    q_features = quality_features(orig_code, refact_proxy, func_name, q_enc)
    reg = _q_regressors[best_q_safe]

    try:
        if _q_scaler is not None and best_q_name in q_scaled_models:
            q_input = _q_scaler.transform(q_features)
        else:
            q_input = q_features
        pred = reg.predict(q_input)[0]
        return float(np.clip(pred, 0.0, 10.0))
    except Exception:
        return DEFAULT_QUALITY


def _predict_metrics(
    orig_code: str,
    refact_proxy: str,
    func_name: str,
    m_enc: int,
    best_time_name: str,
    best_tokens_name: str,
    metrics_available: bool,
) -> Tuple[int, int]:
    """Return predicted time (ms) and token count."""
    if not metrics_available:
        return DEFAULT_TIME_MS, DEFAULT_TOKENS

    m_features = metrics_features(orig_code, refact_proxy, func_name, m_enc)
    time_reg = _m_time.get(best_time_name)
    tokens_reg = _m_tokens.get(best_tokens_name)

    if time_reg is None or tokens_reg is None:
        return DEFAULT_TIME_MS, DEFAULT_TOKENS

    try:
        time_ms = int(np.expm1(time_reg.predict(m_features)[0]))
        tokens = int(np.expm1(tokens_reg.predict(m_features)[0]))
        return time_ms, tokens
    except Exception:
        return DEFAULT_TIME_MS, DEFAULT_TOKENS


def _lookup_price(llm: str) -> float:
    """Find price per token for a given LLM, handling fuzzy matches."""
    try:
        price = float(_prices.get(llm, 0.0))
        if price != 0.0:
            return price
        for k, v in _prices.items():
            kn = k.lower().replace(" ", "_")
            if kn == llm or llm in kn or kn in llm:
                return float(v)
    except Exception:
        pass
    return 0.0


def _normalize_array(arr: np.ndarray) -> np.ndarray:
    """Scale array to [0, 1]; return 0.5 for constant arrays."""
    rng = arr.max() - arr.min()
    if rng > 0:
        return (arr - arr.min()) / rng
    return np.full_like(arr, 0.5, dtype=float)


def _compute_composite(
    scores: np.ndarray,
    times: np.ndarray,
    tokens: np.ndarray,
    costs: np.ndarray,
) -> np.ndarray:
    """Calculate composite score (0‑100) using weighted normalized metrics."""
    norm_scores = _normalize_array(scores)
    norm_costs = _normalize_array(costs)
    norm_times = _normalize_array(times)
    norm_tokens = _normalize_array(tokens)

    composite = (
        0.50 * norm_scores
        + 0.20 * (1.0 - norm_costs)
        + 0.20 * (1.0 - norm_times)
        + 0.10 * (1.0 - norm_tokens)
    ) * 100.0
    return composite


def predict_all_llms(
    orig_code: str,
    func_name: str = "unknown_function",
) -> Dict[str, Dict[str, Any]]:
    """Predict quality, latency, token usage and cost for each LLM.

    Args:
        orig_code: Original source code to evaluate.
        func_name: Name of the function being refactored (default
            ``"unknown_function"``).

    Returns:
        Mapping from LLM name to a dictionary containing:
        ``quality_score``, ``pred_time_ms``, ``pred_tokens``,
        ``pred_cost``, ``price_per_token``, ``openrouter_slug``,
        ``display_name`` and ``composite`` score.

    Raises:
        None.
    """
    _load_models()
    refact_proxy = orig_code

    models_available = bool(_q_regressors) and _q_scaler is not None and _q_le is not None
    metrics_available = bool(_m_time) and _m_scaler is not None and _m_le is not None

    if models_available and _q_meta is not None:
        best_q_name = _q_meta["best_model"]
        best_q_safe = best_q_name.lower().replace(" ", "_")
        q_scaled_models = (
            _q_stats.get("scaled_models", ["Ridge Regression", "SVR"])
            if _q_stats is not None
            else []
        )
    else:
        best_q_name = ""
        best_q_safe = ""
        q_scaled_models = []

    if metrics_available and _m_meta is not None:
        best_time_name = _m_meta.get("best_time_model", "gradient_boosting").lower().replace(" ", "_")
        best_tokens_name = _m_meta.get("best_tokens_model", "gradient_boosting").lower().replace(" ", "_")
    else:
        best_time_name = ""
        best_tokens_name = ""

    results: Dict[str, Dict[str, Any]] = {}
    for llm in MODEL_ORDER:
        q_enc = _encode_label(_q_le, llm, MODEL_ORDER.index(llm))
        m_enc = _encode_label(_m_le, llm, MODEL_ORDER.index(llm))

        quality = _predict_quality(
            orig_code,
            refact_proxy,
            func_name,
            q_enc,
            best_q_name,
            best_q_safe,
            models_available,
            q_scaled_models,
        )

        pred_time, pred_tokens = _predict_metrics(
            orig_code,
            refact_proxy,
            func_name,
            m_enc,
            best_time_name,
            best_tokens_name,
            metrics_available,
        )

        price_per_token = _lookup_price(llm)
        pred_cost = float(pred_tokens) * price_per_token

        results[llm] = {
            "quality_score": quality,
            "pred_time_ms": pred_time,
            "pred_tokens": pred_tokens,
            "pred_cost": pred_cost,
            "price_per_token": price_per_token,
            "openrouter_slug": OPENROUTER_MODEL_MAP.get(llm, llm),
            "display_name": MODEL_DISPLAY.get(llm, llm),
        }

    llm_names = list(results.keys())
    scores = np.array([results[l]["quality_score"] for l in llm_names], dtype=float)
    times = np.array([results[l]["pred_time_ms"] for l in llm_names], dtype=float)
    tokens = np.array([results[l]["pred_tokens"] for l in llm_names], dtype=float)
    costs = np.array([results[l].get("pred_cost", 0.0) for l in llm_names], dtype=float)

    composite = _compute_composite(scores, times, tokens, costs)

    for idx, llm in enumerate(llm_names):
        results[llm]["composite"] = float(composite[idx])

    return results


def get_best_llm(orig_code: str, func_name: str = "unknown_function") -> dict:
    """
    Returns the single best LLM recommendation with full details.

    Returns:
        {
            "model_key": "claude_opus",
            "display_name": "Claude Opus",
            "openrouter_slug": "anthropic/claude-3-opus",
            "quality_score": 8.5,
            "pred_time_ms": 15000,
            "pred_tokens": 1200,
            "pred_cost": 0.0005,
            "composite": 92.3,
            "all_models": { ... }   # full ranking for reference
        }
    """
    results = predict_all_llms(orig_code, func_name)
    best_key = max(results, key=lambda k: results[k]["composite"])
    return {
        "model_key": best_key,
        **results[best_key],
        "all_models": results,
    }


def get_ranking(orig_code: str, func_name: str = "unknown_function") -> list:
    """
    Return a ranked list of all LLMs by composite score (descending).
    Each entry contains all prediction details.
    """
    results = predict_all_llms(orig_code, func_name)
    ranked = sorted(results.items(), key=lambda x: x[1]["composite"], reverse=True)
    return [{"model_key": k, **v} for k, v in ranked]
