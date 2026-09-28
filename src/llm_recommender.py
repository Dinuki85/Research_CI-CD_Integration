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
from typing import Dict, List

MAX_METHOD_LINES = 20
MAX_PARAMS = 4
INDENT_THRESHOLD = 16
MIN_CODE_LENGTH = 5
LONG_LINE_LENGTH = 79
ALLOWED_MAGIC_NUMBERS = {
    "0",
    "1",
    "2",
    "3",
    "10",
    "100",
    "-1",
    "0.0",
    "1.0",
}
ALLOWED_VARIABLE_NAMES = {
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


def _is_long_method(nonempty_lines: List[str]) -> int:
    """Return 1 if the method exceeds MAX_METHOD_LINES, else 0."""
    return int(len(nonempty_lines) > MAX_METHOD_LINES)


def _has_long_param_list(code: str) -> int:
    """Return 1 if a function definition has more than MAX_PARAMS parameters."""
    match = re.search(r"def\s+\w+\s*\(([^)]*)\)", code)
    if not match:
        return 0
    params = [
        p.strip()
        for p in match.group(1).split(",")
        if p.strip() and p.strip() not in ("self", "cls")
    ]
    return int(len(params) > MAX_PARAMS)


def _has_deep_nesting(nonempty_lines: List[str]) -> int:
    """Return 1 if any line is indented by INDENT_THRESHOLD spaces or more."""
    if not nonempty_lines:
        return 0
    max_indent = max(len(line) - len(line.lstrip()) for line in nonempty_lines)
    return int(max_indent >= INDENT_THRESHOLD)


def _has_magic_numbers(code: str) -> int:
    """Return 1 if more than two numeric literals are not in ALLOWED_MAGIC_NUMBERS."""
    numbers = re.findall(r"(?<!\w)(\d+\.?\d*)(?!\w)", code)
    magic = [n for n in numbers if n not in ALLOWED_MAGIC_NUMBERS]
    return int(len(magic) > 2)


def _has_no_docstring(code: str) -> int:
    """Return 1 if the source contains neither triple‑double nor triple‑single quotes."""
    return int('"""' not in code and "'''" not in code)


def _has_no_type_hints(code: str) -> int:
    """Return 1 if the code lacks return annotations or parameter type hints."""
    pattern = r"->\s*\w|:\s*(str|int|float|bool|None|list|dict|tuple|Any|Optional|List|Dict|Tuple)"
    return int(not bool(re.search(pattern, code)))


def _has_long_lines(lines: List[str]) -> int:
    """Return 1 if any line exceeds LONG_LINE_LENGTH characters."""
    return int(any(len(line) > LONG_LINE_LENGTH for line in lines))


def _has_commented_code(lines: List[str]) -> int:
    """Return 1 if there is at least one line that looks like commented‑out code."""
    commented = [
        line
        for line in lines
        if re.match(
            r"\s*#\s*(if |for |while |return |self\.|import |def |class |print\()", line
        )
    ]
    return int(len(commented) >= 1)


def _has_poor_naming(code: str) -> int:
    """Return 1 if more than one single‑letter variable (not allowed) is assigned."""
    poor = [
        var
        for var in re.findall(r"\b([a-zA-Z])\s*(?:\+|-|\*|\/)?=(?!=)", code)
        if var.lower() not in ALLOWED_VARIABLE_NAMES
    ]
    return int(len(poor) > 1)


def detect_code_smells(code_str: str) -> Dict[str, int]:
    """Detect nine common code smells in a Python source string.

    Args:
        code_str: The source code to analyze.

    Returns:
        A dictionary mapping each smell name to 0 (absent) or 1 (present).

    Raises:
        None.
    """
    empty = {s: 0 for s in SMELL_NAMES}
    if not isinstance(code_str, str) or len(code_str.strip()) < MIN_CODE_LENGTH:
        return empty

    code = code_str.replace("\\n", "\n")
    lines = code.split("\n")
    nonempty = [line for line in lines if line.strip()]

    smells: Dict[str, int] = {
        "long_method": _is_long_method(nonempty),
        "long_param_list": _has_long_param_list(code),
        "deep_nesting": _has_deep_nesting(nonempty),
        "magic_numbers": _has_magic_numbers(code),
        "no_docstring": _has_no_docstring(code),
        "no_type_hints": _has_no_type_hints(code),
        "long_lines": _has_long_lines(lines),
        "commented_code": _has_commented_code(lines),
        "poor_naming": _has_poor_naming(code),
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


QUALITY_REGRESSOR_NAMES = [
    "ridge_regression",
    "decision_tree",
    "random_forest",
    "gradient_boosting",
    "xgboost",
    "svr",
]

METRICS_REGRESSOR_NAMES = ["ridge", "random_forest", "gradient_boosting", "xgboost"]
METRICS_PREFIXES = [("time", _m_time), ("tokens", _m_tokens)]

def _load_quality_models() -> None:
    """Load quality‑related models and auxiliary objects.

    Populates the global dictionaries ``_q_regressors``, ``_q_scaler``,
    ``_q_le``, ``_q_stats`` and ``_q_meta`` with objects loaded from
    ``MODELS_DIR``. Missing files are ignored (best‑effort loading).
    """
    global _q_regressors, _q_scaler, _q_le, _q_stats, _q_meta

    for name in QUALITY_REGRESSOR_NAMES:
        model_path = MODELS_DIR / f"quality_{name}.pkl"
        if model_path.exists():
            loaded = _safe_joblib_load(model_path)
            if loaded is not None:
                _q_regressors[name] = loaded

    _q_scaler = _safe_joblib_load(MODELS_DIR / "quality_scaler.pkl")
    _q_le = _safe_joblib_load(MODELS_DIR / "quality_label_encoder.pkl")
    _q_stats = _safe_joblib_load(MODELS_DIR / "quality_stats.pkl")
    _q_meta = _safe_joblib_load(MODELS_DIR / "quality_meta.pkl")


def _load_metrics_models() -> None:
    """Load metrics‑related models and auxiliary objects.

    Populates the global dictionaries ``_m_time`` and ``_m_tokens`` with
    regressors, and sets ``_m_scaler``, ``_m_le``, ``_m_stats`` and
    ``_m_meta``. Missing files are ignored (best‑effort loading).
    """
    global _m_time, _m_tokens, _m_scaler, _m_le, _m_stats, _m_meta

    for name in METRICS_REGRESSOR_NAMES:
        for dict_ref, prefix in [( _m_time, "metrics_time"), ( _m_tokens, "metrics_tokens")]:
            model_path = MODELS_DIR / f"{prefix}_{name}.pkl"
            if model_path.exists():
                loaded = _safe_joblib_load(model_path)
                if loaded is not None:
                    dict_ref[name] = loaded

    _m_scaler = _safe_joblib_load(MODELS_DIR / "metrics_scaler.pkl")
    _m_le = _safe_joblib_load(MODELS_DIR / "metrics_label_encoder.pkl")
    _m_stats = _safe_joblib_load(MODELS_DIR / "metrics_stats.pkl")
    _m_meta = _safe_joblib_load(MODELS_DIR / "metrics_meta.pkl")


def _load_prices() -> None:
    """Load pricing information from ``price.csv`` into ``_prices``."""
    global _prices

    prices_path = SCRIPT_DIR / "price.csv"
    if not prices_path.exists():
        return

    with open(str(prices_path), newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            key = row.get("Model Family") or row.get("Model") or row.get("model")
            inp = row.get("Input") or row.get("input")
            if not key or not inp:
                continue

            numbers = re.findall(r"[0-9]+\.?[0-9]*(?:e[+-]?\d+)?", str(inp).strip())
            try:
                values = [float(num) for num in numbers]
                if values:
                    _prices[key.strip()] = sum(values) / len(values)
            except Exception:
                continue


def _load_models() -> None:
    """Load all machine‑learning models and auxiliary data if not already loaded.

    This function is idempotent; subsequent calls have no effect after the
    first successful load.

    Raises:
        Any exception raised by the underlying loading utilities propagates
        to the caller.
    """
    global _models_loaded

    if _models_loaded:
        return

    _load_quality_models()
    _load_metrics_models()
    _load_prices()

    _models_loaded = True


# ── Feature engineering ──────────────────────────────────────────────────────


import numpy as np

LONG_METHOD_THRESHOLD = 15
ONE = 1.0

def _compute_structural_features(
    original_code: str,
    refactored_code: str,
    function_name: str,
    model_encoding: int,
    q_stats: dict,
) -> list[float]:
    """Calculate structural features for a function.

    Args:
        original_code: Source code before refactoring.
        refactored_code: Source code after refactoring.
        function_name: Name of the function being analysed.
        model_encoding: Integer encoding for the model.
        q_stats: Dictionary with repository statistics.

    Returns:
        List of structural feature values.
    """
    original_line_count = float(count_lines(original_code))
    long_method_flag = int(original_line_count > LONG_METHOD_THRESHOLD)

    refactored_line_count = float(count_lines(refactored_code))
    line_delta = original_line_count - refactored_line_count
    line_delta_ratio = line_delta / (original_line_count + ONE)

    has_future_import_flag = has_future_import(refactored_code)
    has_docstring_flag = has_docstring(refactored_code)
    has_type_hints_flag = has_type_hints(refactored_code)

    refactored_complexity = count_keywords(refactored_code)
    original_complexity = count_keywords(original_code)
    complexity_delta = original_complexity - refactored_complexity

    function_name_length = len(str(function_name))
    is_private = int(str(function_name).startswith("_") and not str(function_name).startswith("__"))
    is_dunder = int(str(function_name).startswith("__") and str(function_name).endswith("__"))
    is_test = int(str(function_name).startswith("test"))

    repo_average = float(q_stats.get("global_repo_mean", original_line_count))
    file_average = float(q_stats.get("global_file_mean", original_line_count))
    global_max = float(q_stats.get("global_max_lines", max(original_line_count, ONE)))
    normalized_line_count = original_line_count / (global_max + ONE)

    return [
        original_line_count,
        long_method_flag,
        refactored_line_count,
        line_delta,
        line_delta_ratio,
        has_future_import_flag,
        has_docstring_flag,
        has_type_hints_flag,
        refactored_complexity,
        original_complexity,
        complexity_delta,
        function_name_length,
        is_private,
        is_dunder,
        is_test,
        repo_average,
        file_average,
        normalized_line_count,
        model_encoding,
    ]

def _compute_smell_features(
    original_code: str,
    refactored_code: str,
) -> list[float]:
    """Calculate smell‑related features for a function.

    Args:
        original_code: Source code before refactoring.
        refactored_code: Source code after refactoring.

    Returns:
        List of smell feature values.
    """
    original_smells = detect_code_smells(original_code)
    refactored_smells = detect_code_smells(refactored_code)

    original_smell_total = sum(original_smells.values())
    refactored_smell_total = sum(refactored_smells.values())
    smell_reduction = max(0, original_smell_total - refactored_smell_total)
    smell_reduction_ratio = smell_reduction / (original_smell_total + ONE)

    aggregate_features = [
        original_smell_total,
        refactored_smell_total,
        smell_reduction,
        smell_reduction_ratio,
    ]

    original_per_smell = [original_smells[name] for name in SMELL_NAMES]
    refactored_per_smell = [refactored_smells[name] for name in SMELL_NAMES]
    fixed_per_smell = [
        max(0, original_smells[name] - refactored_smells[name]) for name in SMELL_NAMES
    ]

    return aggregate_features + original_per_smell + refactored_per_smell + fixed_per_smell

def quality_features(
    orig_code: str,
    refact_code: str,
    func_name: str,
    model_enc: int,
) -> np.ndarray:
    """Build a 50‑dimensional feature vector for quality prediction.

    Args:
        orig_code: Original function source code.
        refact_code: Refactored function source code.
        func_name: Name of the function.
        model_enc: Integer encoding for the model.

    Returns:
        NumPy array of shape (1, 50) containing the feature vector.
    """
    structural = _compute_structural_features(
        original_code=orig_code,
        refactored_code=refact_code,
        function_name=func_name,
        model_encoding=model_enc,
        q_stats=_q_stats,
    )
    smell = _compute_smell_features(orig_code, refact_code)
    return np.array([structural + smell], dtype=float)


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


import numpy as np
from typing import Any, Dict, List, Tuple

DEFAULT_QUALITY_SCORE = 5.0
DEFAULT_PRED_TIME_MS = 2000
DEFAULT_PRED_TOKENS = 500
DEFAULT_PRICE_PER_TOKEN = 0.0


def _load_and_check_models() -> Tuple[bool, bool]:
    """Load models and return flags indicating availability of quality and metric models."""
    _load_models()
    models_available = bool(_q_regressors) and _q_scaler is not None and _q_le is not None
    metrics_available = bool(_m_time) and _m_scaler is not None and _m_le is not None
    return models_available, metrics_available


def _get_best_quality_model() -> Tuple[str, str, List[str]]:
    """Return best quality model name, its safe identifier, and scaled model list."""
    if _q_meta is not None:
        best_name = _q_meta["best_model"]
        safe_name = best_name.lower().replace(" ", "_")
        scaled_models = (
            _q_stats.get("scaled_models", ["Ridge Regression", "SVR"])
            if _q_stats is not None
            else []
        )
        return best_name, safe_name, scaled_models
    return "", "", []


def _get_best_metric_models() -> Tuple[str, str]:
    """Return safe identifiers for the best time and token models."""
    if _m_meta is not None:
        time_name = _m_meta.get("best_time_model", "gradient_boosting").lower().replace(" ", "_")
        tokens_name = _m_meta.get("best_tokens_model", "gradient_boosting").lower().replace(" ", "_")
        return time_name, tokens_name
    return "", ""


def _encode_llm(llm: str, encoder) -> int:
    """Encode an LLM name using the provided label encoder, falling back to index."""
    try:
        return int(encoder.transform([llm])[0]) if encoder is not None else MODEL_ORDER.index(llm)
    except Exception:
        return MODEL_ORDER.index(llm)


def _predict_quality(
    orig_code: str,
    refact_proxy: str,
    func_name: str,
    q_enc: int,
    best_q_name: str,
    best_q_safe: str,
    q_scaled_models: List[str],
    models_available: bool,
) -> float:
    """Predict the quality score for a single LLM."""
    if not (models_available and best_q_safe in _q_regressors):
        return DEFAULT_QUALITY_SCORE

    q_features = quality_features(orig_code, refact_proxy, func_name, q_enc)
    reg = _q_regressors[best_q_safe]

    try:
        if _q_scaler is not None and best_q_name in q_scaled_models:
            q_input = _q_scaler.transform(q_features)
        else:
            q_input = q_features
        score = float(np.clip(reg.predict(q_input)[0], 0.0, 10.0))
    except Exception:
        score = DEFAULT_QUALITY_SCORE

    return score


def _predict_metrics(
    orig_code: str,
    refact_proxy: str,
    func_name: str,
    m_enc: int,
    best_time_name: str,
    best_tokens_name: str,
    metrics_available: bool,
) -> Tuple[int, int]:
    """Predict generation time (ms) and token count for a single LLM."""
    if not metrics_available:
        return DEFAULT_PRED_TIME_MS, DEFAULT_PRED_TOKENS

    m_features = metrics_features(orig_code, refact_proxy, func_name, m_enc)
    time_reg = _m_time.get(best_time_name)
    tokens_reg = _m_tokens.get(best_tokens_name)

    if time_reg is None or tokens_reg is None:
        return DEFAULT_PRED_TIME_MS, DEFAULT_PRED_TOKENS

    try:
        pred_time = int(np.expm1(time_reg.predict(m_features)[0]))
        pred_tokens = int(np.expm1(tokens_reg.predict(m_features)[0]))
    except Exception:
        pred_time = DEFAULT_PRED_TIME_MS
        pred_tokens = DEFAULT_PRED_TOKENS

    return pred_time, pred_tokens


def _price_per_token(llm: str) -> float:
    """Resolve the price per token for a given LLM."""
    try:
        price = float(_prices.get(llm, DEFAULT_PRICE_PER_TOKEN))
        if price != 0.0:
            return price
        for k, v in _prices.items():
            key_norm = k.lower().replace(" ", "_")
            if key_norm == llm or llm in key_norm or key_norm in llm:
                return float(v)
    except Exception:
        pass
    return DEFAULT_PRICE_PER_TOKEN


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Normalize an array to the range [0, 1]; fallback to 0.5 when constant."""
    diff = arr.max() - arr.min()
    if diff > 0:
        return (arr - arr.min()) / diff
    return np.full_like(arr, 0.5, dtype=float)


def predict_all_llms(orig_code: str, func_name: str = "unknown_function") -> Dict[str, Dict[str, Any]]:
    """
    Predict quality, generation time, token count, and cost for each LLM.

    Args:
        orig_code: Original source code to evaluate.
        func_name: Name of the function being evaluated (default: "unknown_function").

    Returns:
        A dictionary keyed by LLM name, each containing:
            - quality_score (float)
            - pred_time_ms (int)
            - pred_tokens (int)
            - pred_cost (float)
            - price_per_token (float)
            - openrouter_slug (str)
            - display_name (str)
            - composite (float): weighted composite score (0‑100).

    Raises:
        None.
    """
    models_available, metrics_available = _load_and_check_models()
    refact_proxy = orig_code

    best_q_name, best_q_safe, q_scaled_models = _get_best_quality_model()
    best_time_name, best_tokens_name = _get_best_metric_models()

    results: Dict[str, Dict[str, Any]] = {}
    for llm in MODEL_ORDER:
        q_enc = _encode_llm(llm, _q_le)
        m_enc = _encode_llm(llm, _m_le)

        quality_score = _predict_quality(
            orig_code,
            refact_proxy,
            func_name,
            q_enc,
            best_q_name,
            best_q_safe,
            q_scaled_models,
            models_available,
        )

        pred_time_ms, pred_tokens = _predict_metrics(
            orig_code,
            refact_proxy,
            func_name,
            m_enc,
            best_time_name,
            best_tokens_name,
            metrics_available,
        )

        token_price = _price_per_token(llm)
        pred_cost = float(pred_tokens) * token_price

        results[llm] = {
            "quality_score": quality_score,
            "pred_time_ms": pred_time_ms,
            "pred_tokens": pred_tokens,
            "pred_cost": pred_cost,
            "price_per_token": token_price,
            "openrouter_slug": OPENROUTER_MODEL_MAP.get(llm, llm),
            "display_name": MODEL_DISPLAY.get(llm, llm),
        }

    # Composite score calculation
    llm_names = list(results.keys())
    scores = np.array([results[name]["quality_score"] for name in llm_names], dtype=float)
    times = np.array([results[name]["pred_time_ms"] for name in llm_names], dtype=float)
    tokens = np.array([results[name]["pred_tokens"] for name in llm_names], dtype=float)
    costs = np.array([results[name].get("pred_cost", 0.0) for name in llm_names], dtype=float)

    composite = (
        0.50 * _normalize(scores)
        + 0.20 * (1.0 - _normalize(costs))
        + 0.20 * (1.0 - _normalize(times))
        + 0.10 * (1.0 - _normalize(tokens))
    ) * 100.0

    for idx, name in enumerate(llm_names):
        results[name]["composite"] = float(composite[idx])

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
