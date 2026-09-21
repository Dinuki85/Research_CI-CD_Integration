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

# Thresholds and allowed values
LONG_METHOD_THRESHOLD: int = 20
LONG_PARAM_THRESHOLD: int = 4
DEEP_NESTING_INDENT: int = 16
MAGIC_NUMBERS_ALLOWED: set[str] = {
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
POOR_NAMING_ALLOWED: set[str] = {
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


def _detect_long_method(nonempty_lines: List[str]) -> int:
    """Return 1 if the method exceeds the line count threshold."""
    return int(len(nonempty_lines) > LONG_METHOD_THRESHOLD)


def _detect_long_param_list(code: str) -> int:
    """Return 1 if a function definition has more than the allowed parameters."""
    signature_match = re.search(r"def\s+\w+\s*\(([^)]*)\)", code)
    if not signature_match:
        return 0
    raw_params = signature_match.group(1).split(",")
    params = [
        p.strip()
        for p in raw_params
        if p.strip() and p.strip() not in ("self", "cls")
    ]
    return int(len(params) > LONG_PARAM_THRESHOLD)


def _detect_deep_nesting(nonempty_lines: List[str]) -> int:
    """Return 1 if any line is indented beyond the nesting threshold."""
    if not nonempty_lines:
        return 0
    max_indent = max(len(line) - len(line.lstrip()) for line in nonempty_lines)
    return int(max_indent >= DEEP_NESTING_INDENT)


def _detect_magic_numbers(code: str) -> int:
    """Return 1 if more than two magic numbers are present."""
    numbers = re.findall(r"(?<!\w)(\d+\.?\d*)(?!\w)", code)
    magic = [n for n in numbers if n not in MAGIC_NUMBERS_ALLOWED]
    return int(len(magic) > 2)


def _detect_no_docstring(code: str) -> int:
    """Return 1 if the file lacks a triple‑quoted docstring."""
    return int('"""' not in code and "'''" not in code)


def _detect_no_type_hints(code: str) -> int:
    """Return 1 if no type hints are found in function signatures or returns."""
    type_hint_pattern = re.compile(
        r"->\s*\w|:\s*(str|int|float|bool|None|list|dict|tuple|Any|Optional|List|Dict|Tuple)"
    )
    return int(not bool(type_hint_pattern.search(code)))


def _detect_long_lines(lines: List[str]) -> int:
    """Return 1 if any line exceeds 79 characters."""
    return int(any(len(line) > 79 for line in lines))


def _detect_commented_code(lines: List[str]) -> int:
    """Return 1 if there is at least one line that looks like commented‑out code."""
    pattern = re.compile(
        r"\s*#\s*(if |for |while |return |self\.|import |def |class |print\()"
    )
    commented = [line for line in lines if pattern.match(line)]
    return int(len(commented) >= 1)


def _detect_poor_naming(code: str) -> int:
    """Return 1 if more than one single‑letter variable name is used improperly."""
    single_letter_assign = re.findall(r"\b([a-zA-Z])\s*(?:\+|-|\*|\/)?=(?!=)", code)
    poor = [
        var
        for var in single_letter_assign
        if var.lower() not in POOR_NAMING_ALLOWED
    ]
    return int(len(poor) > 1)


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
        empty = {name: 0 for name in SMELL_NAMES}
        return empty

    normalized_code = code_str.replace("\\n", "\n")
    lines = normalized_code.split("\n")
    nonempty = [line for line in lines if line.strip()]

    smells: Dict[str, int] = {
        "long_method": _detect_long_method(nonempty),
        "long_param_list": _detect_long_param_list(normalized_code),
        "deep_nesting": _detect_deep_nesting(nonempty),
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


def _load_models() -> None:
    """Load machine‑learning models, scalers, encoders, statistics, and price data.

    This function lazily loads resources on first call and caches them in module‑level
    globals. Subsequent calls are no‑ops.

    Raises:
        Any exception raised by the underlying loading utilities propagates to the
        caller.
    """
    global _models_loaded, _q_regressors, _q_scaler, _q_le, _q_stats, _q_meta
    global _m_time, _m_tokens, _m_scaler, _m_le, _m_stats, _m_meta, _prices

    if _models_loaded:
        return

    _load_quality_models()
    _load_metrics_models()
    _load_price_data()

    _models_loaded = True


# Constants
_QUALITY_MODEL_NAMES = [
    "ridge_regression",
    "decision_tree",
    "random_forest",
    "gradient_boosting",
    "xgboost",
    "svr",
]

_METRICS_MODEL_NAMES = ["ridge", "random_forest", "gradient_boosting", "xgboost"]
_METRICS_PREFIXES = [
    (_m_time, "metrics_time"),
    (_m_tokens, "metrics_tokens"),
]


def _load_quality_models() -> None:
    """Load quality regressors and associated preprocessing objects."""
    for model_name in _QUALITY_MODEL_NAMES:
        file_path = MODELS_DIR / f"quality_{model_name}.pkl"
        model_obj = _safe_joblib_load(file_path) if file_path.exists() else None
        if model_obj is not None:
            _q_regressors[model_name] = model_obj

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


def _load_metrics_models() -> None:
    """Load metrics regressors for time and token predictions and related objects."""
    for model_name in _METRICS_MODEL_NAMES:
        for target_dict, prefix in _METRICS_PREFIXES:
            file_path = MODELS_DIR / f"{prefix}_{model_name}.pkl"
            model_obj = _safe_joblib_load(file_path) if file_path.exists() else None
            if model_obj is not None:
                target_dict[model_name] = model_obj

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


def _load_price_data() -> None:
    """Parse price.csv and populate the _prices mapping."""
    prices_path = SCRIPT_DIR / "price.csv"
    if not prices_path.exists():
        return

    with open(str(prices_path), newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        for row in reader:
            key = row.get("Model Family") or row.get("Model") or row.get("model")
            input_value = row.get("Input") or row.get("input")
            if not key or not input_value:
                continue

            numeric_strings = re.findall(r"[0-9]+\.?[0-9]*(?:e[+-]?\d+)?", str(input_value).strip())
            try:
                numbers = [float(num) for num in numeric_strings]
                if numbers:
                    _prices[key.strip()] = sum(numbers) / len(numbers)
            except Exception:
                continue


# ── Feature engineering ──────────────────────────────────────────────────────


import numpy as np

# Constants
LONG_METHOD_THRESHOLD = 15
DENOMINATOR_OFFSET = 1.0


def _compute_structural_features(
    orig_code: str,
    refact_code: str,
    func_name: str,
    model_enc: int,
    q_stats: dict,
) -> list[float]:
    """Calculate structural quality features for a function.

    Args:
        orig_code: Original source code of the function.
        refact_code: Refactored source code of the function.
        func_name: Name of the function.
        model_enc: Integer encoding of the model identifier.
        q_stats: Dictionary with repository-wide statistics.

    Returns:
        List of 19 structural feature values.
    """
    orig_line_cnt = float(count_lines(orig_code))
    long_method_flag = int(orig_line_cnt > LONG_METHOD_THRESHOLD)

    refact_line_cnt = float(count_lines(refact_code))
    line_delta = orig_line_cnt - refact_line_cnt
    line_delta_ratio = line_delta / (orig_line_cnt + DENOMINATOR_OFFSET)

    has_future_imp = int(has_future_import(refact_code))
    has_doc = int(has_docstring(refact_code))
    has_type = int(has_type_hints(refact_code))

    complexity_refact = count_keywords(refact_code)
    complexity_orig = count_keywords(orig_code)
    complexity_delta = complexity_orig - complexity_refact

    func_name_len = len(func_name)
    is_private = int(func_name.startswith("_") and not func_name.startswith("__"))
    is_dunder = int(func_name.startswith("__") and func_name.endswith("__"))
    is_test = int(func_name.startswith("test"))

    repo_avg = float(q_stats.get("global_repo_mean", orig_line_cnt))
    file_avg = float(q_stats.get("global_file_mean", orig_line_cnt))
    global_max = float(q_stats.get("global_max_lines", max(orig_line_cnt, DENOMINATOR_OFFSET)))
    norm_line_cnt = orig_line_cnt / (global_max + DENOMINATOR_OFFSET)

    return [
        orig_line_cnt,
        long_method_flag,
        refact_line_cnt,
        line_delta,
        line_delta_ratio,
        has_future_imp,
        has_doc,
        has_type,
        complexity_refact,
        complexity_orig,
        complexity_delta,
        func_name_len,
        is_private,
        is_dunder,
        is_test,
        repo_avg,
        file_avg,
        norm_line_cnt,
        model_enc,
    ]


def _compute_smell_features(orig_code: str, refact_code: str) -> list[float]:
    """Calculate code‑smell related features.

    Args:
        orig_code: Original source code.
        refact_code: Refactored source code.

    Returns:
        List of 31 smell feature values.
    """
    orig_smells = detect_code_smells(orig_code)
    refact_smells = detect_code_smells(refact_code)

    orig_total = sum(orig_smells.values())
    refact_total = sum(refact_smells.values())
    reduction = max(0, orig_total - refact_total)
    reduction_ratio = reduction / (orig_total + DENOMINATOR_OFFSET)

    agg_features = [orig_total, refact_total, reduction, reduction_ratio]

    orig_per = [orig_smells[name] for name in SMELL_NAMES]
    refact_per = [refact_smells[name] for name in SMELL_NAMES]
    fixed_per = [
        max(0, orig_smells[name] - refact_smells[name]) for name in SMELL_NAMES
    ]

    return agg_features + orig_per + refact_per + fixed_per


def quality_features(
    orig_code: str,
    refact_code: str,
    func_name: str,
    model_enc: int,
) -> np.ndarray:
    """Build 50 features (19 structural + 31 smell) for quality prediction.

    Args:
        orig_code: Original function source code.
        refact_code: Refactored function source code.
        func_name: Name of the function being analyzed.
        model_enc: Integer encoding representing the model identifier.

    Returns:
        NumPy array of shape (1, 50) containing the feature vector as floats.
    """
    q_stats = _q_stats  # Global statistics dictionary

    structural = _compute_structural_features(
        orig_code, refact_code, func_name, model_enc, q_stats
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


DEFAULT_QUALITY_SCORE = 5.0
DEFAULT_PRED_TIME_MS = 2000
DEFAULT_PRED_TOKENS = 500

def _encode_model(
    le,
    model_name: str,
    fallback_index: int,
) -> int:
    """Encode a model name using a label encoder or fallback index."""
    try:
        return int(le.transform([model_name])[0]) if le is not None else fallback_index
    except Exception:
        return fallback_index


def _predict_quality(
    orig_code: str,
    refact_proxy: str,
    func_name: str,
    q_enc: int,
    models_available: bool,
    best_q_safe: str,
    best_q_name: str,
    q_scaled_models: list[str],
) -> float:
    """Predict quality score (0‑10) or return a default."""
    if not (models_available and best_q_safe in _q_regressors):
        return DEFAULT_QUALITY_SCORE

    q_features = quality_features(orig_code, refact_proxy, func_name, q_enc)
    reg = _q_regressors[best_q_safe]

    try:
        if _q_scaler is not None and best_q_name in q_scaled_models:
            q_input = _q_scaler.transform(q_features)
        else:
            q_input = q_features
        raw = reg.predict(q_input)[0]
        return float(np.clip(raw, 0.0, 10.0))
    except Exception:
        return DEFAULT_QUALITY_SCORE


def _predict_metrics(
    orig_code: str,
    refact_proxy: str,
    func_name: str,
    m_enc: int,
    metrics_available: bool,
    best_time_name: str,
    best_tokens_name: str,
) -> tuple[int, int]:
    """Predict generation time (ms) and token count, or return defaults."""
    if not metrics_available:
        return DEFAULT_PRED_TIME_MS, DEFAULT_PRED_TOKENS

    m_features = metrics_features(orig_code, refact_proxy, func_name, m_enc)
    time_reg = _m_time.get(best_time_name)
    tokens_reg = _m_tokens.get(best_tokens_name)

    if time_reg is None or tokens_reg is None:
        return DEFAULT_PRED_TIME_MS, DEFAULT_PRED_TOKENS

    try:
        time_ms = int(np.expm1(time_reg.predict(m_features)[0]))
        tokens = int(np.expm1(tokens_reg.predict(m_features)[0]))
        return time_ms, tokens
    except Exception:
        return DEFAULT_PRED_TIME_MS, DEFAULT_PRED_TOKENS


def _price_per_token(model_name: str) -> float:
    """Resolve price per token for a model, falling back to fuzzy matching."""
    try:
        price = float(_prices.get(model_name, 0.0))
        if price != 0.0:
            return price
        for k, v in _prices.items():
            key_norm = k.lower().replace(" ", "_")
            if key_norm == model_name or model_name in key_norm or key_norm in model_name:
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


def _add_composite_score(results: dict) -> None:
    """Compute and attach composite scores (0‑100) to each LLM entry."""
    llms = list(results.keys())
    scores = np.array([results[l]["quality_score"] for l in llms], dtype=float)
    times = np.array([results[l]["pred_time_ms"] for l in llms], dtype=float)
    tokens = np.array([results[l]["pred_tokens"] for l in llms], dtype=float)
    costs = np.array([results[l].get("pred_cost", 0.0) for l in llms], dtype=float)

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

    for idx, llm in enumerate(llms):
        results[llm]["composite"] = float(composite[idx])


def predict_all_llms(
    orig_code: str,
    func_name: str = "unknown_function",
) -> dict:
    """Predict quality, time, token usage and cost for each LLM.

    Args:
        orig_code: Original source code to be evaluated.
        func_name: Name of the function being refactored (default
            ``"unknown_function"``).

    Returns:
        A dictionary keyed by LLM name. Each value contains:
        ``quality_score``, ``pred_time_ms``, ``pred_tokens``,
        ``pred_cost``, ``price_per_token``, ``openrouter_slug``,
        ``display_name`` and a composite score.

    Raises:
        Any exception raised by internal model loading is caught and
        defaults are used; the function itself does not raise.
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
        best_time_name = _m_meta.get("best_time_model", "gradient_boosting")
        best_time_name = best_time_name.lower().replace(" ", "_")
        best_tokens_name = _m_meta.get("best_tokens_model", "gradient_boosting")
        best_tokens_name = best_tokens_name.lower().replace(" ", "_")
    else:
        best_time_name = ""
        best_tokens_name = ""

    results: dict[str, dict] = {}
    for llm in MODEL_ORDER:
        q_enc = _encode_model(_q_le, llm, MODEL_ORDER.index(llm))
        m_enc = _encode_model(_m_le, llm, MODEL_ORDER.index(llm))

        quality_score = _predict_quality(
            orig_code,
            refact_proxy,
            func_name,
            q_enc,
            models_available,
            best_q_safe,
            best_q_name,
            q_scaled_models,
        )

        pred_time_ms, pred_tokens = _predict_metrics(
            orig_code,
            refact_proxy,
            func_name,
            m_enc,
            metrics_available,
            best_time_name,
            best_tokens_name,
        )

        price = _price_per_token(llm)
        pred_cost = float(pred_tokens) * price

        results[llm] = {
            "quality_score": quality_score,
            "pred_time_ms": pred_time_ms,
            "pred_tokens": pred_tokens,
            "pred_cost": pred_cost,
            "price_per_token": price,
            "openrouter_slug": OPENROUTER_MODEL_MAP.get(llm, llm),
            "display_name": MODEL_DISPLAY.get(llm, llm),
        }

    _add_composite_score(results)
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
