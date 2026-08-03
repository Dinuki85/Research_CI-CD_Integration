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
MAX_INDENT_SPACES = 16
MAX_LINE_LENGTH = 79
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
ALLOWED_SINGLE_CHAR_VARS = {
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
    """Return 1 if the method has more than MAX_METHOD_LINES non‑empty lines."""
    return int(len(nonempty_lines) > MAX_METHOD_LINES)


def _detect_long_param_list(code: str) -> int:
    """Return 1 if a function definition has more than MAX_PARAMS parameters."""
    signature_match = re.search(r"def\s+\w+\s*\(([^)]*)\)", code)
    if not signature_match:
        return 0
    params = [
        p.strip()
        for p in signature_match.group(1).split(",")
        if p.strip() and p.strip() not in ("self", "cls")
    ]
    return int(len(params) > MAX_PARAMS)


def _detect_deep_nesting(nonempty_lines: List[str]) -> int:
    """Return 1 if any line is indented by MAX_INDENT_SPACES spaces or more."""
    if not nonempty_lines:
        return 0
    max_indent = max(len(line) - len(line.lstrip()) for line in nonempty_lines)
    return int(max_indent >= MAX_INDENT_SPACES)


def _detect_magic_numbers(code: str) -> int:
    """Return 1 if more than two numeric literals are not in the allowed set."""
    numbers = re.findall(r"(?<!\w)(\d+\.?\d*)(?!\w)", code)
    magic = [n for n in numbers if n not in ALLOWED_MAGIC_NUMBERS]
    return int(len(magic) > 2)


def _detect_no_docstring(code: str) -> int:
    """Return 1 if the file contains no triple‑quoted docstring."""
    return int('"""' not in code and "'''" not in code)


def _detect_no_type_hints(code: str) -> int:
    """Return 1 if the code lacks return type hints or parameter annotations."""
    type_hint_pattern = r"->\s*\w|:\s*(str|int|float|bool|None|list|dict|tuple|Any|Optional|List|Dict|Tuple)"
    return int(not bool(re.search(type_hint_pattern, code)))


def _detect_long_lines(lines: List[str]) -> int:
    """Return 1 if any line exceeds MAX_LINE_LENGTH characters."""
    return int(any(len(line) > MAX_LINE_LENGTH for line in lines))


def _detect_commented_code(lines: List[str]) -> int:
    """Return 1 if there are lines that appear to be commented‑out code."""
    pattern = r"\s*#\s*(if |for |while |return |self\.|import |def |class |print\()"
    commented = [line for line in lines if re.match(pattern, line)]
    return int(len(commented) >= 1)


def _detect_poor_naming(code: str) -> int:
    """Return 1 if more than one single‑character variable name is used improperly."""
    single_char_assignments = re.findall(r"\b([a-zA-Z])\s*(?:\+|-|\*|\/)?=(?!=)", code)
    poor = [
        var
        for var in single_char_assignments
        if var.lower() not in ALLOWED_SINGLE_CHAR_VARS
    ]
    return int(len(poor) > 1)


def detect_code_smells(code_str: str) -> Dict[str, int]:
    """Detect common code smells in a Python source string.

    Args:
        code_str: The source code to analyze.

    Returns:
        A dictionary mapping smell names to 0 (absent) or 1 (present).

    Raises:
        None.
    """
    empty_result = {smell: 0 for smell in SMELL_NAMES}
    if not isinstance(code_str, str) or len(code_str.strip()) < 5:
        return empty_result

    normalized_code = code_str.replace("\\n", "\n")
    all_lines = normalized_code.split("\n")
    nonempty_lines = [line for line in all_lines if line.strip()]

    smells: Dict[str, int] = {
        "long_method": _detect_long_method(nonempty_lines),
        "long_param_list": _detect_long_param_list(normalized_code),
        "deep_nesting": _detect_deep_nesting(nonempty_lines),
        "magic_numbers": _detect_magic_numbers(normalized_code),
        "no_docstring": _detect_no_docstring(normalized_code),
        "no_type_hints": _detect_no_type_hints(normalized_code),
        "long_lines": _detect_long_lines(all_lines),
        "commented_code": _detect_commented_code(all_lines),
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


import csv
import re
from pathlib import Path
from typing import Dict, Any

# Constants
QUALITY_MODEL_NAMES = [
    "ridge_regression",
    "decision_tree",
    "random_forest",
    "gradient_boosting",
    "xgboost",
    "svr",
]
METRICS_MODEL_NAMES = ["ridge", "random_forest", "gradient_boosting", "xgboost"]
METRICS_PREFIXES = [("time", _m_time), ("tokens", _m_tokens)]

def _load_models() -> None:
    """Load all model artifacts, scalers, encoders, stats, meta data, and pricing.

    This function populates several global dictionaries and flags. It is safe to call
    multiple times; subsequent calls will be no‑ops once the models are loaded.

    Raises:
        Any exception raised by the underlying file I/O or deserialization
        functions will be propagated.
    """
    global _models_loaded, _q_regressors, _q_scaler, _q_le, _q_stats, _q_meta
    global _m_time, _m_tokens, _m_scaler, _m_le, _m_stats, _m_meta, _prices

    if _models_loaded:
        return

    _load_quality_models()
    _load_metrics_models()
    _load_price_data()

    _models_loaded = True


def _load_quality_models() -> None:
    """Load quality regressors and associated artifacts."""
    global _q_regressors, _q_scaler, _q_le, _q_stats, _q_meta

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


def _load_metrics_models() -> None:
    """Load metrics regressors for time and token predictions and associated artifacts."""
    global _m_time, _m_tokens, _m_scaler, _m_le, _m_stats, _m_meta

    for name in METRICS_MODEL_NAMES:
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


def _load_price_data() -> None:
    """Load pricing information from CSV into the global _prices dictionary."""
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


# ── Feature engineering ──────────────────────────────────────────────────────


LONG_METHOD_THRESHOLD = 15
EPSILON = 1.0


def _compute_structural_features(
    original_code: str,
    refactored_code: str,
    function_name: str,
    model_encoding: int,
) -> list[float]:
    """Calculate structural metrics for a function.

    Args:
        original_code: Source code before refactoring.
        refactored_code: Source code after refactoring.
        function_name: Name of the function being analysed.
        model_encoding: Integer encoding used by the model.

    Returns:
        List of structural feature values.
    """
    original_line_count = float(count_lines(original_code))
    long_method_flag = int(original_line_count > LONG_METHOD_THRESHOLD)

    refactored_line_count = float(count_lines(refactored_code))
    line_delta = original_line_count - refactored_line_count
    line_delta_ratio = line_delta / (original_line_count + EPSILON)

    has_future = int(has_future_import(refactored_code))
    has_doc = int(has_docstring(refactored_code))
    has_type = int(has_type_hints(refactored_code))

    refact_complexity = count_keywords(refactored_code)
    original_complexity = count_keywords(original_code)
    complexity_delta = original_complexity - refact_complexity

    name_length = len(str(function_name))
    is_private = int(
        str(function_name).startswith("_")
        and not str(function_name).startswith("__")
    )
    is_dunder = int(
        str(function_name).startswith("__") and str(function_name).endswith("__")
    )
    is_test = int(str(function_name).startswith("test"))

    repo_avg = float(_q_stats.get("global_repo_mean", original_line_count))
    file_avg = float(_q_stats.get("global_file_mean", original_line_count))
    global_max = float(_q_stats.get("global_max_lines", max(original_line_count, 1)))
    normalized_line_count = original_line_count / (global_max + EPSILON)

    return [
        original_line_count,
        long_method_flag,
        refactored_line_count,
        line_delta,
        line_delta_ratio,
        has_future,
        has_doc,
        has_type,
        refact_complexity,
        original_complexity,
        complexity_delta,
        name_length,
        is_private,
        is_dunder,
        is_test,
        repo_avg,
        file_avg,
        normalized_line_count,
        model_encoding,
    ]


def _compute_smell_features(
    original_code: str,
    refactored_code: str,
) -> list[float]:
    """Calculate code‑smell metrics for a function.

    Args:
        original_code: Source code before refactoring.
        refactored_code: Source code after refactoring.

    Returns:
        List of smell feature values.
    """
    original_smells = detect_code_smells(original_code)
    refactored_smells = detect_code_smells(refactored_code)

    original_total = sum(original_smells.values())
    refactored_total = sum(refactored_smells.values())
    reduction = max(0, original_total - refactored_total)
    reduction_ratio = reduction / (original_total + EPSILON)

    agg_features = [
        original_total,
        refactored_total,
        reduction,
        reduction_ratio,
    ]

    original_per = [original_smells[name] for name in SMELL_NAMES]
    refactored_per = [refactored_smells[name] for name in SMELL_NAMES]
    fixed_per = [
        max(0, original_smells[name] - refactored_smells[name])
        for name in SMELL_NAMES
    ]

    return agg_features + original_per + refactored_per + fixed_per


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
        func_name: Name of the function.
        model_enc: Integer encoding for the model.

    Returns:
        NumPy array containing a single row of feature values.
    """
    structural = _compute_structural_features(
        orig_code, refact_code, func_name, model_enc
    )
    smells = _compute_smell_features(orig_code, refact_code)
    return np.array([structural + smells], dtype=float)


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
DEFAULT_PRICE_PER_TOKEN = 0.0

def _normalize_array(arr: "np.ndarray") -> "np.ndarray":
    """Return min‑max normalized array; fallback to 0.5 when constant."""
    range_val = arr.max() - arr.min()
    if range_val > 0:
        return (arr - arr.min()) / range_val
    return np.full_like(arr, 0.5, dtype=float)


def _get_quality_score(
    orig_code: str,
    refact_proxy: str,
    func_name: str,
    q_enc: int,
    models_available: bool,
    best_q_safe: str,
    best_q_name: str,
    q_scaled_models: list,
) -> float:
    """Predict quality score (0‑10) or return default."""
    if not (models_available and best_q_safe in _q_regressors):
        return DEFAULT_QUALITY_SCORE

    q_features = quality_features(orig_code, refact_proxy, func_name, q_enc)
    reg = _q_regressors[best_q_safe]

    try:
        if _q_scaler is not None and best_q_name in q_scaled_models:
            q_input = _q_scaler.transform(q_features)
        else:
            q_input = q_features
        raw_score = reg.predict(q_input)[0]
        return float(np.clip(raw_score, 0.0, 10.0))
    except Exception:
        return DEFAULT_QUALITY_SCORE


def _get_metrics(
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
        pred_time = int(np.expm1(time_reg.predict(m_features)[0]))
        pred_tokens = int(np.expm1(tokens_reg.predict(m_features)[0]))
        return pred_time, pred_tokens
    except Exception:
        return DEFAULT_PRED_TIME_MS, DEFAULT_PRED_TOKENS


def _get_price_per_token(llm: str) -> float:
    """Resolve price per token for a given LLM, falling back to 0.0."""
    try:
        price = float(_prices.get(llm, DEFAULT_PRICE_PER_TOKEN))
        if price != 0.0:
            return price
        for name, value in _prices.items():
            normalized = name.lower().replace(" ", "_")
            if normalized == llm or llm in normalized or normalized in llm:
                return float(value)
    except Exception:
        pass
    return DEFAULT_PRICE_PER_TOKEN


def predict_all_llms(orig_code: str, func_name: str = "unknown_function") -> dict:
    """Predict quality, latency, token usage and cost for each LLM.

    Args:
        orig_code: Original source code to evaluate.
        func_name: Name of the function being evaluated (default: "unknown_function").

    Returns:
        A dictionary keyed by LLM name containing prediction metrics and a composite score.

    Raises:
        Any exception raised by underlying model loading or prediction is caught and
        defaults are used, so this function never propagates errors.
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

    results: dict[str, dict] = {}
    for llm in MODEL_ORDER:
        try:
            q_enc = int(_q_le.transform([llm])[0]) if _q_le is not None else MODEL_ORDER.index(llm)
        except Exception:
            q_enc = MODEL_ORDER.index(llm)

        try:
            m_enc = int(_m_le.transform([llm])[0]) if _m_le is not None else MODEL_ORDER.index(llm)
        except Exception:
            m_enc = MODEL_ORDER.index(llm)

        quality_score = _get_quality_score(
            orig_code,
            refact_proxy,
            func_name,
            q_enc,
            models_available,
            best_q_safe,
            best_q_name,
            q_scaled_models,
        )

        pred_time_ms, pred_tokens = _get_metrics(
            orig_code,
            refact_proxy,
            func_name,
            m_enc,
            metrics_available,
            best_time_name,
            best_tokens_name,
        )

        price_per_token = _get_price_per_token(llm)

        results[llm] = {
            "quality_score": quality_score,
            "pred_time_ms": pred_time_ms,
            "pred_tokens": pred_tokens,
            "pred_cost": float(pred_tokens) * price_per_token,
            "price_per_token": price_per_token,
            "openrouter_slug": OPENROUTER_MODEL_MAP.get(llm, llm),
            "display_name": MODEL_DISPLAY.get(llm, llm),
        }

    llm_names = list(results.keys())
    scores = np.array([results[name]["quality_score"] for name in llm_names], dtype=float)
    times = np.array([results[name]["pred_time_ms"] for name in llm_names], dtype=float)
    tokens = np.array([results[name]["pred_tokens"] for name in llm_names], dtype=float)
    costs = np.array([results[name].get("pred_cost", 0.0) for name in llm_names], dtype=float)

    composite = (
        0.50 * _normalize_array(scores)
        + 0.20 * (1.0 - _normalize_array(costs))
        + 0.20 * (1.0 - _normalize_array(times))
        + 0.10 * (1.0 - _normalize_array(tokens))
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
