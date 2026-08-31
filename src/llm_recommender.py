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

# Constants for smell detection thresholds
_MAX_METHOD_LINES = 20
_MAX_PARAMS = 4
_MAX_INDENT = 16
_MAGIC_NUMBERS_THRESHOLD = 2
_LONG_LINE_LENGTH = 79
_COMMENTED_CODE_MIN = 1
_POOR_NAMING_THRESHOLD = 1

_ALLOWED_NUMBERS = {
    "0", "1", "2", "3", "10", "100", "-1", "0.0", "1.0"
}
_ALLOWED_VARIABLES = {
    "i", "j", "k", "n", "x", "y", "z", "f", "e", "v",
    "_", "s", "c", "p", "q"
}


def _detect_long_method(nonempty_lines: list[str]) -> int:
    """Return 1 if the method exceeds the line count threshold."""
    return int(len(nonempty_lines) > _MAX_METHOD_LINES)


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
    return int(len(params) > _MAX_PARAMS)


def _detect_deep_nesting(nonempty_lines: list[str]) -> int:
    """Return 1 if any line is indented beyond the nesting threshold."""
    if not nonempty_lines:
        return 0
    max_indent = max(len(line) - len(line.lstrip()) for line in nonempty_lines)
    return int(max_indent >= _MAX_INDENT)


def _detect_magic_numbers(code: str) -> int:
    """Return 1 if more than the allowed count of magic numbers are present."""
    numbers = re.findall(r"(?<!\w)(\d+\.?\d*)(?!\w)", code)
    magic = [n for n in numbers if n not in _ALLOWED_NUMBERS]
    return int(len(magic) > _MAGIC_NUMBERS_THRESHOLD)


def _detect_no_docstring(code: str) -> int:
    """Return 1 when the code lacks a triple‑quoted docstring."""
    return int('"""' not in code and "'''" not in code)


def _detect_no_type_hints(code: str) -> int:
    """Return 1 when the code contains no type hint annotations."""
    pattern = r"->\s*\w|:\s*(str|int|float|bool|None|list|dict|tuple|Any|Optional|List|Dict|Tuple)"
    return int(not bool(re.search(pattern, code)))


def _detect_long_lines(lines: list[str]) -> int:
    """Return 1 if any line exceeds the maximum allowed length."""
    return int(any(len(line) > _LONG_LINE_LENGTH for line in lines))


def _detect_commented_code(lines: list[str]) -> int:
    """Return 1 when there is at least one line of commented‑out code."""
    commented = [
        line for line in lines
        if re.match(r"\s*#\s*(if |for |while |return |self\.|import |def |class |print\()", line)
    ]
    return int(len(commented) >= _COMMENTED_CODE_MIN)


def _detect_poor_naming(code: str) -> int:
    """Return 1 if more than the allowed number of single‑letter variables are used."""
    candidates = re.findall(r"\b([a-zA-Z])\s*(?:\+|-|\*|\/)?=(?!=)", code)
    poor = [v for v in candidates if v.lower() not in _ALLOWED_VARIABLES]
    return int(len(poor) > _POOR_NAMING_THRESHOLD)


def detect_code_smells(code_str: str) -> Dict[str, int]:
    """Detect nine common code smells in a Python source string.

    Args:
        code_str: The source code to analyze.

    Returns:
        A dictionary mapping each smell name to 0 (absent) or 1 (present).

    """
    empty_result = {smell: 0 for smell in SMELL_NAMES}
    if not isinstance(code_str, str) or len(code_str.strip()) < 5:
        return empty_result

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


def _load_models() -> None:
    """Load machine‑learning models, scalers, encoders, statistics, and price data.

    This function populates several module‑level globals with objects loaded from
    disk. It is safe to call multiple times; subsequent calls are no‑ops.

    Raises:
        Any exception raised by the underlying file‑loading utilities will
        propagate to the caller.
    """
    global _models_loaded, _q_regressors, _q_scaler, _q_le, _q_stats, _q_meta
    global _m_time, _m_tokens, _m_scaler, _m_le, _m_stats, _m_meta, _prices

    if _models_loaded:
        return

    _load_quality_regressors()
    _load_metrics_regressors()
    _load_price_data()

    _models_loaded = True


def _load_quality_regressors() -> None:
    """Load quality‑related regressors and associated artifacts."""
    quality_model_names = [
        "ridge_regression",
        "decision_tree",
        "random_forest",
        "gradient_boosting",
        "xgboost",
        "svr",
    ]

    for model_name in quality_model_names:
        model_path = MODELS_DIR / f"quality_{model_name}.pkl"
        if not model_path.exists():
            continue
        loaded = _safe_joblib_load(model_path)
        if loaded is not None:
            _q_regressors[model_name] = loaded

    _q_scaler = _safe_joblib_load(MODELS_DIR / "quality_scaler.pkl")
    _q_le = _safe_joblib_load(MODELS_DIR / "quality_label_encoder.pkl")
    _q_stats = _safe_joblib_load(MODELS_DIR / "quality_stats.pkl")
    _q_meta = _safe_joblib_load(MODELS_DIR / "quality_meta.pkl")


def _load_metrics_regressors() -> None:
    """Load metrics‑related regressors and associated artifacts."""
    metrics_model_names = ["ridge", "random_forest", "gradient_boosting", "xgboost"]
    metric_prefixes = [
        (_m_time, "metrics_time"),
        (_m_tokens, "metrics_tokens"),
    ]

    for model_name in metrics_model_names:
        for target_dict, prefix in metric_prefixes:
            model_path = MODELS_DIR / f"{prefix}_{model_name}.pkl"
            if not model_path.exists():
                continue
            loaded = _safe_joblib_load(model_path)
            if loaded is not None:
                target_dict[model_name] = loaded

    _m_scaler = _safe_joblib_load(MODELS_DIR / "metrics_scaler.pkl")
    _m_le = _safe_joblib_load(MODELS_DIR / "metrics_label_encoder.pkl")
    _m_stats = _safe_joblib_load(MODELS_DIR / "metrics_stats.pkl")
    _m_meta = _safe_joblib_load(MODELS_DIR / "metrics_meta.pkl")


def _load_price_data() -> None:
    """Load price information from CSV into the _prices dictionary."""
    prices_path = SCRIPT_DIR / "price.csv"
    if not prices_path.exists():
        return

    with open(str(prices_path), newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        for row in reader:
            key = row.get("Model Family") or row.get("Model") or row.get("model")
            input_field = row.get("Input") or row.get("input")
            if not key or not input_field:
                continue

            numeric_strings = re.findall(
                r"[0-9]+\.?[0-9]*(?:e[+-]?\d+)?", str(input_field).strip()
            )
            try:
                values = [float(num) for num in numeric_strings]
                if values:
                    _prices[key.strip()] = sum(values) / len(values)
            except Exception:
                continue


# ── Feature engineering ──────────────────────────────────────────────────────


LONG_METHOD_THRESHOLD = 15
DENOMINATOR_OFFSET = 1.0


def _compute_structural_features(
    original_code: str,
    refactored_code: str,
    function_name: str,
    model_encoding: int,
    q_stats: dict,
) -> list[float]:
    """Calculate structural quality features.

    Args:
        original_code: Source code before refactoring.
        refactored_code: Source code after refactoring.
        function_name: Name of the function being analysed.
        model_encoding: Integer encoding used by the prediction model.
        q_stats: Dictionary with repository-wide statistics.

    Returns:
        List of 19 numeric structural features.
    """
    original_line_count = float(count_lines(original_code))
    long_method_flag = int(original_line_count > LONG_METHOD_THRESHOLD)

    refactored_line_count = float(count_lines(refactored_code))
    line_delta = original_line_count - refactored_line_count
    line_delta_ratio = line_delta / (original_line_count + DENOMINATOR_OFFSET)

    has_future = int(has_future_import(refactored_code))
    has_doc = int(has_docstring(refactored_code))
    has_type = int(has_type_hints(refactored_code))

    refact_complexity = count_keywords(refactored_code)
    original_complexity = count_keywords(original_code)
    complexity_delta = original_complexity - refact_complexity

    name_length = len(str(function_name))
    is_private = int(str(function_name).startswith("_") and not str(function_name).startswith("__"))
    is_dunder = int(str(function_name).startswith("__") and str(function_name).endswith("__"))
    is_test = int(str(function_name).startswith("test"))

    repo_avg = float(q_stats.get("global_repo_mean", original_line_count))
    file_avg = float(q_stats.get("global_file_mean", original_line_count))
    global_max = float(q_stats.get("global_max_lines", max(original_line_count, 1)))
    normalized_line_count = original_line_count / (global_max + DENOMINATOR_OFFSET)

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
    """Calculate smell‑related quality features.

    Args:
        original_code: Source code before refactoring.
        refactored_code: Source code after refactoring.

    Returns:
        List of 31 numeric smell features.
    """
    original_smells = detect_code_smells(original_code)
    refactored_smells = detect_code_smells(refactored_code)

    original_total = sum(original_smells.values())
    refactored_total = sum(refactored_smells.values())
    reduction = max(0, original_total - refactored_total)
    reduction_ratio = reduction / (original_total + DENOMINATOR_OFFSET)

    aggregate = [
        original_total,
        refactored_total,
        reduction,
        reduction_ratio,
    ]

    original_per_type = [original_smells[name] for name in SMELL_NAMES]
    refactored_per_type = [refactored_smells[name] for name in SMELL_NAMES]
    fixed_per_type = [
        max(0, original_smells[name] - refactored_smells[name]) for name in SMELL_NAMES
    ]

    return aggregate + original_per_type + refactored_per_type + fixed_per_type


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
        func_name: Name of the function being analysed.
        model_enc: Integer encoding for the model.

    Returns:
        A NumPy array of shape (1, 50) containing the feature vector.
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


DEFAULT_QUALITY_SCORE = 5.0
DEFAULT_PRED_TIME_MS = 2000
DEFAULT_PRED_TOKENS = 500


def _encode_model(
    label_encoder: Optional[Any],
    model_name: str,
    fallback_index: int,
) -> int:
    """Encode a model name using a label encoder or fallback index."""
    try:
        return int(label_encoder.transform([model_name])[0]) if label_encoder is not None else fallback_index
    except Exception:
        return fallback_index


def _compute_quality_score(
    orig_code: str,
    refact_proxy: str,
    func_name: str,
    model_enc: int,
    models_available: bool,
    best_q_safe: str,
    best_q_name: str,
    q_regressors: Dict[str, Any],
    q_scaler: Optional[Any],
    q_scaled_models: List[str],
) -> float:
    """Predict quality score for a single LLM."""
    if not (models_available and best_q_safe in q_regressors):
        return DEFAULT_QUALITY_SCORE

    q_features = quality_features(orig_code, refact_proxy, func_name, model_enc)
    reg = q_regressors[best_q_safe]

    try:
        if q_scaler is not None and best_q_name in q_scaled_models:
            q_input = q_scaler.transform(q_features)
        else:
            q_input = q_features
        raw_score = reg.predict(q_input)[0]
        return float(np.clip(raw_score, 0.0, 10.0))
    except Exception:
        return DEFAULT_QUALITY_SCORE


def _compute_metrics(
    orig_code: str,
    refact_proxy: str,
    func_name: str,
    model_enc: int,
    metrics_available: bool,
    time_regressor: Optional[Any],
    tokens_regressor: Optional[Any],
) -> Tuple[int, int]:
    """Predict generation time (ms) and token count for a single LLM."""
    if not metrics_available or time_regressor is None or tokens_regressor is None:
        return DEFAULT_PRED_TIME_MS, DEFAULT_PRED_TOKENS

    m_features = metrics_features(orig_code, refact_proxy, func_name, model_enc)

    try:
        pred_time_ms = int(np.expm1(time_regressor.predict(m_features)[0]))
        pred_tokens = int(np.expm1(tokens_regressor.predict(m_features)[0]))
        return pred_time_ms, pred_tokens
    except Exception:
        return DEFAULT_PRED_TIME_MS, DEFAULT_PRED_TOKENS


def _get_price_per_token(model_name: str, prices: Dict[str, Any]) -> float:
    """Lookup price per token for a model, with fuzzy matching fallback."""
    try:
        price = float(prices.get(model_name, 0.0))
        if price != 0.0:
            return price
        for key, value in prices.items():
            normalized_key = key.lower().replace(" ", "_")
            if normalized_key == model_name or model_name in normalized_key or normalized_key in model_name:
                return float(value)
    except Exception:
        pass
    return 0.0


def _normalize_array(arr: np.ndarray) -> np.ndarray:
    """Normalize an array to [0, 1]; return 0.5 for constant arrays."""
    range_val = arr.max() - arr.min()
    if range_val > 0:
        return (arr - arr.min()) / range_val
    return np.full_like(arr, 0.5, dtype=float)


def predict_all_llms(orig_code: str, func_name: str = "unknown_function") -> Dict[str, Dict[str, Any]]:
    """Predict quality, latency, token usage and cost for each LLM.

    Args:
        orig_code: Original source code to be evaluated.
        func_name: Name of the function being evaluated (default: "unknown_function").

    Returns:
        A dictionary keyed by LLM name. Each value contains:
            - quality_score (float): Predicted quality (0‑10).
            - pred_time_ms (int): Predicted generation time in milliseconds.
            - pred_tokens (int): Predicted token count.
            - pred_cost (float): Estimated cost (tokens * price per token).
            - price_per_token (float): Token price for the model.
            - openrouter_slug (str): Identifier used by OpenRouter.
            - display_name (str): Human‑readable model name.
            - composite (float): Composite score (0‑100) combining quality, cost,
              latency and token count.

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
        q_enc = _encode_model(_q_le, llm, MODEL_ORDER.index(llm))
        m_enc = _encode_model(_m_le, llm, MODEL_ORDER.index(llm))

        quality_score = _compute_quality_score(
            orig_code,
            refact_proxy,
            func_name,
            q_enc,
            models_available,
            best_q_safe,
            best_q_name,
            _q_regressors,
            _q_scaler,
            q_scaled_models,
        )

        time_reg = _m_time.get(best_time_name) if _m_time else None
        tokens_reg = _m_tokens.get(best_tokens_name) if _m_tokens else None
        pred_time_ms, pred_tokens = _compute_metrics(
            orig_code,
            refact_proxy,
            func_name,
            m_enc,
            metrics_available,
            time_reg,
            tokens_reg,
        )

        price_per_token = _get_price_per_token(llm, _prices)

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
