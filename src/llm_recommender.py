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

SMELL_NAMES = [
    "long_method",
    "long_param_list",
    "deep_nesting",
    "magic_numbers",
    "no_docstring",
    "no_type_hints",
    "long_lines",
    "commented_code",
    "poor_naming",
]

_LONG_METHOD_THRESHOLD = 20
_LONG_PARAM_THRESHOLD = 4
_DEEP_NESTING_INDENT = 16
_MAGIC_NUMBERS_THRESHOLD = 2
_COMMENTED_CODE_MIN = 1
_POOR_NAMING_THRESHOLD = 1
_ALLOWED_NUMBERS = {
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
_ALLOWED_VARS = {
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


def _normalize_code(code_str: str) -> str:
    """Replace escaped newlines with real newlines."""
    return code_str.replace("\\n", "\n")


def _extract_nonempty_lines(code: str) -> list[str]:
    """Return a list of lines that contain non‑whitespace characters."""
    return [line for line in code.split("\n") if line.strip()]


def _has_long_method(nonempty_lines: list[str]) -> int:
    """Detect if the method exceeds the line count threshold."""
    return int(len(nonempty_lines) > _LONG_METHOD_THRESHOLD)


def _has_long_param_list(code: str) -> int:
    """Detect if a function definition has too many parameters."""
    match = re.search(r"def\s+\w+\s*\(([^)]*)\)", code)
    if not match:
        return 0
    params = [
        p.strip()
        for p in match.group(1).split(",")
        if p.strip() and p.strip() not in ("self", "cls")
    ]
    return int(len(params) > _LONG_PARAM_THRESHOLD)


def _has_deep_nesting(nonempty_lines: list[str]) -> int:
    """Detect deep nesting based on indentation width."""
    if not nonempty_lines:
        return 0
    max_indent = max(len(line) - len(line.lstrip()) for line in nonempty_lines)
    return int(max_indent >= _DEEP_NESTING_INDENT)


def _has_magic_numbers(code: str) -> int:
    """Detect excessive magic numbers not in the allowed set."""
    numbers = re.findall(r"(?<!\w)(\d+\.?\d*)(?!\w)", code)
    magic = [n for n in numbers if n not in _ALLOWED_NUMBERS]
    return int(len(magic) > _MAGIC_NUMBERS_THRESHOLD)


def _has_no_docstring(code: str) -> int:
    """Return 1 if the code lacks a triple‑quoted docstring."""
    return int('"""' not in code and "'''" not in code)


def _has_no_type_hints(code: str) -> int:
    """Return 1 if the code appears to lack type hints."""
    pattern = r"->\s*\w|:\s*(str|int|float|bool|None|list|dict|tuple|Any|Optional|List|Dict|Tuple)"
    return int(not bool(re.search(pattern, code)))


def _has_long_lines(lines: list[str]) -> int:
    """Return 1 if any line exceeds 79 characters."""
    return int(any(len(line) > 79 for line in lines))


def _has_commented_code(lines: list[str]) -> int:
    """Return 1 if there are commented‑out code fragments."""
    commented = [
        line
        for line in lines
        if re.match(
            r"\s*#\s*(if |for |while |return |self\.|import |def |class |print\()", line
        )
    ]
    return int(len(commented) >= _COMMENTED_CODE_MIN)


def _has_poor_naming(code: str) -> int:
    """Return 1 if there are multiple single‑letter variable assignments not allowed."""
    candidates = re.findall(r"\b([a-zA-Z])\s*(?:\+|-|\*|/)?=(?!=)", code)
    poor = [v for v in candidates if v.lower() not in _ALLOWED_VARS]
    return int(len(poor) > _POOR_NAMING_THRESHOLD)


def detect_code_smells(code_str: str) -> Dict[str, int]:
    """Detect nine common code smells in a Python source string.

    Args:
        code_str: Source code to analyze.

    Returns:
        Mapping of smell names to 0 (absent) or 1 (present).

    Raises:
        None.
    """
    empty_result = {name: 0 for name in SMELL_NAMES}
    if not isinstance(code_str, str) or len(code_str.strip()) < 5:
        return empty_result

    normalized = _normalize_code(code_str)
    all_lines = normalized.split("\n")
    nonempty = _extract_nonempty_lines(normalized)

    return {
        "long_method": _has_long_method(nonempty),
        "long_param_list": _has_long_param_list(normalized),
        "deep_nesting": _has_deep_nesting(nonempty),
        "magic_numbers": _has_magic_numbers(normalized),
        "no_docstring": _has_no_docstring(normalized),
        "no_type_hints": _has_no_type_hints(normalized),
        "long_lines": _has_long_lines(all_lines),
        "commented_code": _has_commented_code(all_lines),
        "poor_naming": _has_poor_naming(normalized),
    }


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

def _load_models() -> None:
    """Load model artifacts, scalers, encoders, stats, meta information, and pricing data.

    This function populates several module‑level globals on first call. Subsequent
    calls are no‑ops.

    Globals Modified:
        _models_loaded, _q_regressors, _q_scaler, _q_le, _q_stats, _q_meta,
        _m_time, _m_tokens, _m_scaler, _m_le, _m_stats, _m_meta, _prices

    Raises:
        Any exception raised by the underlying file I/O or deserialization is
        propagated to the caller.
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
    """Load quality regressors and associated artifacts."""
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


def _load_metrics_regressors() -> None:
    """Load metrics regressors for time and token predictions."""
    global _m_time, _m_tokens, _m_scaler, _m_le, _m_stats, _m_meta

    for name in METRICS_REGRESSOR_NAMES:
        for dict_ref, prefix in METRICS_PREFIXES:
            model_path = MODELS_DIR / f"metrics_{prefix}_{name}.pkl"
            if model_path.exists():
                loaded = _safe_joblib_load(model_path)
                if loaded is not None:
                    dict_ref[name] = loaded

    _m_scaler = _safe_joblib_load(MODELS_DIR / "metrics_scaler.pkl")
    _m_le = _safe_joblib_load(MODELS_DIR / "metrics_label_encoder.pkl")
    _m_stats = _safe_joblib_load(MODELS_DIR / "metrics_stats.pkl")
    _m_meta = _safe_joblib_load(MODELS_DIR / "metrics_meta.pkl")


def _load_price_data() -> None:
    """Parse price.csv and populate the _prices mapping."""
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


import numpy as np

_LONG_METHOD_THRESHOLD = 15
_SMALL_EPSILON = 1.0


def _compute_structural_features(
    orig_code: str,
    refact_code: str,
    func_name: str,
    model_enc: int,
    q_stats: dict,
) -> list[float]:
    """Calculate structural quality features.

    Args:
        orig_code: Original source code.
        refact_code: Refactored source code.
        func_name: Name of the function being analysed.
        model_enc: Integer encoding identifying the model.
        q_stats: Repository-wide statistics dictionary.

    Returns:
        A list of 19 numeric structural features.
    """
    orig_lines = float(count_lines(orig_code))
    refact_lines = float(count_lines(refact_code))

    long_method_flag = int(orig_lines > _LONG_METHOD_THRESHOLD)
    line_delta = orig_lines - refact_lines
    line_delta_ratio = line_delta / (orig_lines + _SMALL_EPSILON)

    future_import = int(has_future_import(refact_code))
    docstring_present = int(has_docstring(refact_code))
    type_hints_present = int(has_type_hints(refact_code))

    complexity_refact = count_keywords(refact_code)
    complexity_orig = count_keywords(orig_code)
    complexity_delta = complexity_orig - complexity_refact

    name_len = len(func_name)
    is_private = int(func_name.startswith("_") and not func_name.startswith("__"))
    is_dunder = int(func_name.startswith("__") and func_name.endswith("__"))
    is_test = int(func_name.startswith("test"))

    repo_avg = float(q_stats.get("global_repo_mean", orig_lines))
    file_avg = float(q_stats.get("global_file_mean", orig_lines))
    global_max = float(q_stats.get("global_max_lines", max(orig_lines, 1)))
    norm_line_count = orig_lines / (global_max + _SMALL_EPSILON)

    return [
        orig_lines,
        long_method_flag,
        refact_lines,
        line_delta,
        line_delta_ratio,
        future_import,
        docstring_present,
        type_hints_present,
        complexity_refact,
        complexity_orig,
        complexity_delta,
        name_len,
        is_private,
        is_dunder,
        is_test,
        repo_avg,
        file_avg,
        norm_line_count,
        float(model_enc),
    ]


def _compute_smell_features(
    orig_code: str,
    refact_code: str,
) -> list[float]:
    """Calculate code‑smell related features.

    Args:
        orig_code: Original source code.
        refact_code: Refactored source code.

    Returns:
        A list of 31 numeric smell features.
    """
    orig_smells = detect_code_smells(orig_code)
    refact_smells = detect_code_smells(refact_code)

    orig_total = sum(orig_smells.values())
    refact_total = sum(refact_smells.values())
    reduction = max(0, orig_total - refact_total)
    reduction_ratio = reduction / (orig_total + _SMALL_EPSILON)

    agg_features = [
        float(orig_total),
        float(refact_total),
        float(reduction),
        reduction_ratio,
    ]

    orig_per = [float(orig_smells[name]) for name in SMELL_NAMES]
    refact_per = [float(refact_smells[name]) for name in SMELL_NAMES]
    fixed_per = [
        float(max(0, orig_smells[name] - refact_smells[name]))
        for name in SMELL_NAMES
    ]

    return agg_features + orig_per + refact_per + fixed_per


def quality_features(
    orig_code: str,
    refact_code: str,
    func_name: str,
    model_enc: int,
) -> np.ndarray:
    """Build a 50‑dimensional feature vector for quality prediction.

    The vector consists of 19 structural features followed by 31 smell
    features (aggregated counts, per‑smell percentages and reductions).

    Args:
        orig_code: Original function source code.
        refact_code: Refactored function source code.
        func_name: Name of the function being analysed.
        model_enc: Integer encoding representing the model identifier.

    Returns:
        A NumPy array of shape (1, 50) containing the feature values as
        floats.

    Raises:
        KeyError: If required keys are missing from the global statistics
            dictionary ``_q_stats``.
    """
    structural = _compute_structural_features(
        orig_code, refact_code, func_name, model_enc, _q_stats
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


from __future__ import annotations
import numpy as np
from typing import Any, Dict, List, Tuple


DEFAULT_QUALITY_SCORE = 5.0
DEFAULT_PRED_TIME_MS = 2000
DEFAULT_PRED_TOKENS = 500
DEFAULT_PRICE_PER_TOKEN = 0.0


def _encode_label(label_encoder: Any, llm: str) -> int:
    """Encode LLM name using provided label encoder or fallback to index."""
    try:
        return int(label_encoder.transform([llm])[0]) if label_encoder is not None else None
    except Exception:
        return None


def _resolve_quality(
    orig_code: str,
    refact_proxy: str,
    func_name: str,
    q_enc: int,
    models_available: bool,
    best_q_safe: str,
    best_q_name: str,
    q_scaled_models: List[str],
) -> float:
    """Predict quality score (0‑10) or return fallback."""
    if not (models_available and best_q_safe in _q_regressors):
        return DEFAULT_QUALITY_SCORE

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
        return DEFAULT_QUALITY_SCORE


def _resolve_metrics(
    orig_code: str,
    refact_proxy: str,
    func_name: str,
    m_enc: int,
    metrics_available: bool,
    best_time_name: str,
    best_tokens_name: str,
) -> Tuple[int, int]:
    """Predict generation time (ms) and token count, or return fallbacks."""
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


def _resolve_price_per_token(llm: str) -> float:
    """Find price per token for an LLM, handling fuzzy matches."""
    try:
        price = float(_prices.get(llm, DEFAULT_PRICE_PER_TOKEN))
        if price != 0.0:
            return price

        for name, value in _prices.items():
            normalized_name = name.lower().replace(" ", "_")
            if normalized_name == llm or llm in normalized_name or normalized_name in llm:
                return float(value)
        return DEFAULT_PRICE_PER_TOKEN
    except Exception:
        return DEFAULT_PRICE_PER_TOKEN


def _normalize_array(arr: np.ndarray) -> np.ndarray:
    """Return min‑max normalised array; zeros when range is zero."""
    range_val = arr.max() - arr.min()
    if range_val > 0:
        return (arr - arr.min()) / range_val
    return np.full_like(arr, 0.5, dtype=float)


def _compute_composite_score(results: Dict[str, Dict[str, Any]]) -> None:
    """Add composite score (0‑100) to each LLM entry in results."""
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


def predict_all_llms(orig_code: str, func_name: str = "unknown_function") -> Dict[str, Dict[str, Any]]:
    """Predict quality, latency, token usage and cost for each supported LLM.

    Args:
        orig_code: Source code to be analysed.
        func_name: Name of the function represented by ``orig_code``. Defaults
            to ``"unknown_function"``.

    Returns:
        Mapping from LLM name to a dictionary containing:
            - quality_score (float)
            - pred_time_ms (int)
            - pred_tokens (int)
            - pred_cost (float)
            - price_per_token (float)
            - openrouter_slug (str)
            - display_name (str)
            - composite (float): aggregated score in [0, 100].

    Raises:
        Any exception raised inside model inference is caught and substituted
        with safe fallback values; the function itself never propagates.
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
        q_enc = _encode_label(_q_le, llm)
        if q_enc is None:
            q_enc = MODEL_ORDER.index(llm)

        m_enc = _encode_label(_m_le, llm)
        if m_enc is None:
            m_enc = MODEL_ORDER.index(llm)

        quality_score = _resolve_quality(
            orig_code,
            refact_proxy,
            func_name,
            q_enc,
            models_available,
            best_q_safe,
            best_q_name,
            q_scaled_models,
        )

        pred_time_ms, pred_tokens = _resolve_metrics(
            orig_code,
            refact_proxy,
            func_name,
            m_enc,
            metrics_available,
            best_time_name,
            best_tokens_name,
        )

        price_per_token = _resolve_price_per_token(llm)

        results[llm] = {
            "quality_score": quality_score,
            "pred_time_ms": pred_time_ms,
            "pred_tokens": pred_tokens,
            "pred_cost": float(pred_tokens) * price_per_token,
            "price_per_token": price_per_token,
            "openrouter_slug": OPENROUTER_MODEL_MAP.get(llm, llm),
            "display_name": MODEL_DISPLAY.get(llm, llm),
        }

    _compute_composite_score(results)
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
