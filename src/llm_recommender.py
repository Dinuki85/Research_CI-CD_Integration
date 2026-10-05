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

# Threshold constants
_MIN_CODE_LENGTH = 5
_LONG_METHOD_THRESHOLD = 20
_LONG_PARAM_THRESHOLD = 4
_DEEP_NESTING_INDENT = 16
_MAGIC_NUMBER_THRESHOLD = 2
_LONG_LINE_LENGTH = 79

# Allowed literals
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
    """Convert escaped newlines to real newlines."""
    return code_str.replace("\\n", "\n")


def _nonempty_lines(lines: list[str]) -> list[str]:
    """Return lines that contain non‑whitespace characters."""
    return [line for line in lines if line.strip()]


def _has_long_method(nonempty: list[str]) -> int:
    return int(len(nonempty) > _LONG_METHOD_THRESHOLD)


def _has_long_param_list(code: str) -> int:
    match = re.search(r"def\s+\w+\s*\(([^)]*)\)", code)
    if not match:
        return 0
    params = [
        p.strip()
        for p in match.group(1).split(",")
        if p.strip() and p.strip() not in ("self", "cls")
    ]
    return int(len(params) > _LONG_PARAM_THRESHOLD)


def _has_deep_nesting(nonempty: list[str]) -> int:
    if not nonempty:
        return 0
    max_indent = max(len(line) - len(line.lstrip()) for line in nonempty)
    return int(max_indent >= _DEEP_NESTING_INDENT)


def _has_magic_numbers(code: str) -> int:
    numbers = re.findall(r"(?<!\w)(\d+\.?\d*)(?!\w)", code)
    magic = [n for n in numbers if n not in _ALLOWED_NUMBERS]
    return int(len(magic) > _MAGIC_NUMBER_THRESHOLD)


def _has_no_docstring(code: str) -> int:
    return int('"""' not in code and "'''" not in code)


def _has_no_type_hints(code: str) -> int:
    pattern = r"->\s*\w|:\s*(str|int|float|bool|None|list|dict|tuple|Any|Optional|List|Dict|Tuple)"
    return int(not bool(re.search(pattern, code)))


def _has_long_lines(lines: list[str]) -> int:
    return int(any(len(line) > _LONG_LINE_LENGTH for line in lines))


def _has_commented_code(lines: list[str]) -> int:
    comment_pattern = r"\s*#\s*(if |for |while |return |self\.|import |def |class |print\()"
    commented = [line for line in lines if re.match(comment_pattern, line)]
    return int(len(commented) >= 1)


def _has_poor_naming(code: str) -> int:
    variables = re.findall(r"\b([a-zA-Z])\s*(?:\+|-|\*|\/)?=(?!=)", code)
    poor = [v for v in variables if v.lower() not in _ALLOWED_VARS]
    return int(len(poor) > 1)


def detect_code_smells(code_str: str) -> Dict[str, int]:
    """Detect nine common code smells in a Python code snippet.

    Args:
        code_str: Source code as a string. Escape sequences ``\\n`` are accepted.

    Returns:
        A dictionary mapping each smell name to ``1`` if the smell is present,
        otherwise ``0``. All keys from ``SMELL_NAMES`` are guaranteed to appear.

    Raises:
        TypeError: If ``code_str`` is not a string.
    """
    if not isinstance(code_str, str):
        raise TypeError("code_str must be a string")
    if len(code_str.strip()) < _MIN_CODE_LENGTH:
        return {smell: 0 for smell in SMELL_NAMES}

    code = _normalize_code(code_str)
    lines = code.split("\n")
    nonempty = _nonempty_lines(lines)

    smells = {
        "long_method": _has_long_method(nonempty),
        "long_param_list": _has_long_param_list(code),
        "deep_nesting": _has_deep_nesting(nonempty),
        "magic_numbers": _has_magic_numbers(code),
        "no_docstring": _has_no_docstring(code),
        "no_type_hints": _has_no_type_hints(code),
        "long_lines": _has_long_lines(lines),
        "commented_code": _has_commented_code(lines),
        "poor_naming": _has_poor_naming(code),
    }

    # Ensure all expected smell names are present
    for name in SMELL_NAMES:
        smells.setdefault(name, 0)

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


def _load_models():
    global _models_loaded, _q_regressors, _q_scaler, _q_le, _q_stats, _q_meta
    global _m_time, _m_tokens, _m_scaler, _m_le, _m_stats, _m_meta, _prices

    if _models_loaded:
        return

    # Quality regressors (best-effort)
    for name in ["ridge_regression", "decision_tree", "random_forest",
                 "gradient_boosting", "xgboost", "svr"]:
        fpath = MODELS_DIR / f"quality_{name}.pkl"
        loaded = _safe_joblib_load(fpath) if fpath.exists() else None
        if loaded is not None:
            _q_regressors[name] = loaded

    _q_scaler   = _safe_joblib_load(MODELS_DIR / "quality_scaler.pkl")
    _q_le       = _safe_joblib_load(MODELS_DIR / "quality_label_encoder.pkl")
    _q_stats    = _safe_joblib_load(MODELS_DIR / "quality_stats.pkl")
    _q_meta     = _safe_joblib_load(MODELS_DIR / "quality_meta.pkl")

    # Metrics regressors (best-effort)
    for name in ["ridge", "random_forest", "gradient_boosting", "xgboost"]:
        for d, prefix in [(_m_time, "metrics_time"), (_m_tokens, "metrics_tokens")]:
            fpath = MODELS_DIR / f"{prefix}_{name}.pkl"
            loaded = _safe_joblib_load(fpath) if fpath.exists() else None
            if loaded is not None:
                d[name] = loaded

    _m_scaler = _safe_joblib_load(MODELS_DIR / "metrics_scaler.pkl")
    _m_le     = _safe_joblib_load(MODELS_DIR / "metrics_label_encoder.pkl")
    _m_stats  = _safe_joblib_load(MODELS_DIR / "metrics_stats.pkl")
    _m_meta   = _safe_joblib_load(MODELS_DIR / "metrics_meta.pkl")

    # Load prices
    prices_path = SCRIPT_DIR / "price.csv"
    if prices_path.exists():
        with open(str(prices_path), newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                key = row.get("Model Family") or row.get("Model") or row.get("model")
                inp = row.get("Input") or row.get("input")
                if not key or not inp:
                    continue
                s = str(inp).strip()
                nums = re.findall(r"[0-9]+\.?[0-9]*(?:e[+-]?\d+)?", s)
                try:
                    vals = [float(x) for x in nums]
                    if vals:
                        _prices[key.strip()] = sum(vals) / len(vals)
                except Exception:
                    continue

    _models_loaded = True


# ── Feature engineering ──────────────────────────────────────────────────────


LONG_METHOD_THRESHOLD = 15
MAGIC_ONE = 1.0


def _structural_features(
    orig_code: str,
    refact_code: str,
    func_name: str,
    model_enc: int,
    q_stats: dict,
) -> list[float]:
    """Calculate structural features for a function.

    Args:
        orig_code: Original source code of the function.
        refact_code: Refactored source code of the function.
        func_name: Name of the function.
        model_enc: Integer model encoding.
        q_stats: Dictionary with repository statistics.

    Returns:
        List of 19 floating‑point structural features.
    """
    orig_lines = float(count_lines(orig_code))
    long_method = int(orig_lines > LONG_METHOD_THRESHOLD)

    refact_lines = float(count_lines(refact_code))
    line_delta = orig_lines - refact_lines
    line_delta_ratio = line_delta / (orig_lines + MAGIC_ONE)

    has_future = has_future_import(refact_code)
    has_doc = has_docstring(refact_code)
    has_type = has_type_hints(refact_code)

    complexity_refact = count_keywords(refact_code)
    complexity_orig = count_keywords(orig_code)
    complexity_delta = complexity_orig - complexity_refact

    name_len = len(str(func_name))
    is_private = int(str(func_name).startswith("_") and not str(func_name).startswith("__"))
    is_dunder = int(str(func_name).startswith("__") and str(func_name).endswith("__"))
    is_test = int(str(func_name).startswith("test"))

    repo_avg = float(q_stats.get("global_repo_mean", orig_lines))
    file_avg = float(q_stats.get("global_file_mean", orig_lines))
    global_max = float(q_stats.get("global_max_lines", max(orig_lines, MAGIC_ONE)))
    norm_line = orig_lines / (global_max + MAGIC_ONE)

    return [
        orig_lines,
        long_method,
        refact_lines,
        line_delta,
        line_delta_ratio,
        has_future,
        has_doc,
        has_type,
        complexity_refact,
        complexity_orig,
        complexity_delta,
        name_len,
        is_private,
        is_dunder,
        is_test,
        repo_avg,
        file_avg,
        norm_line,
        float(model_enc),
    ]


def _smell_features(orig_code: str, refact_code: str) -> list[float]:
    """Calculate smell‑related features for a function.

    Args:
        orig_code: Original source code of the function.
        refact_code: Refactored source code of the function.

    Returns:
        List of 31 floating‑point smell features.
    """
    orig_smells = detect_code_smells(orig_code)
    refact_smells = detect_code_smells(refact_code)

    orig_total = sum(orig_smells.values())
    refact_total = sum(refact_smells.values())
    reduction = max(0, orig_total - refact_total)
    reduction_ratio = reduction / (orig_total + MAGIC_ONE)

    agg = [float(orig_total), float(refact_total), float(reduction), reduction_ratio]

    orig_per = [float(orig_smells[name]) for name in SMELL_NAMES]
    refact_per = [float(refact_smells[name]) for name in SMELL_NAMES]
    fixed_per = [
        float(max(0, orig_smells[name] - refact_smells[name])) for name in SMELL_NAMES
    ]

    return agg + orig_per + refact_per + fixed_per


def quality_features(
    orig_code: str,
    refact_code: str,
    func_name: str,
    model_enc: int,
) -> np.ndarray:
    """Build 50 features (19 structural + 31 smell) for quality prediction.

    Args:
        orig_code: Original source code of the target function.
        refact_code: Refactored source code of the target function.
        func_name: Name of the function being analyzed.
        model_enc: Integer encoding for the model (e.g., language identifier).

    Returns:
        NumPy array with shape (1, 50) containing the feature vector as float64.

    Raises:
        KeyError: If required statistics are missing from the global _q_stats.
    """
    structural = _structural_features(orig_code, refact_code, func_name, model_enc, _q_stats)
    smells = _smell_features(orig_code, refact_code)
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


import numpy as np
from typing import Any, Dict, List, Tuple

DEFAULT_QUALITY_SCORE = 5.0
DEFAULT_PRED_TIME_MS = 2000
DEFAULT_PRED_TOKENS = 500
COMPOSITE_WEIGHTS = {
    "quality": 0.50,
    "cost": 0.20,
    "time": 0.20,
    "tokens": 0.10,
}


def _load_and_check_models() -> Tuple[bool, bool]:
    """Load models and return availability flags for quality and metrics."""
    _load_models()
    models_available = bool(_q_regressors) and _q_scaler is not None and _q_le is not None
    metrics_available = bool(_m_time) and _m_scaler is not None and _m_le is not None
    return models_available, metrics_available


def _get_best_quality_meta() -> Tuple[str, str, List[str]]:
    """Extract best quality model name, safe identifier, and scaled model list."""
    if _q_meta is not None:
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
    return best_q_name, best_q_safe, q_scaled_models


def _get_best_metrics_meta() -> Tuple[str, str]:
    """Extract best time and tokens model identifiers."""
    if _m_meta is not None:
        best_time_name = _m_meta.get("best_time_model", "gradient_boosting")
        best_tokens_name = _m_meta.get("best_tokens_model", "gradient_boosting")
        best_time_name = best_time_name.lower().replace(" ", "_")
        best_tokens_name = best_tokens_name.lower().replace(" ", "_")
    else:
        best_time_name = ""
        best_tokens_name = ""
    return best_time_name, best_tokens_name


def _encode_label(
    le_encoder: Any, label: str, fallback_index: int
) -> int:
    """Encode a label using a label encoder or fallback to a provided index."""
    try:
        if le_encoder is not None:
            return int(le_encoder.transform([label])[0])
    except Exception:
        pass
    return fallback_index


def _predict_quality(
    orig_code: str,
    refact_proxy: str,
    func_name: str,
    enc: int,
    best_q_name: str,
    best_q_safe: str,
    q_scaled_models: List[str],
    models_available: bool,
) -> float:
    """Predict the quality score for a single LLM."""
    if not models_available or best_q_safe not in _q_regressors:
        return DEFAULT_QUALITY_SCORE

    q_features = quality_features(orig_code, refact_proxy, func_name, enc)
    regressor = _q_regressors[best_q_safe]

    try:
        if _q_scaler is not None and best_q_name in q_scaled_models:
            q_input = _q_scaler.transform(q_features)
        else:
            q_input = q_features
        raw_score = regressor.predict(q_input)[0]
        return float(np.clip(raw_score, 0.0, 10.0))
    except Exception:
        return DEFAULT_QUALITY_SCORE


def _predict_metrics(
    orig_code: str,
    refact_proxy: str,
    func_name: str,
    enc: int,
    best_time_name: str,
    best_tokens_name: str,
    metrics_available: bool,
) -> Tuple[int, int]:
    """Predict generation time (ms) and token count for a single LLM."""
    if not metrics_available:
        return DEFAULT_PRED_TIME_MS, DEFAULT_PRED_TOKENS

    m_features = metrics_features(orig_code, refact_proxy, func_name, enc)
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
    """Determine the price per token for a given LLM, handling fallbacks."""
    try:
        price = float(_prices.get(llm, 0.0))
        if price != 0.0:
            return price
        for key, value in _prices.items():
            normalized_key = key.lower().replace(" ", "_")
            if normalized_key == llm or llm in normalized_key or normalized_key in llm:
                return float(value)
    except Exception:
        pass
    return 0.0


def _normalize_array(arr: np.ndarray) -> np.ndarray:
    """Normalize an array to [0, 1]; if constant, return 0.5 for all entries."""
    spread = arr.max() - arr.min()
    if spread > 0:
        return (arr - arr.min()) / spread
    return np.full_like(arr, 0.5, dtype=float)


def _compute_composite_score(results: Dict[str, Dict[str, Any]]) -> None:
    """Add a composite score (0‑100) to each LLM entry in results."""
    llm_names = list(results.keys())
    scores = np.array([results[l]["quality_score"] for l in llm_names], dtype=float)
    times = np.array([results[l]["pred_time_ms"] for l in llm_names], dtype=float)
    tokens = np.array([results[l]["pred_tokens"] for l in llm_names], dtype=float)
    costs = np.array([results[l].get("pred_cost", 0.0) for l in llm_names], dtype=float)

    norm_scores = _normalize_array(scores)
    norm_costs = _normalize_array(costs)
    norm_times = _normalize_array(times)
    norm_tokens = _normalize_array(tokens)

    composite = (
        COMPOSITE_WEIGHTS["quality"] * norm_scores
        + COMPOSITE_WEIGHTS["cost"] * (1.0 - norm_costs)
        + COMPOSITE_WEIGHTS["time"] * (1.0 - norm_times)
        + COMPOSITE_WEIGHTS["tokens"] * (1.0 - norm_tokens)
    ) * 100.0

    for idx, llm in enumerate(llm_names):
        results[llm]["composite"] = float(composite[idx])


def predict_all_llms(orig_code: str, func_name: str = "unknown_function") -> Dict[str, Dict[str, Any]]:
    """
    Predict quality score, generation time, token usage and cost for each LLM.

    Args:
        orig_code: The original source code string.
        func_name: Name of the function being analyzed (default: "unknown_function").

    Returns:
        A dictionary keyed by LLM name, each containing:
            - quality_score (float)
            - pred_time_ms (int)
            - pred_tokens (int)
            - pred_cost (float)
            - price_per_token (float)
            - openrouter_slug (str)
            - display_name (str)
            - composite (float): aggregate ranking score (0‑100).

    Raises:
        Any exception raised inside model prediction is caught and default
        values are used, ensuring the function always returns a result.
    """
    models_available, metrics_available = _load_and_check_models()
    refact_proxy = orig_code

    best_q_name, best_q_safe, q_scaled_models = _get_best_quality_meta()
    best_time_name, best_tokens_name = _get_best_metrics_meta()

    results: Dict[str, Dict[str, Any]] = {}
    for llm in MODEL_ORDER:
        q_enc = _encode_label(_q_le, llm, MODEL_ORDER.index(llm))
        m_enc = _encode_label(_m_le, llm, MODEL_ORDER.index(llm))

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
