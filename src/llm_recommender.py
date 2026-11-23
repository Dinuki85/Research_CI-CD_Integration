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

LONG_METHOD_THRESHOLD = 20
LONG_PARAM_THRESHOLD = 4
DEEP_NESTING_INDENT = 16
MAGIC_NUMBERS_THRESHOLD = 2
LONG_LINE_LENGTH = 79
COMMENTED_CODE_MIN = 1
POOR_NAMING_THRESHOLD = 1

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


def _detect_long_method(nonempty_lines: list[str]) -> int:
    """Return 1 if the method exceeds the line count threshold."""
    return int(len(nonempty_lines) > LONG_METHOD_THRESHOLD)


def _detect_long_param_list(code: str) -> int:
    """Return 1 if a function definition has more than the allowed parameters."""
    match = re.search(r"def\s+\w+\s*\(([^)]*)\)", code)
    if not match:
        return 0
    params = [
        p.strip()
        for p in match.group(1).split(",")
        if p.strip() and p.strip() not in ("self", "cls")
    ]
    return int(len(params) > LONG_PARAM_THRESHOLD)


def _detect_deep_nesting(nonempty_lines: list[str]) -> int:
    """Return 1 if any line is indented deeper than the allowed indent."""
    if not nonempty_lines:
        return 0
    max_indent = max(len(line) - len(line.lstrip()) for line in nonempty_lines)
    return int(max_indent >= DEEP_NESTING_INDENT)


def _detect_magic_numbers(code: str) -> int:
    """Return 1 if more than the allowed count of magic numbers are present."""
    numbers = re.findall(r"(?<!\w)(\d+\.?\d*)(?!\w)", code)
    magic = [n for n in numbers if n not in _ALLOWED_NUMBERS]
    return int(len(magic) > MAGIC_NUMBERS_THRESHOLD)


def _detect_no_docstring(code: str) -> int:
    """Return 1 if the code lacks a triple‑quoted docstring."""
    return int('"""' not in code and "'''" not in code)


def _detect_no_type_hints(code: str) -> int:
    """Return 1 if no type hint syntax is found."""
    pattern = r"->\s*\w|:\s*(str|int|float|bool|None|list|dict|tuple|Any|Optional|List|Dict|Tuple)"
    return int(not bool(re.search(pattern, code)))


def _detect_long_lines(lines: list[str]) -> int:
    """Return 1 if any line exceeds the maximum length."""
    return int(any(len(line) > LONG_LINE_LENGTH for line in lines))


def _detect_commented_code(lines: list[str]) -> int:
    """Return 1 if there is at least one line that looks like commented‑out code."""
    commented = [
        line
        for line in lines
        if re.match(
            r"\s*#\s*(if |for |while |return |self\.|import |def |class |print\()", line
        )
    ]
    return int(len(commented) >= COMMENTED_CODE_MIN)


def _detect_poor_naming(code: str) -> int:
    """Return 1 if more than the allowed number of single‑letter variables are used."""
    candidates = re.findall(r"\b([a-zA-Z])\s*(?:\+|-|\*|\/)?=(?!=)", code)
    poor = [v for v in candidates if v.lower() not in _ALLOWED_VARS]
    return int(len(poor) > POOR_NAMING_THRESHOLD)


def detect_code_smells(code_str: str) -> Dict[str, int]:
    """Detect common Python code smells.

    Args:
        code_str: Source code as a string.

    Returns:
        A dictionary mapping each smell name to 0 (absent) or 1 (present).

    Raises:
        None.
    """
    empty_result = {smell: 0 for smell in SMELL_NAMES}
    if not isinstance(code_str, str) or len(code_str.strip()) < 5:
        return empty_result

    code = code_str.replace("\\n", "\n")
    lines = code.split("\n")
    nonempty = [line for line in lines if line.strip()]

    smells = {
        "long_method": _detect_long_method(nonempty),
        "long_param_list": _detect_long_param_list(code),
        "deep_nesting": _detect_deep_nesting(nonempty),
        "magic_numbers": _detect_magic_numbers(code),
        "no_docstring": _detect_no_docstring(code),
        "no_type_hints": _detect_no_type_hints(code),
        "long_lines": _detect_long_lines(lines),
        "commented_code": _detect_commented_code(lines),
        "poor_naming": _detect_poor_naming(code),
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
    """Load all model artifacts and price data into global caches.

    This function populates several global dictionaries with machine‑learning
    models, scalers, encoders, statistics, and metadata. It also loads a CSV
    containing price information. Subsequent calls are no‑ops.

    Raises:
        Any exception raised by the underlying ``_safe_joblib_load`` or file
        I/O operations propagates to the caller.
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
    """Load quality‑related models, scaler, encoder, stats and metadata."""
    for name in QUALITY_MODEL_NAMES:
        path = MODELS_DIR / f"quality_{name}.pkl"
        model = _load_pickle_if_exists(path)
        if model is not None:
            _q_regressors[name] = model

    _q_scaler = _load_pickle_if_exists(MODELS_DIR / "quality_scaler.pkl")
    _q_le = _load_pickle_if_exists(MODELS_DIR / "quality_label_encoder.pkl")
    _q_stats = _load_pickle_if_exists(MODELS_DIR / "quality_stats.pkl")
    _q_meta = _load_pickle_if_exists(MODELS_DIR / "quality_meta.pkl")


def _load_metrics_models() -> None:
    """Load metrics‑related models, scaler, encoder, stats and metadata."""
    for name in METRICS_MODEL_NAMES:
        for dict_ref, prefix in [( _m_time, "metrics_time"), ( _m_tokens, "metrics_tokens")]:
            path = MODELS_DIR / f"{prefix}_{name}.pkl"
            model = _load_pickle_if_exists(path)
            if model is not None:
                dict_ref[name] = model

    _m_scaler = _load_pickle_if_exists(MODELS_DIR / "metrics_scaler.pkl")
    _m_le = _load_pickle_if_exists(MODELS_DIR / "metrics_label_encoder.pkl")
    _m_stats = _load_pickle_if_exists(MODELS_DIR / "metrics_stats.pkl")
    _m_meta = _load_pickle_if_exists(MODELS_DIR / "metrics_meta.pkl")


def _load_price_data() -> None:
    """Parse ``price.csv`` and store average price per model key."""
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


def _load_pickle_if_exists(path: Path) -> Any:
    """Return the unpickled object if *path* exists, otherwise ``None``."""
    return _safe_joblib_load(path) if path.exists() else None


# ── Feature engineering ──────────────────────────────────────────────────────


LONG_METHOD_THRESHOLD = 15
EPSILON = 1.0


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
        model_enc: Integer encoding of the model.
        q_stats: Dictionary with repository statistics.

    Returns:
        List of 19 structural feature values.
    """
    orig_lines = float(count_lines(orig_code))
    long_method = int(orig_lines > LONG_METHOD_THRESHOLD)

    refact_lines = float(count_lines(refact_code))
    line_delta = orig_lines - refact_lines
    line_delta_ratio = line_delta / (orig_lines + EPSILON)

    has_fut_imp = has_future_import(refact_code)
    has_doc = has_docstring(refact_code)
    has_type = has_type_hints(refact_code)

    refact_complexity = count_keywords(refact_code)
    orig_complexity = count_keywords(orig_code)
    complexity_delta = orig_complexity - refact_complexity

    name_len = len(str(func_name))
    is_private = int(str(func_name).startswith("_") and not str(func_name).startswith("__"))
    is_dunder = int(str(func_name).startswith("__") and str(func_name).endswith("__"))
    is_test = int(str(func_name).startswith("test"))

    repo_avg = float(q_stats.get("global_repo_mean", orig_lines))
    file_avg = float(q_stats.get("global_file_mean", orig_lines))
    global_max = float(q_stats.get("global_max_lines", max(orig_lines, 1)))
    norm_line_cnt = orig_lines / (global_max + EPSILON)

    return [
        orig_lines,
        long_method,
        refact_lines,
        line_delta,
        line_delta_ratio,
        has_fut_imp,
        has_doc,
        has_type,
        refact_complexity,
        orig_complexity,
        complexity_delta,
        name_len,
        is_private,
        is_dunder,
        is_test,
        repo_avg,
        file_avg,
        norm_line_cnt,
        model_enc,
    ]


def _compute_smell_features(orig_code: str, refact_code: str) -> list[float]:
    """Calculate smell‑related quality features.

    Args:
        orig_code: Original source code.
        refact_code: Refactored source code.

    Returns:
        List of 31 smell feature values.
    """
    orig_smells = detect_code_smells(orig_code)
    refact_smells = detect_code_smells(refact_code)

    orig_count = sum(orig_smells.values())
    refact_count = sum(refact_smells.values())
    reduction = max(0, orig_count - refact_count)
    reduction_ratio = reduction / (orig_count + EPSILON)

    agg = [orig_count, refact_count, reduction, reduction_ratio]

    orig_per = [orig_smells[name] for name in SMELL_NAMES]
    refact_per = [refact_smells[name] for name in SMELL_NAMES]
    fixed_per = [
        max(0, orig_smells[name] - refact_smells[name]) for name in SMELL_NAMES
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
        orig_code: Original source code.
        refact_code: Refactored source code.
        func_name: Name of the function being analysed.
        model_enc: Integer encoding of the model.

    Returns:
        NumPy array of shape (1, 50) containing the feature vector.
    """
    q_stats = _q_stats

    structural = _compute_structural_features(
        orig_code, refact_code, func_name, model_enc, q_stats
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


DEFAULT_QUALITY = 5.0
DEFAULT_TIME_MS = 2000
DEFAULT_TOKENS = 500


def _encode_model(model_name: str, label_encoder, fallback_index: int) -> int:
    """Encode a model name using a label encoder or fallback index."""
    try:
        return int(label_encoder.transform([model_name])[0]) if label_encoder is not None else fallback_index
    except Exception:
        return fallback_index


def _predict_quality(
    orig_code: str,
    refact_proxy: str,
    func_name: str,
    model_enc: int,
    models_available: bool,
    best_q_name: str,
    best_q_safe: str,
    q_scaled_models: List[str],
) -> float:
    """Return a quality score between 0 and 10."""
    if not (models_available and best_q_safe in _q_regressors):
        return DEFAULT_QUALITY

    features = quality_features(orig_code, refact_proxy, func_name, model_enc)
    reg = _q_regressors[best_q_safe]

    try:
        if _q_scaler is not None and best_q_name in q_scaled_models:
            features = _q_scaler.transform(features)
        score = float(np.clip(reg.predict(features)[0], 0.0, 10.0))
    except Exception:
        score = DEFAULT_QUALITY

    return score


def _predict_metrics(
    orig_code: str,
    refact_proxy: str,
    func_name: str,
    model_enc: int,
    metrics_available: bool,
    best_time_name: str,
    best_tokens_name: str,
) -> tuple[int, int]:
    """Return predicted time (ms) and token count."""
    if not metrics_available:
        return DEFAULT_TIME_MS, DEFAULT_TOKENS

    features = metrics_features(orig_code, refact_proxy, func_name, model_enc)
    time_reg = _m_time.get(best_time_name)
    tokens_reg = _m_tokens.get(best_tokens_name)

    if time_reg is None or tokens_reg is None:
        return DEFAULT_TIME_MS, DEFAULT_TOKENS

    try:
        pred_time = int(np.expm1(time_reg.predict(features)[0]))
        pred_tokens = int(np.expm1(tokens_reg.predict(features)[0]))
    except Exception:
        pred_time = DEFAULT_TIME_MS
        pred_tokens = DEFAULT_TOKENS

    return pred_time, pred_tokens


def _lookup_price(model_name: str) -> float:
    """Return the price per token for a given model name."""
    try:
        price = float(_prices.get(model_name, 0.0))
        if price != 0.0:
            return price
        for key, value in _prices.items():
            key_norm = key.lower().replace(" ", "_")
            if key_norm == model_name or model_name in key_norm or key_norm in model_name:
                return float(value)
    except Exception:
        pass
    return 0.0


def _normalize_array(arr: np.ndarray) -> np.ndarray:
    """Normalize an array to the range [0, 1]; return 0.5 if constant."""
    rng = arr.max() - arr.min()
    if rng > 0:
        return (arr - arr.min()) / rng
    return np.full_like(arr, 0.5)


def predict_all_llms(orig_code: str, func_name: str = "unknown_function") -> dict[str, dict[str, Any]]:
    """Predict quality, latency, token usage and cost for each LLM.

    Args:
        orig_code: Original source code string.
        func_name: Name of the function being evaluated.

    Returns:
        A dictionary keyed by LLM name. Each value contains:
            quality_score, pred_time_ms, pred_tokens, pred_cost,
            price_per_token, openrouter_slug, display_name, composite.

    Raises:
        Any exception raised by underlying model loading is caught and
        defaults are used instead.
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

    results: dict[str, dict[str, Any]] = {}
    for llm in MODEL_ORDER:
        q_enc = _encode_model(llm, _q_le, MODEL_ORDER.index(llm))
        m_enc = _encode_model(llm, _m_le, MODEL_ORDER.index(llm))

        quality_score = _predict_quality(
            orig_code,
            refact_proxy,
            func_name,
            q_enc,
            models_available,
            best_q_name,
            best_q_safe,
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

        price_per_token = _lookup_price(llm)
        pred_cost = float(pred_tokens) * price_per_token

        results[llm] = {
            "quality_score": quality_score,
            "pred_time_ms": pred_time_ms,
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
