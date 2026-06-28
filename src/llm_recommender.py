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
MAX_INDENT = 16
MAX_MAGIC_COUNT = 2
MAX_LINE_LENGTH = 79
MIN_POOR_NAMING_COUNT = 2

_ALLOWED_NUMBERS = {
    "0", "1", "2", "3", "10", "100", "-1", "0.0", "1.0"
}
_ALLOWED_VARS = {
    "i", "j", "k", "n", "x", "y", "z", "f", "e", "v",
    "_", "s", "c", "p", "q"
}


def _split_code(code_str: str) -> List[str]:
    """Replace escaped newlines and split the code into lines."""
    normalized = code_str.replace("\\n", "\n")
    return normalized.split("\n")


def _nonempty_lines(lines: List[str]) -> List[str]:
    """Return lines that contain non‑whitespace characters."""
    return [line for line in lines if line.strip()]


def _detect_long_method(nonempty: List[str]) -> int:
    """Return 1 if the method exceeds MAX_METHOD_LINES, else 0."""
    return int(len(nonempty) > MAX_METHOD_LINES)


def _detect_long_param_list(code: str) -> int:
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


def _detect_deep_nesting(nonempty: List[str]) -> int:
    """Return 1 if any line is indented by MAX_INDENT spaces or more."""
    if not nonempty:
        return 0
    max_indent = max(len(line) - len(line.lstrip()) for line in nonempty)
    return int(max_indent >= MAX_INDENT)


def _detect_magic_numbers(code: str) -> int:
    """Return 1 if more than MAX_MAGIC_COUNT numeric literals are not allowed."""
    numbers = re.findall(r"(?<!\w)(\d+\.?\d*)(?!\w)", code)
    magic = [n for n in numbers if n not in _ALLOWED_NUMBERS]
    return int(len(magic) > MAX_MAGIC_COUNT)


def _detect_no_docstring(code: str) -> int:
    """Return 1 when the code lacks both triple‑double and triple‑single quotes."""
    return int('"""' not in code and "'''" not in code)


def _detect_no_type_hints(code: str) -> int:
    """Return 1 if the code contains no return‑type hint nor parameter annotation."""
    pattern = (
        r"->\s*\w|:\s*(str|int|float|bool|None|list|dict|tuple|Any|Optional|List|Dict|Tuple)"
    )
    return int(not bool(re.search(pattern, code)))


def _detect_long_lines(lines: List[str]) -> int:
    """Return 1 if any line exceeds MAX_LINE_LENGTH characters."""
    return int(any(len(line) > MAX_LINE_LENGTH for line in lines))


def _detect_commented_code(lines: List[str]) -> int:
    """Return 1 when there is at least one line that looks like commented‑out code."""
    commented = [
        line
        for line in lines
        if re.match(
            r"\s*#\s*(if |for |while |return |self\.|import |def |class |print\()", line
        )
    ]
    return int(len(commented) >= 1)


def _detect_poor_naming(code: str) -> int:
    """Return 1 if more than one single‑letter variable (outside the allowed set) is assigned."""
    poor = [
        var
        for var in re.findall(r"\b([a-zA-Z])\s*(?:\+|-|\*|\/)?=(?!=)", code)
        if var.lower() not in _ALLOWED_VARS
    ]
    return int(len(poor) >= MIN_POOR_NAMING_COUNT)


def detect_code_smells(code_str: str) -> Dict[str, int]:
    """Detect nine common code smells in a Python source string.

    Args:
        code_str: The source code to analyze.

    Returns:
        A dictionary mapping each smell name to 0 (absent) or 1 (present).

    Raises:
        TypeError: If ``code_str`` is not a string.
    """
    if not isinstance(code_str, str) or len(code_str.strip()) < 5:
        return {smell: 0 for smell in SMELL_NAMES}

    lines = _split_code(code_str)
    nonempty = _nonempty_lines(lines)
    code = "\n".join(lines)

    return {
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
    """Load model artifacts into global caches.

    This function populates several global dictionaries and variables with
    pre‑trained models, scalers, label encoders, statistics, metadata, and
    pricing information. It is a no‑op if the models have already been loaded.

    Args:
        None.

    Returns:
        None.

    Raises:
        Any exception raised by the underlying file I/O or deserialization
        functions propagates to the caller.
    """
    global _models_loaded, _q_regressors, _q_scaler, _q_le, _q_stats, _q_meta
    global _m_time, _m_tokens, _m_scaler, _m_le, _m_stats, _m_meta, _prices

    if _models_loaded:
        return

    _load_quality_regressors()
    _load_metrics_regressors()
    _load_prices()

    _models_loaded = True


_QUALITY_REGRESSOR_NAMES = [
    "ridge_regression",
    "decision_tree",
    "random_forest",
    "gradient_boosting",
    "xgboost",
    "svr",
]

_METRICS_REGRESSOR_NAMES = ["ridge", "random_forest", "gradient_boosting", "xgboost"]
_METRICS_PREFIXES = [("_m_time", "metrics_time"), ("_m_tokens", "metrics_tokens")]


def _load_quality_regressors() -> None:
    """Load quality‑related regressors and associated artifacts."""
    for name in _QUALITY_REGRESSOR_NAMES:
        path = MODELS_DIR / f"quality_{name}.pkl"
        if not path.exists():
            continue
        model = _safe_joblib_load(path)
        if model is not None:
            _q_regressors[name] = model

    _q_scaler = _safe_joblib_load(MODELS_DIR / "quality_scaler.pkl")
    _q_le = _safe_joblib_load(MODELS_DIR / "quality_label_encoder.pkl")
    _q_stats = _safe_joblib_load(MODELS_DIR / "quality_stats.pkl")
    _q_meta = _safe_joblib_load(MODELS_DIR / "quality_meta.pkl")


def _load_metrics_regressors() -> None:
    """Load metrics‑related regressors and associated artifacts."""
    for name in _METRICS_REGRESSOR_NAMES:
        for dict_name, prefix in _METRICS_PREFIXES:
            target_dict = globals()[dict_name]
            path = MODELS_DIR / f"{prefix}_{name}.pkl"
            if not path.exists():
                continue
            model = _safe_joblib_load(path)
            if model is not None:
                target_dict[name] = model

    _m_scaler = _safe_joblib_load(MODELS_DIR / "metrics_scaler.pkl")
    _m_le = _safe_joblib_load(MODELS_DIR / "metrics_label_encoder.pkl")
    _m_stats = _safe_joblib_load(MODELS_DIR / "metrics_stats.pkl")
    _m_meta = _safe_joblib_load(MODELS_DIR / "metrics_meta.pkl")


def _load_prices() -> None:
    """Load pricing information from CSV into the global _prices dict."""
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
from typing import List, Dict

# Constants
LONG_METHOD_THRESHOLD = 15
DENOMINATOR_OFFSET = 1.0

def _structural_features(
    orig_code: str,
    refact_code: str,
    func_name: str,
    model_enc: int,
    q_stats: Dict[str, float],
) -> List[float]:
    """Calculate structural quality features for a function.

    Args:
        orig_code: Original source code of the function.
        refact_code: Refactored source code of the function.
        func_name: Name of the function.
        model_enc: Integer encoding representing the model/context.
        q_stats: Dictionary with repository‑wide statistics.

    Returns:
        A list of 19 numeric structural features.
    """
    orig_line_cnt = float(count_lines(orig_code))
    long_method = int(orig_line_cnt > LONG_METHOD_THRESHOLD)

    refact_line_cnt = float(count_lines(refact_code))
    line_delta = orig_line_cnt - refact_line_cnt
    line_delta_ratio = line_delta / (orig_line_cnt + DENOMINATOR_OFFSET)

    has_future_imp = has_future_import(refact_code)
    has_doc = has_docstring(refact_code)
    has_hints = has_type_hints(refact_code)

    complexity_refact = count_keywords(refact_code)
    complexity_orig = count_keywords(orig_code)
    complexity_delta = complexity_orig - complexity_refact

    name_len = len(func_name)
    is_private = int(func_name.startswith("_") and not func_name.startswith("__"))
    is_dunder = int(func_name.startswith("__") and func_name.endswith("__"))
    is_test = int(func_name.startswith("test"))

    repo_avg = float(q_stats.get("global_repo_mean", orig_line_cnt))
    file_avg = float(q_stats.get("global_file_mean", orig_line_cnt))
    global_max = float(q_stats.get("global_max_lines", max(orig_line_cnt, 1)))
    norm_line_cnt = orig_line_cnt / (global_max + DENOMINATOR_OFFSET)

    return [
        orig_line_cnt,
        long_method,
        refact_line_cnt,
        line_delta,
        line_delta_ratio,
        has_future_imp,
        has_doc,
        has_hints,
        complexity_refact,
        complexity_orig,
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


def _smell_features(orig_code: str, refact_code: str) -> List[float]:
    """Calculate smell‑related quality features.

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
    reduction_ratio = reduction / (orig_total + DENOMINATOR_OFFSET)

    agg_features = [
        orig_total,
        refact_total,
        reduction,
        reduction_ratio,
    ]

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
    """Build a 50‑dimensional feature vector for quality prediction.

    The vector consists of 19 structural features and 31 smell‑related
    features derived from the original and refactored implementations of a
    function.

    Args:
        orig_code: Original source code of the function.
        refact_code: Refactored source code of the function.
        func_name: Name of the function being analysed.
        model_enc: Integer encoding for the model/context.

    Returns:
        A NumPy array of shape (1, 50) with dtype ``float`` containing the
        concatenated features.
    """
    structural = _structural_features(
        orig_code, refact_code, func_name, model_enc, _q_stats
    )
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
from typing import Any, Dict, List

DEFAULT_QUALITY = 5.0
DEFAULT_TIME_MS = 2000
DEFAULT_TOKENS = 500


def _encode_model(
    model_name: str,
    label_encoder: Any,
    fallback_order: List[str],
) -> int:
    """Encode a model name using a label encoder or fallback order."""
    try:
        return int(label_encoder.transform([model_name])[0])
    except Exception:
        return fallback_order.index(model_name)


def _predict_quality(
    orig_code: str,
    refact_proxy: str,
    func_name: str,
    model_enc: int,
    models_available: bool,
    best_q_safe: str,
    q_regressors: Dict[str, Any],
    q_scaler: Any,
    q_stats: Dict[str, Any],
    q_meta: Dict[str, Any],
) -> float:
    """Predict the quality score for a single LLM."""
    if not (models_available and best_q_safe in q_regressors):
        return DEFAULT_QUALITY

    q_features = quality_features(
        orig_code, refact_proxy, func_name, model_enc
    )
    reg = q_regressors[best_q_safe]

    try:
        scaled_models = (
            q_stats.get("scaled_models", [])
            if q_stats is not None
            else []
        )
        if q_scaler is not None and best_q_safe in scaled_models:
            q_input = q_scaler.transform(q_features)
        else:
            q_input = q_features
        raw_score = reg.predict(q_input)[0]
        return float(np.clip(raw_score, 0.0, 10.0))
    except Exception:
        return DEFAULT_QUALITY


def _predict_metrics(
    orig_code: str,
    refact_proxy: str,
    func_name: str,
    model_enc: int,
    metrics_available: bool,
    best_time_name: str,
    best_tokens_name: str,
    time_models: Dict[str, Any],
    token_models: Dict[str, Any],
) -> tuple[int, int]:
    """Predict generation time (ms) and token count for a single LLM."""
    if not metrics_available:
        return DEFAULT_TIME_MS, DEFAULT_TOKENS

    m_features = metrics_features(
        orig_code, refact_proxy, func_name, model_enc
    )
    time_reg = time_models.get(best_time_name)
    tokens_reg = token_models.get(best_tokens_name)

    if time_reg is None or tokens_reg is None:
        return DEFAULT_TIME_MS, DEFAULT_TOKENS

    try:
        time_ms = int(np.expm1(time_reg.predict(m_features)[0]))
        tokens = int(np.expm1(tokens_reg.predict(m_features)[0]))
        return time_ms, tokens
    except Exception:
        return DEFAULT_TIME_MS, DEFAULT_TOKENS


def _price_per_token(
    model_name: str,
    price_map: Dict[str, float],
) -> float:
    """Resolve the price per token for a model, handling fuzzy matches."""
    try:
        price = float(price_map.get(model_name, 0.0))
        if price != 0.0:
            return price

        for key, value in price_map.items():
            key_norm = key.lower().replace(" ", "_")
            if (
                key_norm == model_name
                or model_name in key_norm
                or key_norm in model_name
            ):
                return float(value)
        return 0.0
    except Exception:
        return 0.0


def _normalize_array(arr: np.ndarray) -> np.ndarray:
    """Normalize an array to the range [0, 1]; return 0.5 when constant."""
    rng = arr.max() - arr.min()
    if rng > 0:
        return (arr - arr.min()) / rng
    return np.full_like(arr, 0.5, dtype=float)


def predict_all_llms(
    orig_code: str,
    func_name: str = "unknown_function",
) -> Dict[str, Dict[str, Any]]:
    """Predict quality, time, token usage and cost for each LLM.

    Args:
        orig_code: Original source code string.
        func_name: Name of the function being evaluated.

    Returns:
        A dictionary keyed by LLM name. Each value contains:
            quality_score, pred_time_ms, pred_tokens, pred_cost,
            price_per_token, openrouter_slug, display_name,
            and a composite score (0‑100).

    Raises:
        Any exception raised by underlying model loading or prediction
        is caught internally and defaults are used.
    """
    _load_models()
    refact_proxy = orig_code

    models_available = bool(_q_regressors) and _q_scaler is not None and _q_le is not None
    metrics_available = bool(_m_time) and _m_scaler is not None and _m_le is not None

    if models_available and _q_meta is not None:
        best_q_name = _q_meta["best_model"]
        best_q_safe = best_q_name.lower().replace(" ", "_")
    else:
        best_q_name = ""
        best_q_safe = ""

    if metrics_available and _m_meta is not None:
        best_time_name = _m_meta.get("best_time_model", "gradient_boosting").lower().replace(" ", "_")
        best_tokens_name = _m_meta.get("best_tokens_model", "gradient_boosting").lower().replace(" ", "_")
    else:
        best_time_name = ""
        best_tokens_name = ""

    results: Dict[str, Dict[str, Any]] = {}
    for llm in MODEL_ORDER:
        q_enc = _encode_model(llm, _q_le, MODEL_ORDER)
        m_enc = _encode_model(llm, _m_le, MODEL_ORDER)

        quality = _predict_quality(
            orig_code,
            refact_proxy,
            func_name,
            q_enc,
            models_available,
            best_q_safe,
            _q_regressors,
            _q_scaler,
            _q_stats,
            _q_meta,
        )

        time_ms, tokens = _predict_metrics(
            orig_code,
            refact_proxy,
            func_name,
            m_enc,
            metrics_available,
            best_time_name,
            best_tokens_name,
            _m_time,
            _m_tokens,
        )

        price = _price_per_token(llm, _prices)

        results[llm] = {
            "quality_score": quality,
            "pred_time_ms": time_ms,
            "pred_tokens": tokens,
            "pred_cost": float(tokens) * price,
            "price_per_token": price,
            "openrouter_slug": OPENROUTER_MODEL_MAP.get(llm, llm),
            "display_name": MODEL_DISPLAY.get(llm, llm),
        }

    llm_names = list(results.keys())
    scores = np.array([results[n]["quality_score"] for n in llm_names], dtype=float)
    times = np.array([results[n]["pred_time_ms"] for n in llm_names], dtype=float)
    token_counts = np.array([results[n]["pred_tokens"] for n in llm_names], dtype=float)
    costs = np.array([results[n].get("pred_cost", 0.0) for n in llm_names], dtype=float)

    norm_scores = _normalize_array(scores)
    norm_costs = _normalize_array(costs)
    norm_times = _normalize_array(times)
    norm_tokens = _normalize_array(token_counts)

    composite = (
        0.50 * norm_scores
        + 0.20 * (1.0 - norm_costs)
        + 0.20 * (1.0 - norm_times)
        + 0.10 * (1.0 - norm_tokens)
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
