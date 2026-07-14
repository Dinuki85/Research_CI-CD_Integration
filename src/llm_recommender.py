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
LONG_LINE_LIMIT = 79

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


def _split_code(code_str: str) -> List[str]:
    """Replace escaped newlines and split code into lines."""
    code = code_str.replace("\\n", "\n")
    return code.split("\n")


def _nonempty_lines(lines: List[str]) -> List[str]:
    """Return lines that contain non‑whitespace characters."""
    return [ln for ln in lines if ln.strip()]


def _has_long_method(nonempty: List[str]) -> int:
    """Detect long method smell."""
    return int(len(nonempty) > MAX_METHOD_LINES)


def _has_long_param_list(code: str) -> int:
    """Detect long parameter list smell."""
    match = re.search(r"def\s+\w+\s*\(([^)]*)\)", code)
    if not match:
        return 0
    params = [
        p.strip()
        for p in match.group(1).split(",")
        if p.strip() and p.strip() not in ("self", "cls")
    ]
    return int(len(params) > MAX_PARAMS)


def _has_deep_nesting(nonempty: List[str]) -> int:
    """Detect deep nesting smell."""
    if not nonempty:
        return 0
    max_indent = max(len(l) - len(l.lstrip()) for l in nonempty)
    return int(max_indent >= MAX_INDENT)


def _has_magic_numbers(code: str) -> int:
    """Detect magic numbers smell."""
    numbers = re.findall(r"(?<!\w)(\d+\.?\d*)(?!\w)", code)
    magic = [n for n in numbers if n not in _ALLOWED_NUMBERS]
    return int(len(magic) > 2)


def _has_no_docstring(code: str) -> int:
    """Detect missing docstring smell."""
    return int('"""' not in code and "'''" not in code)


def _has_no_type_hints(code: str) -> int:
    """Detect missing type hints smell."""
    pattern = r"->\s*\w|:\s*(str|int|float|bool|None|list|dict|tuple|Any|Optional|List|Dict|Tuple)"
    return int(not bool(re.search(pattern, code)))


def _has_long_lines(lines: List[str]) -> int:
    """Detect long lines smell."""
    return int(any(len(l) > LONG_LINE_LIMIT for l in lines))


def _has_commented_code(lines: List[str]) -> int:
    """Detect commented‑out code smell."""
    commented = [
        l
        for l in lines
        if re.match(
            r"\s*#\s*(if |for |while |return |self\.|import |def |class |print\()", l
        )
    ]
    return int(len(commented) >= 1)


def _has_poor_naming(code: str) -> int:
    """Detect poor naming smell."""
    candidates = re.findall(r"\b([a-zA-Z])\s*(?:\+|-|\*|\/)?=(?!=)", code)
    poor = [v for v in candidates if v.lower() not in _ALLOWED_VARS]
    return int(len(poor) > 1)


def detect_code_smells(code_str: str) -> Dict[str, int]:
    """Detect nine common code smells in a Python source string.

    Args:
        code_str: Source code to analyze.

    Returns:
        Mapping of smell names to 0 (absent) or 1 (present).

    Raises:
        None.
    """
    empty = {name: 0 for name in SMELL_NAMES}
    if not isinstance(code_str, str) or len(code_str.strip()) < 5:
        return empty

    lines = _split_code(code_str)
    nonempty = _nonempty_lines(lines)
    code = "\n".join(lines)

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


QUALITY_MODEL_NAMES = [
    "ridge_regression",
    "decision_tree",
    "random_forest",
    "gradient_boosting",
    "xgboost",
    "svr",
]

METRICS_MODEL_NAMES = ["ridge", "random_forest", "gradient_boosting", "xgboost"]
METRICS_PREFIXES = [("metrics_time", "_m_time"), ("metrics_tokens", "_m_tokens")]


def _load_quality_regressors() -> None:
    """Load quality regressors and related artifacts."""
    global _q_regressors, _q_scaler, _q_le, _q_stats, _q_meta

    for name in QUALITY_MODEL_NAMES:
        model_path = MODELS_DIR / f"quality_{name}.pkl"
        if not model_path.exists():
            continue
        loaded = _safe_joblib_load(model_path)
        if loaded is not None:
            _q_regressors[name] = loaded

    _q_scaler = _safe_joblib_load(MODELS_DIR / "quality_scaler.pkl")
    _q_le = _safe_joblib_load(MODELS_DIR / "quality_label_encoder.pkl")
    _q_stats = _safe_joblib_load(MODELS_DIR / "quality_stats.pkl")
    _q_meta = _safe_joblib_load(MODELS_DIR / "quality_meta.pkl")


def _load_metrics_regressors() -> None:
    """Load metrics regressors and related artifacts."""
    global _m_time, _m_tokens, _m_scaler, _m_le, _m_stats, _m_meta

    for name in METRICS_MODEL_NAMES:
        for dict_name, attr_name in METRICS_PREFIXES:
            target_dict = globals()[attr_name]
            model_path = MODELS_DIR / f"{dict_name}_{name}.pkl"
            if not model_path.exists():
                continue
            loaded = _safe_joblib_load(model_path)
            if loaded is not None:
                target_dict[name] = loaded

    _m_scaler = _safe_joblib_load(MODELS_DIR / "metrics_scaler.pkl")
    _m_le = _safe_joblib_load(MODELS_DIR / "metrics_label_encoder.pkl")
    _m_stats = _safe_joblib_load(MODELS_DIR / "metrics_stats.pkl")
    _m_meta = _safe_joblib_load(MODELS_DIR / "metrics_meta.pkl")


def _load_prices() -> None:
    """Load price information from CSV into the _prices mapping."""
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
    """Load all model artifacts and price data if not already loaded.

    This function populates several global dictionaries and objects with
    pre‑trained models, scalers, encoders, statistics, metadata, and price
    information. It is safe to call multiple times; subsequent calls are
    no‑ops after the first successful load.

    Raises:
        Any exception raised by `_safe_joblib_load` or file I/O operations
        will propagate to the caller.
    """
    global _models_loaded

    if _models_loaded:
        return

    _load_quality_regressors()
    _load_metrics_regressors()
    _load_prices()

    _models_loaded = True


# ── Feature engineering ──────────────────────────────────────────────────────


import numpy as np
from typing import Dict, List

# Constants
LONG_METHOD_THRESHOLD = 15
SMELL_REDUCTION_FLOOR = 0
SMELL_RATIO_DENOMINATOR_OFFSET = 1


def _compute_structural_features(
    orig_code: str,
    refact_code: str,
    func_name: str,
    model_enc: int,
    q_stats: Dict[str, float],
) -> List[float]:
    """Calculate structural features for a function."""
    orig_line_cnt = float(count_lines(orig_code))
    long_method_flag = int(orig_line_cnt > LONG_METHOD_THRESHOLD)

    refact_line_cnt = float(count_lines(refact_code))
    line_delta = orig_line_cnt - refact_line_cnt
    line_delta_ratio = line_delta / (orig_line_cnt + SMELL_RATIO_DENOMINATOR_OFFSET)

    has_fut_imp = has_future_import(refact_code)
    has_doc = has_docstring(refact_code)
    has_type = has_type_hints(refact_code)

    complexity_refact = count_keywords(refact_code)
    complexity_orig = count_keywords(orig_code)
    complexity_delta = complexity_orig - complexity_refact

    name_len = len(str(func_name))
    is_private = int(str(func_name).startswith("_") and not str(func_name).startswith("__"))
    is_dunder = int(str(func_name).startswith("__") and str(func_name).endswith("__"))
    is_test = int(str(func_name).startswith("test"))

    repo_avg = float(q_stats.get("global_repo_mean", orig_line_cnt))
    file_avg = float(q_stats.get("global_file_mean", orig_line_cnt))
    global_max = float(q_stats.get("global_max_lines", max(orig_line_cnt, 1)))
    norm_line_cnt = orig_line_cnt / (global_max + SMELL_RATIO_DENOMINATOR_OFFSET)

    return [
        orig_line_cnt,
        long_method_flag,
        refact_line_cnt,
        line_delta,
        line_delta_ratio,
        has_fut_imp,
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
        norm_line_cnt,
        model_enc,
    ]


def _compute_smell_features(orig_code: str, refact_code: str) -> List[float]:
    """Calculate smell‑related features for a function."""
    orig_smells = detect_code_smells(orig_code)
    refact_smells = detect_code_smells(refact_code)

    orig_count = sum(orig_smells.values())
    refact_count = sum(refact_smells.values())
    reduction = max(SMELL_REDUCTION_FLOOR, orig_count - refact_count)
    reduction_ratio = reduction / (orig_count + SMELL_RATIO_DENOMINATOR_OFFSET)

    agg_features = [orig_count, refact_count, reduction, reduction_ratio]

    orig_per = [orig_smells[name] for name in SMELL_NAMES]
    refact_per = [refact_smells[name] for name in SMELL_NAMES]
    fixed_per = [
        max(SMELL_REDUCTION_FLOOR, orig_smells[name] - refact_smells[name])
        for name in SMELL_NAMES
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
        orig_code: Original source code of the function.
        refact_code: Refactored source code of the function.
        func_name: Name of the function being analyzed.
        model_enc: Integer encoding of the model identifier.

    Returns:
        A NumPy array of shape (1, 50) containing the computed features as floats.

    Raises:
        Any exception raised by helper utilities (e.g., count_lines,
        detect_code_smells) propagates unchanged.
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


import numpy as np
from typing import Any, Dict, List, Tuple

# Default fallback values
DEFAULT_QUALITY_SCORE = 5.0
DEFAULT_PRED_TIME_MS = 2000
DEFAULT_PRED_TOKENS = 500


def _encode_label(
    label_encoder: Any, label: str, fallback_index: int
) -> int:
    """Encode a label using a fitted encoder or fallback index."""
    try:
        if label_encoder is not None:
            return int(label_encoder.transform([label])[0])
    except Exception:
        pass
    return fallback_index


def _predict_quality(
    llm: str,
    q_enc: int,
    orig_code: str,
    refact_proxy: str,
    func_name: str,
    models_available: bool,
    best_q_safe: str,
    best_q_name: str,
    q_scaled_models: List[str],
) -> float:
    """Predict quality score for a single LLM."""
    if not (models_available and best_q_safe in _q_regressors):
        return DEFAULT_QUALITY_SCORE

    q_features = quality_features(
        orig_code, refact_proxy, func_name, q_enc
    )
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


def _predict_metrics(
    llm: str,
    m_enc: int,
    orig_code: str,
    refact_proxy: str,
    func_name: str,
    metrics_available: bool,
    best_time_name: str,
    best_tokens_name: str,
) -> Tuple[int, int]:
    """Predict generation time (ms) and token count for a single LLM."""
    if not metrics_available:
        return DEFAULT_PRED_TIME_MS, DEFAULT_PRED_TOKENS

    m_features = metrics_features(
        orig_code, refact_proxy, func_name, m_enc
    )
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


def _price_per_token(llm: str) -> float:
    """Resolve price per token for a given LLM, handling fuzzy matches."""
    try:
        price = float(_prices.get(llm, 0.0))
        if price != 0.0:
            return price
        for key, value in _prices.items():
            normalized_key = key.lower().replace(" ", "_")
            if (
                normalized_key == llm
                or llm in normalized_key
                or normalized_key in llm
            ):
                return float(value)
    except Exception:
        pass
    return 0.0


def _normalize_array(arr: np.ndarray) -> np.ndarray:
    """Scale array to [0, 1]; return 0.5 for constant arrays."""
    range_val = arr.max() - arr.min()
    if range_val > 0:
        return (arr - arr.min()) / range_val
    return np.full_like(arr, 0.5, dtype=float)


def predict_all_llms(
    orig_code: str, func_name: str = "unknown_function"
) -> Dict[str, Dict[str, Any]]:
    """Predict quality, latency, token usage and cost for each LLM.

    Args:
        orig_code: Original source code to be evaluated.
        func_name: Name of the function being refactored (default
            "unknown_function").

    Returns:
        Mapping from LLM name to a dictionary containing:
            - quality_score (float)
            - pred_time_ms (int)
            - pred_tokens (int)
            - pred_cost (float)
            - price_per_token (float)
            - openrouter_slug (str)
            - display_name (str)
            - composite (float): weighted composite score (0‑100).

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

    results: Dict[str, Dict[str, Any]] = {}
    for llm in MODEL_ORDER:
        q_enc = _encode_label(_q_le, llm, MODEL_ORDER.index(llm))
        m_enc = _encode_label(_m_le, llm, MODEL_ORDER.index(llm))

        quality_score = _predict_quality(
            llm,
            q_enc,
            orig_code,
            refact_proxy,
            func_name,
            models_available,
            best_q_safe,
            best_q_name,
            q_scaled_models,
        )

        pred_time_ms, pred_tokens = _predict_metrics(
            llm,
            m_enc,
            orig_code,
            refact_proxy,
            func_name,
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
