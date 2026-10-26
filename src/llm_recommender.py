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

_MIN_CODE_LENGTH = 5
_LONG_METHOD_THRESHOLD = 20
_LONG_PARAM_LIMIT = 4
_INDENT_THRESHOLD = 16
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
_LONG_LINE_LIMIT = 79
_COMMENTED_CODE_MIN = 1
_POOR_NAMING_LIMIT = 1
_ALLOWED_VARIABLES = {
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


def _extract_nonempty_lines(code: str) -> List[str]:
    """Return a list of non‑empty lines from the given code."""
    return [line for line in code.split("\n") if line.strip()]


def _detect_long_method(nonempty_lines: List[str]) -> int:
    """Return 1 if the method exceeds the line threshold, else 0."""
    return int(len(nonempty_lines) > _LONG_METHOD_THRESHOLD)


def _detect_long_param_list(code: str) -> int:
    """Return 1 if a function definition has too many parameters, else 0."""
    match = re.search(r"def\s+\w+\s*\(([^)]*)\)", code)
    if not match:
        return 0
    params = [
        p.strip()
        for p in match.group(1).split(",")
        if p.strip() and p.strip() not in ("self", "cls")
    ]
    return int(len(params) > _LONG_PARAM_LIMIT)


def _detect_deep_nesting(nonempty_lines: List[str]) -> int:
    """Return 1 if any line is indented beyond the allowed depth, else 0."""
    if not nonempty_lines:
        return 0
    max_indent = max(len(line) - len(line.lstrip()) for line in nonempty_lines)
    return int(max_indent >= _INDENT_THRESHOLD)


def _detect_magic_numbers(code: str) -> int:
    """Return 1 if more than two magic numbers are present, else 0."""
    numbers = re.findall(r"(?<!\w)(\d+\.?\d*)(?!\w)", code)
    magic = [n for n in numbers if n not in _ALLOWED_NUMBERS]
    return int(len(magic) > 2)


def _detect_no_docstring(code: str) -> int:
    """Return 1 when the file lacks a triple‑quoted docstring, else 0."""
    return int('"""' not in code and "'''" not in code)


def _detect_no_type_hints(code: str) -> int:
    """Return 1 when no type hints are found, else 0."""
    pattern = (
        r"->\s*\w|:\s*(str|int|float|bool|None|list|dict|tuple|Any|Optional|"
        r"List|Dict|Tuple)"
    )
    return int(not bool(re.search(pattern, code)))


def _detect_long_lines(lines: List[str]) -> int:
    """Return 1 if any line exceeds the maximum allowed length, else 0."""
    return int(any(len(line) > _LONG_LINE_LIMIT for line in lines))


def _detect_commented_code(lines: List[str]) -> int:
    """Return 1 when commented‑out code patterns are present, else 0."""
    commented = [
        line
        for line in lines
        if re.match(
            r"\s*#\s*(if |for |while |return |self\.|import |def |class |print\()",
            line,
        )
    ]
    return int(len(commented) >= _COMMENTED_CODE_MIN)


def _detect_poor_naming(code: str) -> int:
    """Return 1 if more than the allowed number of single‑letter variables are used."""
    candidates = re.findall(r"\b([a-zA-Z])\s*(?:\+|-|\*|\/)?=(?!=)", code)
    poor = [v for v in candidates if v.lower() not in _ALLOWED_VARIABLES]
    return int(len(poor) > _POOR_NAMING_LIMIT)


def detect_code_smells(code_str: str) -> Dict[str, int]:
    """Detect nine common code smells in a Python source string.

    Args:
        code_str: The source code to analyze.

    Returns:
        A dictionary mapping each smell name to 0 (not detected) or 1 (detected).

    Raises:
        TypeError: If ``code_str`` is not a string.
    """
    if not isinstance(code_str, str):
        raise TypeError("code_str must be a string")

    if len(code_str.strip()) < _MIN_CODE_LENGTH:
        return {name: 0 for name in SMELL_NAMES}

    normalized_code = code_str.replace("\\n", "\n")
    lines = normalized_code.split("\n")
    nonempty_lines = _extract_nonempty_lines(normalized_code)

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


import numpy as np
from typing import List, Dict, Any

LONG_METHOD_THRESHOLD = 15
SMALL_EPS = 1.0

def _compute_structural_features(
    orig_code: str,
    refact_code: str,
    func_name: str,
    model_enc: int,
    q_stats: Dict[str, Any],
) -> List[float]:
    """Calculate structural quality features for a function.

    Args:
        orig_code: Original source code of the function.
        refact_code: Refactored source code of the function.
        func_name: Name of the function.
        model_enc: Integer encoding of the model.
        q_stats: Dictionary containing repository‑wide statistics.

    Returns:
        List of 19 numeric structural features.
    """
    orig_line_cnt = float(count_lines(orig_code))
    refact_line_cnt = float(count_lines(refact_code))
    long_method_flag = int(orig_line_cnt > LONG_METHOD_THRESHOLD)

    line_delta = orig_line_cnt - refact_line_cnt
    line_delta_ratio = line_delta / (orig_line_cnt + SMALL_EPS)

    has_future_imp = has_future_import(refact_code)
    has_doc = has_docstring(refact_code)
    has_type = has_type_hints(refact_code)

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
    norm_line_cnt = orig_line_cnt / (global_max + SMALL_EPS)

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
        name_len,
        is_private,
        is_dunder,
        is_test,
        repo_avg,
        file_avg,
        norm_line_cnt,
        model_enc,
    ]


def _compute_smell_features(
    orig_code: str,
    refact_code: str,
) -> List[float]:
    """Calculate code‑smell related features.

    Args:
        orig_code: Original source code.
        refact_code: Refactored source code.

    Returns:
        List of 31 numeric smell features.
    """
    orig_smells = detect_code_smells(orig_code)
    refact_smells = detect_code_smells(refact_code)

    orig_total = sum(orig_smells.values())
    refact_total = sum(refact_smells.values())
    reduction = max(0, orig_total - refact_total)
    reduction_ratio = reduction / (orig_total + SMALL_EPS)

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
    """Build a 50‑element feature vector for quality prediction.

    The vector consists of 19 structural features and 31 code‑smell features.

    Args:
        orig_code: Original function source code.
        refact_code: Refactored function source code.
        func_name: Name of the function being analysed.
        model_enc: Integer encoding identifying the model/context.

    Returns:
        NumPy array of shape (1, 50) with dtype ``float``.
    """
    q_stats = _q_stats  # Global statistics dictionary
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


DEFAULT_QUALITY_SCORE = 5.0
DEFAULT_PRED_TIME_MS = 2000
DEFAULT_PRED_TOKENS = 500


def _encode_label(
    label_encoder,
    model_name: str,
    fallback_index: int,
) -> int:
    """Encode a model name using a label encoder or fallback index."""
    try:
        return int(label_encoder.transform([model_name])[0])
    except Exception:
        return fallback_index


def _get_quality_score(
    orig_code: str,
    refact_proxy: str,
    func_name: str,
    model_name: str,
    q_enc: int,
) -> float:
    """Predict quality score for a single model."""
    if not (_q_regressors and _q_scaler and _q_le and _q_meta):
        return DEFAULT_QUALITY_SCORE

    best_q_name = _q_meta.get("best_model", "")
    best_q_safe = best_q_name.lower().replace(" ", "_")
    if best_q_safe not in _q_regressors:
        return DEFAULT_QUALITY_SCORE

    q_features = quality_features(
        orig_code, refact_proxy, func_name, q_enc
    )
    reg = _q_regressors[best_q_safe]

    q_scaled_models = (
        _q_stats.get("scaled_models", [])
        if _q_stats is not None
        else []
    )
    if _q_scaler is not None and best_q_name in q_scaled_models:
        q_input = _q_scaler.transform(q_features)
    else:
        q_input = q_features

    try:
        raw_score = reg.predict(q_input)[0]
        return float(np.clip(raw_score, 0.0, 10.0))
    except Exception:
        return DEFAULT_QUALITY_SCORE


def _get_metrics(
    orig_code: str,
    refact_proxy: str,
    func_name: str,
    m_enc: int,
) -> tuple[int, int]:
    """Predict generation time (ms) and token count for a single model."""
    if not (_m_time and _m_scaler and _m_le and _m_meta):
        return DEFAULT_PRED_TIME_MS, DEFAULT_PRED_TOKENS

    best_time_name = _m_meta.get("best_time_model", "gradient_boosting")
    best_tokens_name = _m_meta.get("best_tokens_model", "gradient_boosting")
    best_time_key = best_time_name.lower().replace(" ", "_")
    best_tokens_key = best_tokens_name.lower().replace(" ", "_")

    time_reg = _m_time.get(best_time_key)
    tokens_reg = _m_tokens.get(best_tokens_key)
    if time_reg is None or tokens_reg is None:
        return DEFAULT_PRED_TIME_MS, DEFAULT_PRED_TOKENS

    m_features = metrics_features(
        orig_code, refact_proxy, func_name, m_enc
    )
    try:
        pred_time = int(np.expm1(time_reg.predict(m_features)[0]))
        pred_tokens = int(np.expm1(tokens_reg.predict(m_features)[0]))
        return pred_time, pred_tokens
    except Exception:
        return DEFAULT_PRED_TIME_MS, DEFAULT_PRED_TOKENS


def _get_price_per_token(model_name: str) -> float:
    """Resolve price per token for a model, handling fuzzy matches."""
    try:
        price = float(_prices.get(model_name, 0.0))
        if price != 0.0:
            return price
        for key, val in _prices.items():
            normalized = key.lower().replace(" ", "_")
            if (
                normalized == model_name
                or model_name in normalized
                or normalized in model_name
            ):
                return float(val)
    except Exception:
        pass
    return 0.0


def _normalize_array(arr: np.ndarray) -> np.ndarray:
    """Scale array to [0, 1]; returns 0.5 for constant arrays."""
    span = arr.max() - arr.min()
    if span > 0:
        return (arr - arr.min()) / span
    return np.full_like(arr, 0.5, dtype=float)


def _compute_composite_score(metrics: dict[str, dict]) -> dict[str, float]:
    """Calculate composite scores (0‑100) for each model."""
    model_names = list(metrics.keys())
    scores = np.array(
        [metrics[m]["quality_score"] for m in model_names], dtype=float
    )
    times = np.array(
        [metrics[m]["pred_time_ms"] for m in model_names], dtype=float
    )
    tokens = np.array(
        [metrics[m]["pred_tokens"] for m in model_names], dtype=float
    )
    costs = np.array(
        [metrics[m].get("pred_cost", 0.0) for m in model_names], dtype=float
    )

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

    return {model_names[i]: float(composite[i]) for i in range(len(model_names))}


def predict_all_llms(
    orig_code: str,
    func_name: str = "unknown_function",
) -> dict[str, dict]:
    """Predict quality, latency, token usage and cost for all supported LLMs.

    Args:
        orig_code: Source code to be evaluated.
        func_name: Name of the function within ``orig_code`` (defaults to
            ``"unknown_function"``).

    Returns:
        A dictionary keyed by model name. Each value contains:
        * ``quality_score`` – predicted quality (0‑10).
        * ``pred_time_ms`` – predicted generation time in milliseconds.
        * ``pred_tokens`` – predicted token count.
        * ``pred_cost`` – estimated cost (tokens × price per token).
        * ``price_per_token`` – resolved price per token.
        * ``openrouter_slug`` – slug used by OpenRouter.
        * ``display_name`` – human‑readable model name.
        * ``composite`` – overall composite score (0‑100).

    Raises:
        Any exception raised inside model loading or prediction is caught and
        default fallback values are used, ensuring the function always returns
        a complete dictionary.
    """
    _load_models()
    refact_proxy = orig_code

    results: dict[str, dict] = {}
    for model_name in MODEL_ORDER:
        q_enc = _encode_label(_q_le, model_name, MODEL_ORDER.index(model_name))
        m_enc = _encode_label(_m_le, model_name, MODEL_ORDER.index(model_name))

        quality = _get_quality_score(
            orig_code, refact_proxy, func_name, model_name, q_enc
        )
        pred_time, pred_tokens = _get_metrics(
            orig_code, refact_proxy, func_name, m_enc
        )
        price = _get_price_per_token(model_name)

        results[model_name] = {
            "quality_score": quality,
            "pred_time_ms": pred_time,
            "pred_tokens": pred_tokens,
            "pred_cost": float(pred_tokens) * price,
            "price_per_token": price,
            "openrouter_slug": OPENROUTER_MODEL_MAP.get(
                model_name, model_name
            ),
            "display_name": MODEL_DISPLAY.get(model_name, model_name),
        }

    composite_scores = _compute_composite_score(results)
    for model_name, comp in composite_scores.items():
        results[model_name]["composite"] = comp

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
