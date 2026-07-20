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
MAX_NESTING_SPACES = 16
MAX_LINE_LENGTH = 79
MIN_CODE_LENGTH = 5
MAX_MAGIC_NUMBERS = 2
MIN_POOR_NAMING_COUNT = 2
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
ALLOWED_TEMP_VARS = {
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
    """Return a list of lines that contain non‑whitespace characters."""
    return [line for line in code.split("\n") if line.strip()]


def _detect_long_method(nonempty_lines: List[str]) -> int:
    """Return 1 if the method has more than MAX_METHOD_LINES non‑empty lines."""
    return int(len(nonempty_lines) > MAX_METHOD_LINES)


def _detect_long_param_list(code: str) -> int:
    """Return 1 if a function definition has more than MAX_PARAMS parameters."""
    signature_match = re.search(r"def\s+\w+\s*\(([^)]*)\)", code)
    if not signature_match:
        return 0
    params = [
        param.strip()
        for param in signature_match.group(1).split(",")
        if param.strip() and param.strip() not in ("self", "cls")
    ]
    return int(len(params) > MAX_PARAMS)


def _detect_deep_nesting(nonempty_lines: List[str]) -> int:
    """Return 1 if any line is indented by MAX_NESTING_SPACES spaces or more."""
    if not nonempty_lines:
        return 0
    max_indent = max(len(line) - len(line.lstrip()) for line in nonempty_lines)
    return int(max_indent >= MAX_NESTING_SPACES)


def _detect_magic_numbers(code: str) -> int:
    """Return 1 if more than MAX_MAGIC_NUMBERS numeric literals are present."""
    numbers = re.findall(r"(?<!\w)(\d+\.?\d*)(?!\w)", code)
    magic_numbers = [n for n in numbers if n not in ALLOWED_MAGIC_NUMBERS]
    return int(len(magic_numbers) > MAX_MAGIC_NUMBERS)


def _detect_missing_docstring(code: str) -> int:
    """Return 1 when neither triple double nor triple single quotes appear."""
    return int('"""' not in code and "'''" not in code)


def _detect_missing_type_hints(code: str) -> int:
    """Return 1 when no return type annotation or parameter annotation is found."""
    type_hint_pattern = re.compile(
        r"->\s*\w|:\s*(str|int|float|bool|None|list|dict|tuple|Any|Optional|List|Dict|Tuple)"
    )
    return int(not bool(type_hint_pattern.search(code)))


def _detect_long_lines(lines: List[str]) -> int:
    """Return 1 if any line exceeds MAX_LINE_LENGTH characters."""
    return int(any(len(line) > MAX_LINE_LENGTH for line in lines))


def _detect_commented_out_code(lines: List[str]) -> int:
    """Return 1 when there is at least one line that looks like commented‑out code."""
    pattern = re.compile(
        r"\s*#\s*(if |for |while |return |self\.|import |def |class |print\()"
    )
    commented = [line for line in lines if pattern.match(line)]
    return int(bool(commented))


def _detect_poor_naming(code: str) -> int:
    """Return 1 if more than one single‑letter variable name is used outside the allowed set."""
    matches = re.findall(r"\b([a-zA-Z])\s*(?:\+|-|\*|/)?=(?!=)", code)
    poor = [var for var in matches if var.lower() not in ALLOWED_TEMP_VARS]
    return int(len(poor) >= MIN_POOR_NAMING_COUNT)


def detect_code_smells(code_str: str) -> Dict[str, int]:
    """Detect nine common code smells in a Python source string.

    Args:
        code_str: The source code to analyze.

    Returns:
        A dictionary mapping smell names to 1 (presence) or 0 (absence).

    Raises:
        None.
    """
    empty_result = {smell: 0 for smell in SMELL_NAMES}
    if not isinstance(code_str, str) or len(code_str.strip()) < MIN_CODE_LENGTH:
        return empty_result

    normalized_code = code_str.replace("\\n", "\n")
    all_lines = normalized_code.split("\n")
    nonempty_lines = _extract_nonempty_lines(normalized_code)

    smells: Dict[str, int] = {
        "long_method": _detect_long_method(nonempty_lines),
        "long_param_list": _detect_long_param_list(normalized_code),
        "deep_nesting": _detect_deep_nesting(nonempty_lines),
        "magic_numbers": _detect_magic_numbers(normalized_code),
        "no_docstring": _detect_missing_docstring(normalized_code),
        "no_type_hints": _detect_missing_type_hints(normalized_code),
        "long_lines": _detect_long_lines(all_lines),
        "commented_code": _detect_commented_out_code(all_lines),
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


QUALITY_REGRESSOR_NAMES = [
    "ridge_regression",
    "decision_tree",
    "random_forest",
    "gradient_boosting",
    "xgboost",
    "svr",
]

METRICS_REGRESSOR_NAMES = ["ridge", "random_forest", "gradient_boosting", "xgboost"]
METRICS_PREFIXES = [("time", _m_time), ("tokens", _m_tokens)]  # type: ignore


def _load_quality_regressors() -> None:
    """Load quality regression models and associated artefacts."""
    for name in QUALITY_REGRESSOR_NAMES:
        file_path = MODELS_DIR / f"quality_{name}.pkl"
        if not file_path.exists():
            continue
        model = _safe_joblib_load(file_path)
        if model is not None:
            _q_regressors[name] = model

    global _q_scaler, _q_le, _q_stats, _q_meta
    _q_scaler = _safe_joblib_load(MODELS_DIR / "quality_scaler.pkl")
    _q_le = _safe_joblib_load(MODELS_DIR / "quality_label_encoder.pkl")
    _q_stats = _safe_joblib_load(MODELS_DIR / "quality_stats.pkl")
    _q_meta = _safe_joblib_load(MODELS_DIR / "quality_meta.pkl")


def _load_metrics_regressors() -> None:
    """Load metrics regression models and associated artefacts."""
    for name in METRICS_REGRESSOR_NAMES:
        for dict_obj, prefix in [( _m_time, "metrics_time"),
                                 ( _m_tokens, "metrics_tokens")]:
            file_path = MODELS_DIR / f"{prefix}_{name}.pkl"
            if not file_path.exists():
                continue
            model = _safe_joblib_load(file_path)
            if model is not None:
                dict_obj[name] = model

    global _m_scaler, _m_le, _m_stats, _m_meta
    _m_scaler = _safe_joblib_load(MODELS_DIR / "metrics_scaler.pkl")
    _m_le = _safe_joblib_load(MODELS_DIR / "metrics_label_encoder.pkl")
    _m_stats = _safe_joblib_load(MODELS_DIR / "metrics_stats.pkl")
    _m_meta = _safe_joblib_load(MODELS_DIR / "metrics_meta.pkl")


def _load_prices() -> None:
    """Parse price CSV and populate the _prices mapping."""
    prices_path = SCRIPT_DIR / "price.csv"
    if not prices_path.exists():
        return

    with open(str(prices_path), newline="") as fh:
        for row in csv.DictReader(fh):
            key = (
                row.get("Model Family")
                or row.get("Model")
                or row.get("model")
            )
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
    """Load all machine‑learning artefacts once per process.

    This function is idempotent; subsequent calls exit early if the
    artefacts have already been loaded.

    Raises:
        Any exception raised by the underlying loading utilities is
        propagated to the caller.
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

# Magic numbers
LONG_METHOD_THRESHOLD = 15
SMALL_EPSILON = 1.0  # used to avoid division by zero


def _structural_features(
    orig_code: str,
    refact_code: str,
    func_name: str,
    model_enc: int,
    q_stats: Dict[str, float],
) -> List[float]:
    """Calculate structural features for a function.

    Args:
        orig_code: Original source code of the function.
        refact_code: Refactored source code of the function.
        func_name: Name of the function.
        model_enc: Integer encoding of the model context.
        q_stats: Dictionary with repository statistics.

    Returns:
        List of 19 numeric structural features.
    """
    orig_line_cnt = float(count_lines(orig_code))
    long_method = int(orig_line_cnt > LONG_METHOD_THRESHOLD)

    refact_line_cnt = float(count_lines(refact_code))
    line_delta = orig_line_cnt - refact_line_cnt
    line_delta_ratio = line_delta / (orig_line_cnt + SMALL_EPSILON)

    future_import = has_future_import(refact_code)
    docstring_present = has_docstring(refact_code)
    type_hints_present = has_type_hints(refact_code)

    refact_complexity = count_keywords(refact_code)
    orig_complexity = count_keywords(orig_code)
    complexity_delta = orig_complexity - refact_complexity

    name_len = len(func_name)
    is_private = int(func_name.startswith("_") and not func_name.startswith("__"))
    is_dunder = int(func_name.startswith("__") and func_name.endswith("__"))
    is_test = int(func_name.startswith("test"))

    repo_mean = float(q_stats.get("global_repo_mean", orig_line_cnt))
    file_mean = float(q_stats.get("global_file_mean", orig_line_cnt))
    global_max = float(q_stats.get("global_max_lines", max(orig_line_cnt, 1)))
    norm_line_cnt = orig_line_cnt / (global_max + SMALL_EPSILON)

    return [
        orig_line_cnt,
        long_method,
        refact_line_cnt,
        line_delta,
        line_delta_ratio,
        future_import,
        docstring_present,
        type_hints_present,
        refact_complexity,
        orig_complexity,
        complexity_delta,
        name_len,
        is_private,
        is_dunder,
        is_test,
        repo_mean,
        file_mean,
        norm_line_cnt,
        model_enc,
    ]


def _smell_features(orig_code: str, refact_code: str) -> List[float]:
    """Calculate code‑smell features for a function.

    Args:
        orig_code: Original source code of the function.
        refact_code: Refactored source code of the function.

    Returns:
        List of 31 numeric smell‑related features.
    """
    orig_smells = detect_code_smells(orig_code)
    refact_smells = detect_code_smells(refact_code)

    orig_total = sum(orig_smells.values())
    refact_total = sum(refact_smells.values())
    reduction = max(0, orig_total - refact_total)
    reduction_ratio = reduction / (orig_total + SMALL_EPSILON)

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
        orig_code: Original source code of the function.
        refact_code: Refactored source code of the function.
        func_name: Name of the function being analyzed.
        model_enc: Integer encoding representing the model context.

    Returns:
        NumPy array of shape (1, 50) containing the feature vector as floats.
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


DEFAULT_QUALITY_SCORE = 5.0
DEFAULT_PRED_TIME_MS = 2000
DEFAULT_PRED_TOKENS = 500
COMPOSITE_WEIGHTS = {
    "quality": 0.50,
    "cost": 0.20,
    "time": 0.20,
    "tokens": 0.10,
}


def _encode_model(
    model_name: str,
    label_encoder,
    fallback_order: list[str],
) -> int:
    """Encode a model name using the provided label encoder.

    If encoding fails, fallback to the index of ``model_name`` in
    ``fallback_order``.
    """
    try:
        return int(label_encoder.transform([model_name])[0]) if label_encoder else fallback_order.index(
            model_name
        )
    except Exception:
        return fallback_order.index(model_name)


def _predict_quality(
    orig_code: str,
    refact_proxy: str,
    func_name: str,
    q_enc: int,
    models_available: bool,
    best_q_name: str,
    best_q_safe: str,
    q_scaled_models: list[str],
) -> float:
    """Predict the quality score for a single LLM."""
    if not (models_available and best_q_safe in _q_regressors):
        return DEFAULT_QUALITY_SCORE

    features = quality_features(orig_code, refact_proxy, func_name, q_enc)
    reg = _q_regressors[best_q_safe]

    try:
        if _q_scaler is not None and best_q_name in q_scaled_models:
            features = _q_scaler.transform(features)
        pred = reg.predict(features)[0]
        return float(np.clip(pred, 0.0, 10.0))
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
    """Predict generation time (ms) and token count for a single LLM."""
    if not metrics_available:
        return DEFAULT_PRED_TIME_MS, DEFAULT_PRED_TOKENS

    features = metrics_features(orig_code, refact_proxy, func_name, m_enc)
    time_reg = _m_time.get(best_time_name)
    tokens_reg = _m_tokens.get(best_tokens_name)

    if time_reg is None or tokens_reg is None:
        return DEFAULT_PRED_TIME_MS, DEFAULT_PRED_TOKENS

    try:
        pred_time = int(np.expm1(time_reg.predict(features)[0]))
        pred_tokens = int(np.expm1(tokens_reg.predict(features)[0]))
        return pred_time, pred_tokens
    except Exception:
        return DEFAULT_PRED_TIME_MS, DEFAULT_PRED_TOKENS


def _price_per_token(model_name: str) -> float:
    """Resolve the price per token for *model_name*."""
    try:
        price = float(_prices.get(model_name, 0.0))
        if price != 0.0:
            return price

        for alias, value in _prices.items():
            normalized = alias.lower().replace(" ", "_")
            if (
                normalized == model_name
                or model_name in normalized
                or normalized in model_name
            ):
                return float(value)
        return 0.0
    except Exception:
        return 0.0


def _normalize_array(arr: np.ndarray) -> np.ndarray:
    """Scale *arr* to the range [0, 1]; returns 0.5 when constant."""
    span = arr.max() - arr.min()
    if span > 0:
        return (arr - arr.min()) / span
    return np.full_like(arr, 0.5, dtype=float)


def _compute_composite(
    scores: np.ndarray,
    times: np.ndarray,
    tokens: np.ndarray,
    costs: np.ndarray,
) -> np.ndarray:
    """Calculate the composite score (0‑100) for each LLM."""
    norm_scores = _normalize_array(scores)
    norm_costs = _normalize_array(costs)
    norm_times = _normalize_array(times)
    norm_tokens = _normalize_array(tokens)

    composite = (
        COMPOSITE_WEIGHTS["quality"] * norm_scores
        + COMPOSITE_WEIGHTS["cost"] * (1.0 - norm_costs)
        + COMPOSITE_WEIGHTS["time"] * (1.0 - norm_times)
        + COMPOSITE_WEIGHTS["tokens"] * (1.0 - norm_tokens)
    )
    return composite * 100.0


def predict_all_llms(orig_code: str, func_name: str = "unknown_function") -> dict:
    """Predict quality, latency, token usage and cost for each LLM.

    Args:
        orig_code: The source code to be analysed.
        func_name: Name of the function being evaluated.

    Returns:
        A dictionary keyed by model name, each containing:
        ``quality_score``, ``pred_time_ms``, ``pred_tokens``,
        ``pred_cost``, ``price_per_token``, ``openrouter_slug``,
        ``display_name`` and ``composite`` (overall score).

    Raises:
        Any exception raised by underlying model loading is caught and
        default values are used instead.
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
        best_tokens_name = _m_meta.get("best_tokens_model", "gradient_boosting").lower().replace(
            " ", "_"
        )
    else:
        best_time_name = ""
        best_tokens_name = ""

    results: dict[str, dict] = {}
    for llm in MODEL_ORDER:
        q_enc = _encode_model(llm, _q_le, MODEL_ORDER)
        m_enc = _encode_model(llm, _m_le, MODEL_ORDER)

        quality = _predict_quality(
            orig_code,
            refact_proxy,
            func_name,
            q_enc,
            models_available,
            best_q_name,
            best_q_safe,
            q_scaled_models,
        )
        pred_time, pred_tokens = _predict_metrics(
            orig_code,
            refact_proxy,
            func_name,
            m_enc,
            metrics_available,
            best_time_name,
            best_tokens_name,
        )
        price = _price_per_token(llm)

        results[llm] = {
            "quality_score": quality,
            "pred_time_ms": pred_time,
            "pred_tokens": pred_tokens,
            "pred_cost": float(pred_tokens) * price,
            "price_per_token": price,
            "openrouter_slug": OPENROUTER_MODEL_MAP.get(llm, llm),
            "display_name": MODEL_DISPLAY.get(llm, llm),
        }

    llm_names = list(results.keys())
    scores = np.array([results[n]["quality_score"] for n in llm_names], dtype=float)
    times = np.array([results[n]["pred_time_ms"] for n in llm_names], dtype=float)
    tokens = np.array([results[n]["pred_tokens"] for n in llm_names], dtype=float)
    costs = np.array([results[n].get("pred_cost", 0.0) for n in llm_names], dtype=float)

    composite = _compute_composite(scores, times, tokens, costs)

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
