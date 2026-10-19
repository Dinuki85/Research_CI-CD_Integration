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
MAX_LINE_LENGTH = 79
MAGIC_NUMBER_ALLOWLIST = {
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
ALLOWED_VAR_NAMES = {
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


def _split_lines(code: str) -> List[str]:
    """Split code into a list of lines."""
    return code.split("\n")


def _nonempty_lines(lines: List[str]) -> List[str]:
    """Return lines that contain non‑whitespace characters."""
    return [line for line in lines if line.strip()]


def _detect_long_method(nonempty: List[str]) -> int:
    """Return 1 if the method exceeds MAX_METHOD_LINES, else 0."""
    return int(len(nonempty) > MAX_METHOD_LINES)


def _detect_long_param_list(code: str) -> int:
    """Return 1 if a function definition has more than MAX_PARAMS parameters."""
    signature = re.search(r"def\s+\w+\s*\(([^)]*)\)", code)
    if not signature:
        return 0
    raw_params = signature.group(1).split(",")
    params = [
        p.strip()
        for p in raw_params
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
    """Return 1 if more than two numeric literals are not in the allowlist."""
    numbers = re.findall(r"(?<!\w)(\d+\.?\d*)(?!\w)", code)
    magic = [n for n in numbers if n not in MAGIC_NUMBER_ALLOWLIST]
    return int(len(magic) > 2)


def _detect_no_docstring(code: str) -> int:
    """Return 1 if the file contains no triple‑quoted docstring."""
    return int('"""' not in code and "'''" not in code)


def _detect_no_type_hints(code: str) -> int:
    """Return 1 if no return type hint or parameter annotation is found."""
    pattern = (
        r"->\s*\w|:\s*(str|int|float|bool|None|list|dict|tuple|Any|"
        r"Optional|List|Dict|Tuple)"
    )
    return int(not bool(re.search(pattern, code)))


def _detect_long_lines(lines: List[str]) -> int:
    """Return 1 if any line exceeds MAX_LINE_LENGTH characters."""
    return int(any(len(line) > MAX_LINE_LENGTH for line in lines))


def _detect_commented_code(lines: List[str]) -> int:
    """Return 1 if there is at least one line that looks like commented‑out code."""
    pattern = r"\s*#\s*(if |for |while |return |self\.|import |def |class |print\()"
    commented = [line for line in lines if re.match(pattern, line)]
    return int(len(commented) >= 1)


def _detect_poor_naming(code: str) -> int:
    """Return 1 if more than one single‑letter variable name is used outside the allowlist."""
    candidates = re.findall(r"\b([a-zA-Z])\s*(?:\+|-|\*|/)?=(?!=)", code)
    poor = [v for v in candidates if v.lower() not in ALLOWED_VAR_NAMES]
    return int(len(poor) > 1)


def detect_code_smells(code_str: str) -> Dict[str, int]:
    """Detect nine common Python code smells.

    Args:
        code_str: Source code as a string. Escaped newlines (\"\\n\") are allowed.

    Returns:
        A dictionary mapping each smell name to 0 (absent) or 1 (present).

    Raises:
        TypeError: If ``code_str`` is not a string.
    """
    if not isinstance(code_str, str):
        raise TypeError("code_str must be a string")

    if len(code_str.strip()) < 5:
        return {smell: 0 for smell in SMELL_NAMES}

    code = _normalize_code(code_str)
    lines = _split_lines(code)
    nonempty = _nonempty_lines(lines)

    smells: Dict[str, int] = {
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

# Constants for model names
_QUALITY_REGRESSOR_NAMES = [
    "ridge_regression",
    "decision_tree",
    "random_forest",
    "gradient_boosting",
    "xgboost",
    "svr",
]

_METRICS_REGRESSOR_NAMES = ["ridge", "random_forest", "gradient_boosting", "xgboost"]
_METRICS_PREFIXES = [("metrics_time", "_m_time"), ("metrics_tokens", "_m_tokens")]

def _load_models() -> None:
    """Load all machine‑learning models, scalers, encoders, and price data.

    This function populates several module‑level globals on first call.
    Subsequent calls are no‑ops.

    Raises:
        Any exception raised by ``_safe_joblib_load`` or file I/O is
        propagated to the caller.
    """
    global _models_loaded, _q_regressors, _q_scaler, _q_le, _q_stats, _q_meta
    global _m_time, _m_tokens, _m_scaler, _m_le, _m_stats, _m_meta, _prices

    if _models_loaded:
        return

    _load_quality_regressors()
    _load_quality_auxiliary()
    _load_metrics_regressors()
    _load_metrics_auxiliary()
    _load_price_data()

    _models_loaded = True


def _load_quality_regressors() -> None:
    """Load quality regression models into ``_q_regressors``."""
    for name in _QUALITY_REGRESSOR_NAMES:
        model_path = MODELS_DIR / f"quality_{name}.pkl"
        if not model_path.exists():
            continue
        loaded = _safe_joblib_load(model_path)
        if loaded is not None:
            _q_regressors[name] = loaded


def _load_quality_auxiliary() -> None:
    """Load quality scaler, label encoder, stats, and meta information."""
    global _q_scaler, _q_le, _q_stats, _q_meta

    _q_scaler = _safe_joblib_load(MODELS_DIR / "quality_scaler.pkl")
    _q_le = _safe_joblib_load(MODELS_DIR / "quality_label_encoder.pkl")
    _q_stats = _safe_joblib_load(MODELS_DIR / "quality_stats.pkl")
    _q_meta = _safe_joblib_load(MODELS_DIR / "quality_meta.pkl")


def _load_metrics_regressors() -> None:
    """Load metrics regression models into ``_m_time`` and ``_m_tokens``."""
    for name in _METRICS_REGRESSOR_NAMES:
        for dict_name, attr_name in _METRICS_PREFIXES:
            target_dict = globals()[attr_name]
            model_path = MODELS_DIR / f"{dict_name}_{name}.pkl"
            if not model_path.exists():
                continue
            loaded = _safe_joblib_load(model_path)
            if loaded is not None:
                target_dict[name] = loaded


def _load_metrics_auxiliary() -> None:
    """Load metrics scaler, label encoder, stats, and meta information."""
    global _m_scaler, _m_le, _m_stats, _m_meta

    _m_scaler = _safe_joblib_load(MODELS_DIR / "metrics_scaler.pkl")
    _m_le = _safe_joblib_load(MODELS_DIR / "metrics_label_encoder.pkl")
    _m_stats = _safe_joblib_load(MODELS_DIR / "metrics_stats.pkl")
    _m_meta = _safe_joblib_load(MODELS_DIR / "metrics_meta.pkl")


def _load_price_data() -> None:
    """Parse ``price.csv`` and populate the ``_prices`` dictionary."""
    global _prices

    price_path = SCRIPT_DIR / "price.csv"
    if not price_path.exists():
        return

    with open(str(price_path), newline="") as fh:
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

# Constants
LONG_METHOD_THRESHOLD = 15
DENOMINATOR_OFFSET = 1.0

def quality_features(
    original_code: str,
    refactored_code: str,
    function_name: str,
    model_encoding: int,
) -> np.ndarray:
    """Build a 50‑dimensional feature vector for quality prediction.

    Args:
        original_code: Source code of the original function.
        refactored_code: Source code after refactoring.
        function_name: Name of the examined function.
        model_encoding: Integer encoding of the model used for prediction.

    Returns:
        A ``np.ndarray`` with shape ``(1, 50)`` containing structural and
        smell‑related features as ``float`` values.

    Raises:
        Any exception raised by the helper utilities (e.g. ``count_lines``) is
        propagated unchanged.
    """
    structural = _build_structural_features(
        original_code,
        refactored_code,
        function_name,
        model_encoding,
    )
    smell = _build_smell_features(original_code, refactored_code)
    return np.array([structural + smell], dtype=float)


def _build_structural_features(
    original_code: str,
    refactored_code: str,
    function_name: str,
    model_encoding: int,
) -> list[float]:
    """Calculate structural features for the given function."""
    original_line_cnt = float(count_lines(original_code))
    refactored_line_cnt = float(count_lines(refactored_code))

    long_method_flag = int(original_line_cnt > LONG_METHOD_THRESHOLD)
    line_delta = original_line_cnt - refactored_line_cnt
    line_delta_ratio = line_delta / (original_line_cnt + DENOMINATOR_OFFSET)

    has_future_imp = has_future_import(refactored_code)
    has_doc = has_docstring(refactored_code)
    has_type = has_type_hints(refactored_code)

    refact_complexity = count_keywords(refactored_code)
    original_complexity = count_keywords(original_code)
    complexity_delta = original_complexity - refact_complexity

    name_length = len(function_name)
    is_private = int(function_name.startswith("_") and not function_name.startswith("__"))
    is_dunder = int(function_name.startswith("__") and function_name.endswith("__"))
    is_test = int(function_name.startswith("test"))

    repo_avg = float(_q_stats.get("global_repo_mean", original_line_cnt))
    file_avg = float(_q_stats.get("global_file_mean", original_line_cnt))
    global_max = float(_q_stats.get("global_max_lines", max(original_line_cnt, DENOMINATOR_OFFSET)))
    normalized_line_cnt = original_line_cnt / (global_max + DENOMINATOR_OFFSET)

    return [
        original_line_cnt,
        long_method_flag,
        refactored_line_cnt,
        line_delta,
        line_delta_ratio,
        has_future_imp,
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
        normalized_line_cnt,
        model_encoding,
    ]


def _build_smell_features(
    original_code: str,
    refactored_code: str,
) -> list[float]:
    """Calculate smell‑related features for the given pair of code snippets."""
    original_smells = detect_code_smells(original_code)
    refactored_smells = detect_code_smells(refactored_code)

    original_total = sum(original_smells.values())
    refactored_total = sum(refactored_smells.values())
    reduction = max(0, original_total - refactored_total)
    reduction_ratio = reduction / (original_total + DENOMINATOR_OFFSET)

    aggregation = [
        original_total,
        refactored_total,
        reduction,
        reduction_ratio,
    ]

    original_per_smell = [original_smells[name] for name in SMELL_NAMES]
    refactored_per_smell = [refactored_smells[name] for name in SMELL_NAMES]
    fixed_per_smell = [
        max(0, original_smells[name] - refactored_smells[name]) for name in SMELL_NAMES
    ]

    return aggregation + original_per_smell + refactored_per_smell + fixed_per_smell


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


from typing import Any, Dict, List
import numpy as np

DEFAULT_QUALITY_SCORE = 5.0
DEFAULT_PRED_TIME_MS = 2000
DEFAULT_PRED_TOKENS = 500


def _encode_label(
    label_encoder: Any, llm: str, fallback_index: int
) -> int:
    """Encode LLM name using a label encoder or fallback index."""
    try:
        if label_encoder is not None:
            return int(label_encoder.transform([llm])[0])
    except Exception:
        pass
    return fallback_index


def _predict_quality(
    orig_code: str,
    refact_proxy: str,
    func_name: str,
    q_enc: int,
    best_q_safe: str,
    best_q_name: str,
    q_regressors: Dict[str, Any],
    q_scaler: Any,
    q_scaled_models: List[str],
) -> float:
    """Predict quality score (0‑10) for a single LLM."""
    if not q_regressors or best_q_safe not in q_regressors:
        return DEFAULT_QUALITY_SCORE

    features = quality_features(orig_code, refact_proxy, func_name, q_enc)

    reg = q_regressors[best_q_safe]
    try:
        if q_scaler is not None and best_q_name in q_scaled_models:
            inp = q_scaler.transform(features)
        else:
            inp = features
        pred = reg.predict(inp)[0]
        return float(np.clip(pred, 0.0, 10.0))
    except Exception:
        return DEFAULT_QUALITY_SCORE


def _predict_metrics(
    orig_code: str,
    refact_proxy: str,
    func_name: str,
    m_enc: int,
    time_reg: Any,
    tokens_reg: Any,
    m_scaler: Any,
) -> tuple[int, int]:
    """Predict generation time (ms) and token count for a single LLM."""
    if time_reg is None or tokens_reg is None:
        return DEFAULT_PRED_TIME_MS, DEFAULT_PRED_TOKENS

    features = metrics_features(orig_code, refact_proxy, func_name, m_enc)

    try:
        if m_scaler is not None:
            features = m_scaler.transform(features)
        time_ms = int(np.expm1(time_reg.predict(features)[0]))
        tokens = int(np.expm1(tokens_reg.predict(features)[0]))
        return time_ms, tokens
    except Exception:
        return DEFAULT_PRED_TIME_MS, DEFAULT_PRED_TOKENS


def _price_per_token(llm: str, prices: Dict[str, Any]) -> float:
    """Resolve price per token for a given LLM."""
    try:
        price = float(prices.get(llm, 0.0))
        if price != 0.0:
            return price
        for alt, val in prices.items():
            alt_key = alt.lower().replace(" ", "_")
            if alt_key == llm or llm in alt_key or alt_key in llm:
                return float(val)
    except Exception:
        pass
    return 0.0


def _normalize_array(arr: np.ndarray) -> np.ndarray:
    """Scale array to [0, 1]; fallback to 0.5 when constant."""
    rng = arr.max() - arr.min()
    if rng > 0:
        return (arr - arr.min()) / rng
    return np.full_like(arr, 0.5, dtype=float)


def predict_all_llms(
    orig_code: str, func_name: str = "unknown_function"
) -> Dict[str, Dict[str, Any]]:
    """Predict quality, latency, token usage and cost for all supported LLMs.

    Args:
        orig_code: Original source code to be evaluated.
        func_name: Name of the function containing the code.

    Returns:
        Mapping from LLM name to a dictionary of metrics, including a
        composite score (0‑100).

    """
    _load_models()
    refact_proxy = orig_code

    models_ready = bool(_q_regressors) and _q_scaler is not None and _q_le is not None
    metrics_ready = bool(_m_time) and _m_scaler is not None and _m_le is not None

    best_q_name = _q_meta["best_model"] if models_ready and _q_meta else ""
    best_q_safe = best_q_name.lower().replace(" ", "_")
    q_scaled_models = (
        _q_stats.get("scaled_models", ["Ridge Regression", "SVR"])
        if _q_stats is not None
        else []
    )

    if metrics_ready and _m_meta is not None:
        best_time_name = _m_meta.get("best_time_model", "gradient_boosting").lower().replace(
            " ", "_"
        )
        best_tokens_name = _m_meta.get("best_tokens_model", "gradient_boosting").lower().replace(
            " ", "_"
        )
    else:
        best_time_name = ""
        best_tokens_name = ""

    results: Dict[str, Dict[str, Any]] = {}
    for llm in MODEL_ORDER:
        q_enc = _encode_label(_q_le, llm, MODEL_ORDER.index(llm))
        m_enc = _encode_label(_m_le, llm, MODEL_ORDER.index(llm))

        quality_score = _predict_quality(
            orig_code,
            refact_proxy,
            func_name,
            q_enc,
            best_q_safe,
            best_q_name,
            _q_regressors,
            _q_scaler,
            q_scaled_models,
        ) if models_ready else DEFAULT_QUALITY_SCORE

        time_reg = _m_time.get(best_time_name) if metrics_ready else None
        tokens_reg = _m_tokens.get(best_tokens_name) if metrics_ready else None
        pred_time_ms, pred_tokens = (
            _predict_metrics(
                orig_code,
                refact_proxy,
                func_name,
                m_enc,
                time_reg,
                tokens_reg,
                _m_scaler,
            )
            if metrics_ready
            else (DEFAULT_PRED_TIME_MS, DEFAULT_PRED_TOKENS)
        )

        price = _price_per_token(llm, _prices)
        results[llm] = {
            "quality_score": quality_score,
            "pred_time_ms": pred_time_ms,
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

    composite = (
        0.50 * _normalize_array(scores)
        + 0.20 * (1.0 - _normalize_array(costs))
        + 0.20 * (1.0 - _normalize_array(times))
        + 0.10 * (1.0 - _normalize_array(tokens))
    ) * 100.0

    for i, name in enumerate(llm_names):
        results[name]["composite"] = float(composite[i])

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
