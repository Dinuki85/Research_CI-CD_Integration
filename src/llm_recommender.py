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
ALLOWED_MAGIC_NUMBERS = {"0", "1", "2", "3", "10", "100", "-1", "0.0", "1.0"}
MAX_ALLOWED_MAGIC_COUNT = 2
MAX_LINE_LENGTH = 79
ALLOWED_VARIABLES = {
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
COMMENTED_CODE_PATTERNS = (
    r"\s*#\s*(if |for |while |return |self\.|import |def |class |print\()"
)
TYPE_HINT_REGEX = re.compile(
    r"->\s*\w|:\s*(str|int|float|bool|None|list|dict|tuple|Any|Optional|List|Dict|Tuple)"
)


def _extract_nonempty_lines(source: str) -> List[str]:
    return [line for line in source.split("\n") if line.strip()]


def _has_long_method(nonempty_lines: List[str]) -> int:
    return int(len(nonempty_lines) > MAX_METHOD_LINES)


def _has_long_parameter_list(source: str) -> int:
    match = re.search(r"def\s+\w+\s*\(([^)]*)\)", source)
    if not match:
        return 0
    params = [
        p.strip()
        for p in match.group(1).split(",")
        if p.strip() and p.strip() not in ("self", "cls")
    ]
    return int(len(params) > MAX_PARAMS)


def _has_deep_nesting(nonempty_lines: List[str]) -> int:
    if not nonempty_lines:
        return 0
    max_indent = max(len(line) - len(line.lstrip()) for line in nonempty_lines)
    return int(max_indent >= MAX_NESTING_SPACES)


def _has_magic_numbers(source: str) -> int:
    numbers = re.findall(r"(?<!\w)(\d+\.?\d*)(?!\w)", source)
    magic = [n for n in numbers if n not in ALLOWED_MAGIC_NUMBERS]
    return int(len(magic) > MAX_ALLOWED_MAGIC_COUNT)


def _has_no_docstring(source: str) -> int:
    return int('"""' not in source and "'''" not in source)


def _has_no_type_hints(source: str) -> int:
    return int(not bool(TYPE_HINT_REGEX.search(source)))


def _has_long_lines(lines: List[str]) -> int:
    return int(any(len(line) > MAX_LINE_LENGTH for line in lines))


def _has_commented_code(lines: List[str]) -> int:
    commented = [line for line in lines if re.match(COMMENTED_CODE_PATTERNS, line)]
    return int(len(commented) >= 1)


def _has_poor_naming(source: str) -> int:
    poor = [
        var
        for var in re.findall(r"\b([a-zA-Z])\s*(?:\+|-|\*|\/)?=(?!=)", source)
        if var.lower() not in ALLOWED_VARIABLES
    ]
    return int(len(poor) > 1)


def detect_code_smells(code_str: str) -> Dict[str, int]:
    """Detect common code smells in a Python source string.

    Args:
        code_str: The source code to analyse.

    Returns:
        A dictionary mapping smell names to 0 (absent) or 1 (present).

    Raises:
        None.
    """
    empty_result = {smell: 0 for smell in SMELL_NAMES}
    if not isinstance(code_str, str) or len(code_str.strip()) < 5:
        return empty_result

    normalized = code_str.replace("\\n", "\n")
    all_lines = normalized.split("\n")
    nonempty_lines = _extract_nonempty_lines(normalized)

    smells: Dict[str, int] = {
        "long_method": _has_long_method(nonempty_lines),
        "long_param_list": _has_long_parameter_list(normalized),
        "deep_nesting": _has_deep_nesting(nonempty_lines),
        "magic_numbers": _has_magic_numbers(normalized),
        "no_docstring": _has_no_docstring(normalized),
        "no_type_hints": _has_no_type_hints(normalized),
        "long_lines": _has_long_lines(all_lines),
        "commented_code": _has_commented_code(all_lines),
        "poor_naming": _has_poor_naming(normalized),
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


from pathlib import Path
import csv
import re
from typing import Dict, Any, List

# Constants
QUALITY_REGRESSOR_NAMES: List[str] = [
    "ridge_regression",
    "decision_tree",
    "random_forest",
    "gradient_boosting",
    "xgboost",
    "svr",
]

METRICS_REGRESSOR_NAMES: List[str] = [
    "ridge",
    "random_forest",
    "gradient_boosting",
    "xgboost",
]

METRICS_PREFIXES: List[tuple[Dict[str, Any], str]] = []  # Filled during runtime


def _load_quality_regressors() -> None:
    """Load quality regressors and related artefacts."""
    for name in QUALITY_REGRESSOR_NAMES:
        fpath: Path = MODELS_DIR / f"quality_{name}.pkl"
        if fpath.exists():
            loaded = _safe_joblib_load(fpath)
            if loaded is not None:
                _q_regressors[name] = loaded

    global _q_scaler, _q_le, _q_stats, _q_meta
    _q_scaler = _safe_joblib_load(MODELS_DIR / "quality_scaler.pkl")
    _q_le = _safe_joblib_load(MODELS_DIR / "quality_label_encoder.pkl")
    _q_stats = _safe_joblib_load(MODELS_DIR / "quality_stats.pkl")
    _q_meta = _safe_joblib_load(MODELS_DIR / "quality_meta.pkl")


def _load_metrics_regressors() -> None:
    """Load metrics regressors and related artefacts."""
    for name in METRICS_REGRESSOR_NAMES:
        for container, prefix in [( _m_time, "metrics_time"),
                                  ( _m_tokens, "metrics_tokens")]:
            fpath: Path = MODELS_DIR / f"{prefix}_{name}.pkl"
            if fpath.exists():
                loaded = _safe_joblib_load(fpath)
                if loaded is not None:
                    container[name] = loaded

    global _m_scaler, _m_le, _m_stats, _m_meta
    _m_scaler = _safe_joblib_load(MODELS_DIR / "metrics_scaler.pkl")
    _m_le = _safe_joblib_load(MODELS_DIR / "metrics_label_encoder.pkl")
    _m_stats = _safe_joblib_load(MODELS_DIR / "metrics_stats.pkl")
    _m_meta = _safe_joblib_load(MODELS_DIR / "metrics_meta.pkl")


def _load_prices() -> None:
    """Load price information from CSV into the _prices dictionary."""
    prices_path: Path = SCRIPT_DIR / "price.csv"
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
    """Load all model artefacts once.

    This function populates several module‑level globals with the
    deserialized objects required for predictions. It is idempotent:
    subsequent calls have no effect after the first successful load.

    Raises:
        Any exception raised by the underlying I/O or deserialization
        functions will propagate to the caller.
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

LONG_METHOD_THRESHOLD = 15
DIV_EPSILON = 1.0


def _calculate_structural_features(
    orig_code: str,
    refact_code: str,
    func_name: str,
    model_enc: int,
    q_stats: dict,
) -> list[float]:
    """Compute structural features for a function.

    Args:
        orig_code: Original source code of the function.
        refact_code: Refactored source code of the function.
        func_name: Name of the function.
        model_enc: Integer encoding of the model identifier.
        q_stats: Dictionary with repository-wide statistics.

    Returns:
        List of 19 structural feature values.
    """
    orig_line_cnt = float(count_lines(orig_code))
    refact_line_cnt = float(count_lines(refact_code))
    long_method_flag = int(orig_line_cnt > LONG_METHOD_THRESHOLD)

    line_delta = orig_line_cnt - refact_line_cnt
    line_delta_ratio = line_delta / (orig_line_cnt + DIV_EPSILON)

    has_fut_imp = has_future_import(refact_code)
    has_doc = has_docstring(refact_code)
    has_type = has_type_hints(refact_code)

    complexity_ref = count_keywords(refact_code)
    complexity_orig = count_keywords(orig_code)
    complexity_delta = complexity_orig - complexity_ref

    func_name_len = len(func_name)
    is_private = int(func_name.startswith("_") and not func_name.startswith("__"))
    is_dunder = int(func_name.startswith("__") and func_name.endswith("__"))
    is_test = int(func_name.startswith("test"))

    repo_avg = float(q_stats.get("global_repo_mean", orig_line_cnt))
    file_avg = float(q_stats.get("global_file_mean", orig_line_cnt))
    global_max = float(q_stats.get("global_max_lines", max(orig_line_cnt, DIV_EPSILON)))
    norm_line_cnt = orig_line_cnt / (global_max + DIV_EPSILON)

    return [
        orig_line_cnt,
        long_method_flag,
        refact_line_cnt,
        line_delta,
        line_delta_ratio,
        has_fut_imp,
        has_doc,
        has_type,
        complexity_ref,
        complexity_orig,
        complexity_delta,
        func_name_len,
        is_private,
        is_dunder,
        is_test,
        repo_avg,
        file_avg,
        norm_line_cnt,
        model_enc,
    ]


def _calculate_smell_features(orig_code: str, refact_code: str) -> list[float]:
    """Compute smell‑related features for a function.

    Args:
        orig_code: Original source code of the function.
        refact_code: Refactored source code of the function.

    Returns:
        List of 31 smell feature values.
    """
    orig_smells = detect_code_smells(orig_code)
    refact_smells = detect_code_smells(refact_code)

    orig_total = sum(orig_smells.values())
    refact_total = sum(refact_smells.values())

    smell_reduction = max(0, orig_total - refact_total)
    smell_reduction_ratio = smell_reduction / (orig_total + DIV_EPSILON)

    agg_features = [
        orig_total,
        refact_total,
        smell_reduction,
        smell_reduction_ratio,
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

    Args:
        orig_code: Original source code of the target function.
        refact_code: Refactored source code of the target function.
        func_name: Name of the target function.
        model_enc: Integer identifier for the model encoding.

    Returns:
        A NumPy array of shape (1, 50) containing the feature vector as floats.
    """
    structural = _calculate_structural_features(
        orig_code, refact_code, func_name, model_enc, _q_stats
    )
    smells = _calculate_smell_features(orig_code, refact_code)
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
COMPOSITE_WEIGHTS = {
    "quality": 0.50,
    "cost": 0.20,
    "time": 0.20,
    "tokens": 0.10,
}
from typing import Any, Dict, List
import numpy as np


def _load_quality_score(
    orig_code: str,
    refact_proxy: str,
    func_name: str,
    q_enc: int,
    models_available: bool,
    best_q_name: str,
    best_q_safe: str,
    q_scaled_models: List[str],
) -> float:
    """Return the predicted quality score (0‑10) for a given LLM."""
    if not models_available or best_q_safe not in _q_regressors:
        return DEFAULT_QUALITY_SCORE

    q_features = quality_features(orig_code, refact_proxy, func_name, q_enc)
    reg = _q_regressors[best_q_safe]

    try:
        if _q_scaler is not None and best_q_name in q_scaled_models:
            q_input = _q_scaler.transform(q_features)
        else:
            q_input = q_features
        raw = reg.predict(q_input)[0]
        return float(np.clip(raw, 0.0, 10.0))
    except Exception:
        return DEFAULT_QUALITY_SCORE


def _load_metrics(
    orig_code: str,
    refact_proxy: str,
    func_name: str,
    m_enc: int,
    metrics_available: bool,
    best_time_name: str,
    best_tokens_name: str,
) -> tuple[int, int]:
    """Return predicted time (ms) and token count for a given LLM."""
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


def _lookup_price_per_token(llm: str) -> float:
    """Return the price per token for the given LLM name."""
    try:
        price = float(_prices.get(llm, DEFAULT_PRICE_PER_TOKEN))
        if price != 0.0:
            return price
        for k, v in _prices.items():
            key_norm = k.lower().replace(" ", "_")
            if key_norm == llm or llm in key_norm or key_norm in llm:
                return float(v)
        return DEFAULT_PRICE_PER_TOKEN
    except Exception:
        return DEFAULT_PRICE_PER_TOKEN


def _normalize_array(arr: np.ndarray) -> np.ndarray:
    """Return a min‑max normalised copy of the array (range 0‑1)."""
    diff = arr.max() - arr.min()
    if diff > 0:
        return (arr - arr.min()) / diff
    return np.full_like(arr, 0.5, dtype=float)


def predict_all_llms(orig_code: str, func_name: str = "unknown_function") -> Dict[str, Dict[str, Any]]:
    """Predict quality, latency, token usage and cost for each LLM.

    Args:
        orig_code: Original source code to analyse.
        func_name: Name of the function being refactored (default
            ``"unknown_function"``).

    Returns:
        Mapping from LLM name to a dictionary containing keys:
        ``quality_score``, ``pred_time_ms``, ``pred_tokens``,
        ``pred_cost``, ``price_per_token``, ``openrouter_slug``,
        ``display_name`` and ``composite`` (overall score).

    Raises:
        Any exception raised inside model loading or feature extraction
        propagates as a runtime error.
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
        try:
            q_enc = int(_q_le.transform([llm])[0]) if _q_le is not None else MODEL_ORDER.index(llm)
        except Exception:
            q_enc = MODEL_ORDER.index(llm)

        try:
            m_enc = int(_m_le.transform([llm])[0]) if _m_le is not None else MODEL_ORDER.index(llm)
        except Exception:
            m_enc = MODEL_ORDER.index(llm)

        quality_score = _load_quality_score(
            orig_code,
            refact_proxy,
            func_name,
            q_enc,
            models_available,
            best_q_name,
            best_q_safe,
            q_scaled_models,
        )

        pred_time_ms, pred_tokens = _load_metrics(
            orig_code,
            refact_proxy,
            func_name,
            m_enc,
            metrics_available,
            best_time_name,
            best_tokens_name,
        )

        price_per_token = _lookup_price_per_token(llm)

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
    scores = np.array([results[l]["quality_score"] for l in llm_names], dtype=float)
    times = np.array([results[l]["pred_time_ms"] for l in llm_names], dtype=float)
    tokens = np.array([results[l]["pred_tokens"] for l in llm_names], dtype=float)
    costs = np.array([results[l]["pred_cost"] for l in llm_names], dtype=float)

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
