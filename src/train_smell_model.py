#!/usr/bin/env python3
"""
Train Smell Detection Model
============================
Trains an SVM classifier to detect long-method code smell in Python functions.

Replicates the approach from codeworks/model_synthetic.ipynb:
  1. Load dataset.csv from codeworks/
  2. Extract 12 safe features (avoids data leakage)
  3. Gaussian noise augmentation (4 copies, sigma=0.25, num_lines_noise=12)
  4. SMOTE oversampling for class balance
  5. Train SVM with standardization
  6. Save model and scaler to app/models/

Usage:
    python src/train_smell_model.py
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import joblib

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
CODEWORKS_CSV = Path(__file__).resolve().parent.parent.parent / "codeworks" / "dataset.csv"
MODELS_DIR    = Path(__file__).resolve().parent.parent / "models"

# ── Feature definitions (must match code_smell_detector.py) ─────────────────
SAFE_FEATURES = [
    "num_lines",
    "code_avg_line_len",
    "code_max_line_len",
    "code_indent_depth",
    "code_has_loop",
    "code_has_conditional",
    "code_has_return",
    "code_has_try",
    "code_num_returns",
    "code_num_ifs",
    "code_num_loops",
    "code_comment_lines",
]

BINARY_COLS = [
    "code_has_loop",
    "code_has_conditional",
    "code_has_return",
    "code_has_try",
]

SEED = 42
rng = np.random.default_rng(SEED)


def extract_code_features(code):
    """Same as code_smell_detector.extract_code_features."""
    code = str(code)
    lines = code.split("\n")
    nonempty = [l for l in lines if l.strip()]
    line_lengths = [len(l) for l in nonempty]

    return pd.Series({
        "num_lines":             len(nonempty),
        "code_avg_line_len":     float(np.mean(line_lengths)) if line_lengths else 0.0,
        "code_max_line_len":     float(max(line_lengths)) if line_lengths else 0.0,
        "code_indent_depth":     float(max(len(l) - len(l.lstrip()) for l in nonempty)) if nonempty else 0.0,
        "code_has_loop":         int(any(kw in code for kw in ["for ", "while "])),
        "code_has_conditional":  int(any(kw in code for kw in ["if ", "elif ", "else:"])),
        "code_has_return":       int("return" in code),
        "code_has_try":          int("try:" in code or "except" in code),
        "code_num_returns":      code.count("return"),
        "code_num_ifs":          code.count("if "),
        "code_num_loops":        code.count("for ") + code.count("while "),
        "code_comment_lines":    sum(1 for l in lines if l.strip().startswith("#")),
    })


from typing import List, Dict, Any

DEFAULT_PERTURBATIONS = 3
SYNTHETIC_DATA_MESSAGE = (
    "  codeworks/dataset.csv not found — generating synthetic training data"
)
GENERATED_EXAMPLE_MSG = (
    "  Generated {total} synthetic training examples (No={no}, Yes={yes})"
)
SHORT_PERTURB_SUFFIX = "\n    # extra comment\n    pass"
LONG_PERTURB_SUFFIX = (
    "\n    # additional logging\n"
    "    logger.info(f\"Processed batch\")\n"
    "    metrics.record(count=len(results))"
)


def _short_templates() -> List[str]:
    """Return a list of short (non‑smelly) code templates."""
    return [
        "\"\"\"Get the value.\"\"\"\nreturn self._value",
        "\"\"\"Set the name.\"\"\"\nself._name = name",
        "return sum(data) / len(data)",
        "return [x for x in items if x > 0]",
        "return a + b",
        "\"\"\"Check if valid.\"\"\"\nreturn value is not None and value > 0",
        "\"\"\"Format the string.\"\"\"\nreturn f\"{prefix}: {msg}\"",
        "result = []\nfor i in range(n):\n    result.append(i * 2)\nreturn result",
        "total = 0\nfor x in numbers:\n    total += x\nreturn total / len(numbers)",
        "count = 0\nfor item in collection:\n    if item.active:\n        count += 1\nreturn count",
        "def wrapper(func):\n    \"\"\"Simple decorator.\"\"\"\n    return func",
        "if condition:\n    return option_a\nreturn option_b",
        "if x < 0:\n    return -1\nelif x == 0:\n    return 0\nreturn 1",
        "try:\n    return int(value)\nexcept ValueError:\n    return 0",
    ]


def _long_templates() -> List[str]:
    """Return a list of long (smelly) code templates."""
    return [
        "\"\"\"Process data with full validation pipeline.\"\"\"\n"
        "errors = []\n"
        "results = []\n"
        "for idx, item in enumerate(data):\n"
        "    try:\n"
        "        if not isinstance(item, dict):\n"
        "            errors.append(f\"Item {idx} is not a dict\")\n"
        "            continue\n"
        "        required = ['id', 'name', 'value']\n"
        "        missing = [f for f in required if f not in item]\n"
        "        if missing:\n"
        "            errors.append(f\"Item {idx} missing: {missing}\")\n"
        "            continue\n"
        "        transformed = {\n"
        "            'id': item['id'],\n"
        "            'name': item['name'].strip(),\n"
        "            'value': float(item['value']),\n"
        "            'processed_at': datetime.now().isoformat(),\n"
        "        }\n"
        "        results.append(transformed)\n"
        "    except Exception as e:\n"
        "        errors.append(f\"Item {idx} error: {e}\")\n"
        "return {'results': results, 'errors': errors, 'count': len(results)}",
        "\"\"\"Parse and validate configuration from a file.\"\"\"\n"
        "config = {}\n"
        "with open(filepath, 'r') as f:\n"
        "    for line in f:\n"
        "        line = line.strip()\n"
        "        if not line or line.startswith('#'):\n"
        "            continue\n"
        "        if '=' not in line:\n"
        "            continue\n"
        "        key, value = line.split('=', 1)\n"
        "        key = key.strip()\n"
        "        value = value.strip()\n"
        "        if value.lower() in ('true', 'yes', '1'):\n"
        "            config[key] = True\n"
        "        elif value.lower() in ('false', 'no', '0'):\n"
        "            config[key] = False\n"
        "        else:\n"
        "            try:\n"
        "                config[key] = int(value)\n"
        "            except ValueError:\n"
        "                try:\n"
        "                    config[key] = float(value)\n"
        "                except ValueError:\n"
        "                    config[key] = value\n"
        "required = ['host', 'port', 'debug']\n"
        "missing = [k for k in required if k not in config]\n"
        "if missing:\n"
        "    raise ValueError(f\"Missing required config: {missing}\")\n"
        "config.setdefault('timeout', 30)\n"
        "config.setdefault('retries', 3)\n"
        "return config",
        "\"\"\"Process items in batches with retry logic.\"\"\"\n"
        "results = []\n"
        "failed = []\n"
        "for i in range(0, len(items), batch_size):\n"
        "    batch = items[i:i + batch_size]\n"
        "    for attempt in range(max_retries):\n"
        "        try:\n"
        "            batch_results = processor.process_batch(batch)\n"
        "            results.extend(batch_results)\n"
        "            break\n"
        "        except ConnectionError as e:\n"
        "            if attempt == max_retries - 1:\n"
        "                failed.extend(batch)\n"
        "            time.sleep(2 ** attempt)\n"
        "        except Exception as e:\n"
        "            failed.append({'batch': batch, 'error': str(e)})\n"
        "            break\n"
        "summary = {\n"
        "    'total': len(items),\n"
        "    'processed': len(results),\n"
        "    'failed': len(failed),\n"
        "    'success_rate': len(results) / len(items) * 100,\n"
        "}\n"
        "return results, failed, summary",
    ]


def _extract_features(codes: List[str]) -> List[Dict[str, Any]]:
    """Extract feature dictionaries from a list of source strings."""
    return [extract_code_features(code) for code in codes]


def generate_synthetic_dataset() -> pd.DataFrame:
    """Generate a synthetic training dataset when ``codeworks/dataset.csv`` is missing.

    The function creates short (non‑smelly) and long (smelly) Python code snippets,
    extracts their features, and returns a DataFrame suitable for model training.

    Returns:
        pd.DataFrame: DataFrame containing feature columns and a ``long_method``
            label column with values ``\"Yes\"`` (smelly) or ``\"No\"`` (clean).

    """
    print(SYNTHETIC_DATA_MESSAGE)
    rng = np.random.default_rng(SEED)  # retained for side‑effects compatibility

    short_templates = _short_templates()
    long_templates = _long_templates()

    # Base short examples
    short_codes = short_templates.copy()
    short_features = _extract_features(short_codes)
    short_labels = ["No"] * len(short_codes)

    # Base long examples
    long_codes = long_templates.copy()
    long_features = _extract_features(long_codes)
    long_labels = ["Yes"] * len(long_codes)

    # Perturb short templates
    for _ in range(DEFAULT_PERTURBATIONS):
        for tmpl in short_templates:
            padded = tmpl + SHORT_PERTURB_SUFFIX
            short_codes.append(padded)
            short_features.append(extract_code_features(padded))
            short_labels.append("No")

    # Extend long templates
    for _ in range(DEFAULT_PERTURBATIONS):
        for tmpl in long_templates:
            extended = tmpl + LONG_PERTURB_SUFFIX
            long_codes.append(extended)
            long_features.append(extract_code_features(extended))
            long_labels.append("Yes")

    all_features = short_features + long_features
    all_labels = short_labels + long_labels

    synthetic_df = pd.DataFrame(all_features)
    synthetic_df["long_method"] = all_labels

    print(
        GENERATED_EXAMPLE_MSG.format(
            total=len(synthetic_df),
            no=all_labels.count("No"),
            yes=all_labels.count("Yes"),
        )
    )
    return synthetic_df


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from imblearn.over_sampling import SMOTE
import joblib
from pathlib import Path

# Constants
HEADER_LINE = "=" * 60
SIGMA = 0.25
NUM_LINES_NOISE_STD = 12.0
N_NOISE_COPIES = 4
RANDOM_STATE = SEED
SMOTE_NEIGHBORS = 5
C_SVM = 1.0

def _print_header() -> None:
    """Print a formatted header for the CLI."""
    print(HEADER_LINE)
    print("  Training Long-Method Code Smell Detector")
    print(HEADER_LINE)


def _load_dataset() -> pd.DataFrame:
    """Load the dataset from CSV or generate a synthetic one.

    Returns:
        pd.DataFrame: Combined feature and label data.
    """
    if CODEWORKS_CSV.exists():
        df_raw = pd.read_csv(str(CODEWORKS_CSV), encoding="latin-1")
        df_raw = df_raw[["code_section", "Approximate Number of Lines", "Long Method"]].copy()
        df_raw.columns = ["code_section", "num_lines", "long_method"]
        print(f"\nLoaded {len(df_raw)} rows from dataset.csv")
        feats = df_raw["code_section"].apply(extract_code_features)
        return pd.concat([feats, df_raw[["long_method"]]], axis=1)
    return generate_synthetic_dataset()


def _prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode labels, extract feature matrix and split into train/test sets.

    Args:
        df: DataFrame containing features and the target column 'long_method'.

    Returns:
        X_train, X_test, y_train, y_test, feature_std, binary_idx, num_lines_idx.
    """
    le = LabelEncoder()
    y = le.fit_transform(df["long_method"])
    print(f"Class distribution: No={sum(y == 0)}, Yes={sum(y == 1)}")

    X = df[SAFE_FEATURES].copy()
    print(f"Feature matrix: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X.values,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

    feature_std = X.std()
    binary_idx = [SAFE_FEATURES.index(col) for col in BINARY_COLS]
    num_lines_idx = SAFE_FEATURES.index("num_lines")
    return X_train, X_test, y_train, y_test, feature_std, binary_idx, num_lines_idx


def _augment_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_std: pd.Series,
    binary_idx: list[int],
    num_lines_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create noisy copies of training data and apply SMOTE for balancing.

    Args:
        X_train: Original training feature matrix.
        y_train: Original training labels.
        feature_std: Standard deviation of each feature column.
        binary_idx: Indices of binary feature columns.
        num_lines_idx: Index of the `num_lines` feature.

    Returns:
        Tuple of augmented feature matrix and label array.
    """
    noise_copies = []
    label_copies = []

    for _ in range(N_NOISE_COPIES):
        noise = rng.normal(
            loc=0.0,
            scale=(SIGMA * feature_std).values,
            size=X_train.shape,
        )
        noise[:, num_lines_idx] = rng.normal(
            0, NUM_LINES_NOISE_STD, size=len(X_train)
        )
        X_noisy = X_train + noise
        X_noisy[:, binary_idx] = np.clip(np.round(X_noisy[:, binary_idx]), 0, 1)
        X_noisy = np.clip(X_noisy, 0, None)
        noise_copies.append(X_noisy)
        label_copies.append(y_train.copy())

    X_augmented = np.vstack([X_train] + noise_copies)
    y_augmented = np.concatenate([y_train] + label_copies)

    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=SMOTE_NEIGHBORS)
    X_resampled, y_resampled = smote.fit_resample(X_augmented, y_augmented)

    print(
        f"Augmented train: {X_resampled.shape}, classes: {np.bincount(y_resampled)}"
    )
    return X_resampled, y_resampled


def _train_pipeline(
    X: np.ndarray, y: np.ndarray
) -> Pipeline:
    """Fit a scaling + SVM pipeline on the provided data.

    Args:
        X: Feature matrix.
        y: Target vector.

    Returns:
        Trained scikit-learn Pipeline.
    """
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(C=C_SVM, probability=True, random_state=RANDOM_STATE)),
        ]
    )
    pipeline.fit(X, y)
    return pipeline


def _cross_validate(pipeline: Pipeline, X: np.ndarray, y: np.ndarray) -> None:
    """Perform 5‑fold stratified cross‑validation and print results."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    acc_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    f1_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1")
    print(f"\nCV Accuracy: {acc_scores.mean():.4f} ± {acc_scores.std():.4f}")
    print(f"CV F1 Score:  {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")


def _evaluate(pipeline: Pipeline, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """Evaluate the model on the real test set and print accuracy."""
    accuracy = pipeline.score(X_test, y_test)
    print(f"Test Accuracy (real data): {accuracy:.4f}")


def _save_model(pipeline: Pipeline) -> None:
    """Persist the trained SVM and scaler to disk."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline.named_steps["svm"], str(MODELS_DIR / "smell_svm.pkl"))
    joblib.dump(pipeline.named_steps["scaler"], str(MODELS_DIR / "smell_scaler.pkl"))
    print(f"\n✅ Model saved to: {MODELS_DIR / 'smell_svm.pkl'}")
    print(f"✅ Scaler saved to: {MODELS_DIR / 'smell_scaler.pkl'}")


def main() -> None:
    """Execute the full training pipeline for the Long‑Method code smell detector."""
    _print_header()
    dataset = _load_dataset()
    (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_std,
        binary_idx,
        num_lines_idx,
    ) = _prepare_features(dataset)

    X_aug, y_aug = _augment_data(
        X_train, y_train, feature_std, binary_idx, num_lines_idx
    )
    model_pipeline = _train_pipeline(X_aug, y_aug)
    _cross_validate(model_pipeline, X_aug, y_aug)
    _evaluate(model_pipeline, X_test, y_test)
    _save_model(model_pipeline)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
