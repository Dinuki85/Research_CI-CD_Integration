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


import numpy as np
import pandas as pd
from datetime import datetime
import time
from typing import List, Dict

# Constants
NUM_PERTURBATIONS: int = 3
SHORT_LABEL: str = "No"
LONG_LABEL: str = "Yes"


def _short_templates() -> List[str]:
    """Return a list of short‑function code snippets."""
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
    """Return a list of long‑function code snippets."""
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


def _extract_features(codes: List[str]) -> List[Dict]:
    """Extract feature dictionaries from a list of code strings."""
    return [extract_code_features(code) for code in codes]


def _perturb_short(base_templates: List[str]) -> List[tuple]:
    """Create perturbed short‑function variants."""
    perturbed = []
    for _ in range(NUM_PERTURBATIONS):
        for tmpl in base_templates:
            padded = f"{tmpl}\n    # extra comment\n    pass"
            features = extract_code_features(padded)
            perturbed.append((padded, features, SHORT_LABEL))
    return perturbed


def _perturb_long(base_templates: List[str]) -> List[tuple]:
    """Create extended long‑function variants."""
    perturbed = []
    for _ in range(NUM_PERTURBATIONS):
        for tmpl in base_templates:
            extended = (
                f"{tmpl}\n"
                "    # additional logging\n"
                "    logger.info(f\"Processed batch\")\n"
                "    metrics.record(count=len(results))"
            )
            features = extract_code_features(extended)
            perturbed.append((extended, features, LONG_LABEL))
    return perturbed


def generate_synthetic_dataset() -> pd.DataFrame:
    """Generate a synthetic training dataset when ``codeworks/dataset.csv`` is missing.

    The dataset contains short (label ``No``) and long (label ``Yes``) function
    snippets together with extracted code features.

    Returns:
        pandas.DataFrame: DataFrame with feature columns and a ``long_method``
        column containing the labels.

    Raises:
        RuntimeError: If feature extraction fails for any snippet.
    """
    print(
        "  codeworks/dataset.csv not found — generating synthetic training data"
    )
    rng = np.random.default_rng(SEED)  # noqa: F841 – kept for compatibility

    short_codes = _short_templates()
    long_codes = _long_templates()

    short_features = _extract_features(short_codes)
    long_features = _extract_features(long_codes)

    short_data = list(zip(short_codes, short_features, [SHORT_LABEL] * len(short_codes)))
    long_data = list(zip(long_codes, long_features, [LONG_LABEL] * len(long_codes)))

    short_data.extend(_perturb_short(_short_templates()))
    long_data.extend(_perturb_long(_long_templates()))

    all_features = [feat for _, feat, _ in short_data + long_data]
    all_labels = [label for _, _, label in short_data + long_data]

    synthetic_df = pd.DataFrame(all_features)
    synthetic_df["long_method"] = all_labels

    print(
        f"  Generated {len(synthetic_df)} synthetic training examples "
        f"(No={all_labels.count(SHORT_LABEL)}, "
        f"Yes={all_labels.count(LONG_LABEL)})"
    )
    return synthetic_df


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import joblib
from pathlib import Path
from typing import Tuple, List

# Magic numbers replaced with named constants
SIGMA: float = 0.25
NUM_LINES_NOISE_STD: float = 12.0
N_NOISE_COPIES: int = 4

def _print_header() -> None:
    """Print the script header."""
    print("=" * 60)
    print("  Training Long-Method Code Smell Detector")
    print("=" * 60)


def _load_dataset() -> pd.DataFrame:
    """
    Load the dataset from CSV if available; otherwise generate a synthetic dataset.

    Returns:
        pd.DataFrame: Combined dataframe with features and target.
    """
    if CODEWORKS_CSV.exists():
        raw_df = pd.read_csv(str(CODEWORKS_CSV), encoding="latin-1")
        raw_df = raw_df[["code_section", "Approximate Number of Lines", "Long Method"]].copy()
        raw_df.columns = ["code_section", "num_lines", "long_method"]
        print(f"\nLoaded {len(raw_df)} rows from dataset.csv")
        features = raw_df["code_section"].apply(extract_code_features)
        combined_df = pd.concat([features, raw_df[["long_method"]]], axis=1)
    else:
        combined_df = generate_synthetic_dataset()
    return combined_df


def _encode_labels(df: pd.DataFrame) -> Tuple[np.ndarray, LabelEncoder]:
    """
    Encode the target labels.

    Args:
        df: Dataframe containing the 'long_method' column.

    Returns:
        Tuple of encoded label array and the fitted LabelEncoder.
    """
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["long_method"])
    print(f"Class distribution: No={sum(y==0)}, Yes={sum(y==1)}")
    return y, label_encoder


def _prepare_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Select safe features and return as a NumPy array.

    Args:
        df: Dataframe containing feature columns.

    Returns:
        Feature matrix as NumPy array.
    """
    X = df[SAFE_FEATURES].copy()
    print(f"Feature matrix: {X.shape}")
    return X.values


def _augment_training_data(
    X_train: np.ndarray, y_train: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment training data with Gaussian noise and apply SMOTE for balancing.

    Args:
        X_train: Original training feature matrix.
        y_train: Original training labels.

    Returns:
        Tuple of augmented feature matrix and labels after SMOTE.
    """
    feature_stds = pd.DataFrame(X_train, columns=SAFE_FEATURES).std()
    binary_feature_indices = [SAFE_FEATURES.index(col) for col in BINARY_COLS]
    num_lines_index = SAFE_FEATURES.index("num_lines")

    noise_copies: List[np.ndarray] = []
    label_copies: List[np.ndarray] = []

    for _ in range(N_NOISE_COPIES):
        noise = rng.normal(
            loc=0.0,
            scale=(SIGMA * feature_stds).values,
            size=X_train.shape,
        )
        noise[:, num_lines_index] = rng.normal(
            0, NUM_LINES_NOISE_STD, size=len(X_train)
        )
        X_noisy = X_train + noise
        X_noisy[:, binary_feature_indices] = np.clip(
            np.round(X_noisy[:, binary_feature_indices]), 0, 1
        )
        X_noisy = np.clip(X_noisy, 0, None)
        noise_copies.append(X_noisy)
        label_copies.append(y_train.copy())

    X_augmented = np.vstack([X_train] + noise_copies)
    y_augmented = np.concatenate([y_train] + label_copies)

    smote = SMOTE(random_state=SEED, k_neighbors=5)
    X_resampled, y_resampled = smote.fit_resample(X_augmented, y_augmented)

    print(
        f"Augmented train: {X_resampled.shape}, classes: {np.bincount(y_resampled)}"
    )
    return X_resampled, y_resampled


def _train_pipeline(
    X_train: np.ndarray, y_train: np.ndarray
) -> Pipeline:
    """
    Train an SVM pipeline with standard scaling.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.

    Returns:
        Fitted scikit-learn Pipeline.
    """
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(C=1.0, probability=True, random_state=SEED)),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def _cross_validate(pipeline: Pipeline, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform stratified 5‑fold cross‑validation.

    Args:
        pipeline: Trained pipeline to evaluate.
        X: Feature matrix.
        y: Labels.

    Returns:
        Tuple containing accuracy scores and F1 scores arrays.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    acc_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    f1_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1")
    print(f"\nCV Accuracy: {acc_scores.mean():.4f} ± {acc_scores.std():.4f}")
    print(f"CV F1 Score:  {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")
    return acc_scores, f1_scores


def _evaluate(pipeline: Pipeline, X_test: np.ndarray, y_test: np.ndarray) -> float:
    """
    Evaluate the pipeline on a real test set.

    Args:
        pipeline: Trained pipeline.
        X_test: Test feature matrix.
        y_test: Test labels.

    Returns:
        Test accuracy.
    """
    accuracy = pipeline.score(X_test, y_test)
    print(f"Test Accuracy (real data): {accuracy:.4f}")
    return accuracy


def _save_artifacts(pipeline: Pipeline) -> None:
    """
    Save the trained SVM model and scaler to disk.

    Args:
        pipeline: Trained pipeline containing 'svm' and 'scaler' steps.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline.named_steps["svm"], str(MODELS_DIR / "smell_svm.pkl"))
    joblib.dump(pipeline.named_steps["scaler"], str(MODELS_DIR / "smell_scaler.pkl"))
    print(f"\n✅ Model saved to: {MODELS_DIR / 'smell_svm.pkl'}")
    print(f"✅ Scaler saved to: {MODELS_DIR / 'smell_scaler.pkl'}")


def main() -> None:
    """
    Execute the full training pipeline for the Long‑Method code smell detector.
    """
    _print_header()

    all_df = _load_dataset()
    y, _ = _encode_labels(all_df)
    X = _prepare_feature_matrix(all_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=SEED,
        stratify=y,
    )
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

    X_aug, y_aug = _augment_training_data(X_train, y_train)
    pipeline = _train_pipeline(X_aug, y_aug)

    _cross_validate(pipeline, X_aug, y_aug)
    _evaluate(pipeline, X_test, y_test)
    _save_artifacts(pipeline)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
