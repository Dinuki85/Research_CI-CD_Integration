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
SEED: int = 42
PERTURBATION_ITERATIONS: int = 3
SHORT_LABEL: str = "No"
LONG_LABEL: str = "Yes"
SUCCESS_RATE_MULTIPLIER: int = 100
SLEEP_BASE: int = 2


def _short_templates() -> List[str]:
    """Return a list of short/clean function templates."""
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
    """Return a list of long/bad‑smell function templates."""
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
        "            time.sleep(SLEEP_BASE ** attempt)\n"
        "        except Exception as e:\n"
        "            failed.append({'batch': batch, 'error': str(e)})\n"
        "            break\n"
        "summary = {\n"
        "    'total': len(items),\n"
        "    'processed': len(results),\n"
        "    'failed': len(failed),\n"
        "    'success_rate': len(results) / len(items) * SUCCESS_RATE_MULTIPLIER,\n"
        "}\n"
        "return results, failed, summary",
    ]


def _extract_features(codes: List[str]) -> List[Dict]:
    """Extract feature dictionaries from a list of code strings."""
    return [extract_code_features(code) for code in codes]


def _perturb_short(template: str) -> str:
    """Add a harmless comment and a pass statement to a short template."""
    return f"{template}\n    # extra comment\n    pass"


def _extend_long(template: str) -> str:
    """Append logging and metric collection to a long template."""
    return (
        f"{template}\n"
        "    # additional logging\n"
        "    logger.info(f\"Processed batch\")\n"
        "    metrics.record(count=len(results))"
    )


def generate_synthetic_dataset() -> pd.DataFrame:
    """Generate a synthetic training dataset when ``codeworks/dataset.csv`` is missing.

    The function creates short (clean) and long (smelly) Python code snippets,
    extracts their features, and returns a :class:`pandas.DataFrame` suitable for
    model training.

    Returns:
        pandas.DataFrame: A dataframe where each row corresponds to a code snippet
        and includes a ``long_method`` column with labels ``\"Yes\"`` or ``\"No\"``.
    """
    print("  codeworks/dataset.csv not found — generating synthetic training data")
    rng = np.random.default_rng(SEED)  # noqa: F841 – retained for potential future use

    short_codes = _short_templates()
    long_codes = _long_templates()

    short_features = _extract_features(short_codes)
    short_labels = [SHORT_LABEL] * len(short_codes)

    long_features = _extract_features(long_codes)
    long_labels = [LONG_LABEL] * len(long_codes)

    # Add perturbed variations of short functions
    for _ in range(PERTURBATION_ITERATIONS):
        for template in _short_templates():
            perturbed = _perturb_short(template)
            short_codes.append(perturbed)
            short_features.append(extract_code_features(perturbed))
            short_labels.append(SHORT_LABEL)

    # Add extended variations of long functions
    for _ in range(PERTURBATION_ITERATIONS):
        for template in _long_templates():
            extended = _extend_long(template)
            long_codes.append(extended)
            long_features.append(extract_code_features(extended))
            long_labels.append(LONG_LABEL)

    all_features = short_features + long_features
    all_labels = short_labels + long_labels

    synthetic_df = pd.DataFrame(all_features)
    synthetic_df["long_method"] = all_labels

    print(
        f"  Generated {len(synthetic_df)} synthetic training examples "
        f"(No={all_labels.count(SHORT_LABEL)}, Yes={all_labels.count(LONG_LABEL)})"
    )
    return synthetic_df


import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from imblearn.over_sampling import SMOTE
import joblib

# Constants
SIGMA: float = 0.25
NUM_LINES_NOISE_STD: float = 12.0
N_NOISE_COPIES: int = 4
SEED: int = 42  # Example seed; replace with actual value if defined elsewhere

def load_dataset(
    csv_path: Path,
    safe_features: list[str],
) -> pd.DataFrame:
    """Load the dataset from CSV or generate a synthetic one.

    Args:
        csv_path: Path to the CSV file containing the raw data.
        safe_features: List of feature column names to retain.

    Returns:
        DataFrame with extracted features and target column ``long_method``.
    """
    if csv_path.exists():
        df_raw = pd.read_csv(str(csv_path), encoding="latin-1")
        df_raw = df_raw[
            ["code_section", "Approximate Number of Lines", "Long Method"]
        ].copy()
        df_raw.columns = ["code_section", "num_lines", "long_method"]
        print(f"\nLoaded {len(df_raw)} rows from {csv_path.name}")

        feats = df_raw["code_section"].apply(extract_code_features)
        df_all = pd.concat([feats, df_raw[["long_method"]]], axis=1)
    else:
        df_all = generate_synthetic_dataset()

    df_all = df_all[safe_features + ["long_method"]]
    return df_all


def encode_labels(df: pd.DataFrame) -> tuple[np.ndarray, LabelEncoder]:
    """Encode the ``long_method`` target column.

    Args:
        df: DataFrame containing a ``long_method`` column.

    Returns:
        Tuple of encoded labels array and the fitted ``LabelEncoder``.
    """
    le = LabelEncoder()
    y = le.fit_transform(df["long_method"])
    print(f"Class distribution: No={sum(y == 0)}, Yes={sum(y == 1)}")
    return y, le


def augment_training_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    binary_columns: list[int],
    num_lines_idx: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Create noisy copies of the training set and apply SMOTE.

    Args:
        X_train: Original training feature matrix.
        y_train: Original training labels.
        binary_columns: Indices of binary feature columns.
        num_lines_idx: Index of the ``num_lines`` feature column.
        rng: Random number generator.

    Returns:
        Tuple of augmented feature matrix and label array.
    """
    feature_stds = np.std(X_train, axis=0, ddof=0)
    noise_copies = []

    for _ in range(N_NOISE_COPIES):
        noise = rng.normal(
            loc=0.0,
            scale=SIGMA * feature_stds,
            size=X_train.shape,
        )
        noise[:, num_lines_idx] = rng.normal(
            0, NUM_LINES_NOISE_STD, size=X_train.shape[0]
        )
        X_noisy = X_train + noise
        X_noisy[:, binary_columns] = np.clip(
            np.round(X_noisy[:, binary_columns]), 0, 1
        )
        X_noisy = np.clip(X_noisy, 0, None)
        noise_copies.append(X_noisy)

    X_augmented = np.vstack([X_train] + noise_copies)
    y_augmented = np.concatenate([y_train] + [y_train.copy()] * N_NOISE_COPIES)

    smote = SMOTE(random_state=SEED, k_neighbors=5)
    X_resampled, y_resampled = smote.fit_resample(X_augmented, y_augmented)

    print(
        f"Augmented train: {X_resampled.shape}, "
        f"classes: {np.bincount(y_resampled)}"
    )
    return X_resampled, y_resampled


def train_svm(
    X: np.ndarray,
    y: np.ndarray,
) -> Pipeline:
    """Train an SVM classifier inside a preprocessing pipeline.

    Args:
        X: Feature matrix.
        y: Target labels.

    Returns:
        Fitted scikit‑learn ``Pipeline`` with scaling and SVM steps.
    """
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(C=1.0, probability=True, random_state=SEED)),
        ]
    )
    pipeline.fit(X, y)
    return pipeline


def cross_validate(
    pipeline: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
) -> None:
    """Perform stratified 5‑fold cross‑validation and print metrics.

    Args:
        pipeline: Trained pipeline to evaluate.
        X: Feature matrix.
        y: Target labels.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    acc_scores = cross_val_score(
        pipeline, X, y, cv=cv, scoring="accuracy"
    )
    f1_scores = cross_val_score(
        pipeline, X, y, cv=cv, scoring="f1"
    )
    print(f"\nCV Accuracy: {acc_scores.mean():.4f} ± {acc_scores.std():.4f}")
    print(f"CV F1 Score:  {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")


def evaluate_on_test(
    pipeline: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Evaluate the pipeline on the real test set and print accuracy.

    Args:
        pipeline: Trained pipeline.
        X_test: Test feature matrix.
        y_test: Test labels.
    """
    accuracy = pipeline.score(X_test, y_test)
    print(f"Test Accuracy (real data): {accuracy:.4f}")


def save_artifacts(
    pipeline: Pipeline,
    models_dir: Path,
) -> None:
    """Persist the trained SVM model and scaler to disk.

    Args:
        pipeline: Trained pipeline containing ``svm`` and ``scaler`` steps.
        models_dir: Directory where model files will be saved.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        pipeline.named_steps["svm"], str(models_dir / "smell_svm.pkl")
    )
    joblib.dump(
        pipeline.named_steps["scaler"], str(models_dir / "smell_scaler.pkl")
    )
    print(f"\n✅ Model saved to: {models_dir / 'smell_svm.pkl'}")
    print(f"✅ Scaler saved to: {models_dir / 'smell_scaler.pkl'}")


def main() -> None:
    """Orchestrate loading data, training, evaluating, and saving the model."""
    print("=" * 60)
    print("  Training Long-Method Code Smell Detector")
    print("=" * 60)

    df = load_dataset(CODEWORKS_CSV, SAFE_FEATURES)
    y, _ = encode_labels(df)

    X = df[SAFE_FEATURES].to_numpy()
    print(f"Feature matrix: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=SEED,
        stratify=y,
    )
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

    binary_idx = [SAFE_FEATURES.index(col) for col in BINARY_COLS]
    num_lines_idx = SAFE_FEATURES.index("num_lines")

    X_aug, y_aug = augment_training_data(
        X_train,
        y_train,
        binary_idx,
        num_lines_idx,
        rng,
    )

    svm_pipeline = train_svm(X_aug, y_aug)
    cross_validate(svm_pipeline, X_aug, y_aug)
    evaluate_on_test(svm_pipeline, X_test, y_test)
    save_artifacts(svm_pipeline, MODELS_DIR)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
