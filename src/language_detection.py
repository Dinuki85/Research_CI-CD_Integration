import os
import sys
import joblib
from pathlib import Path
from scipy.sparse import hstack, csr_matrix

import os
import sys
from pathlib import Path
from typing import Any, List, Set, Tuple, Optional

import joblib
from scipy.sparse import csr_matrix, hstack

MODEL_PATH: Path = Path("models/best_language_classifier.joblib")
VALID_EXTENSIONS: Set[str] = {".py", ".java", ".cs", ".js", ".php"}
GITHUB_OUTPUT_ENV: str = "GITHUB_OUTPUT"
PYTHON_FILES_OUTPUT: str = "python_files.txt"


def load_model(model_path: Path) -> Tuple[Any, Any, Any, List[str]]:
    """Load the language classifier model bundle.

    Args:
        model_path: Path to the joblib file containing the model bundle.

    Returns:
        A tuple containing the trained model, character-level TF‑IDF vectorizer,
        word-level TF‑IDF vectorizer, and the list of class names.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    bundle = joblib.load(model_path)
    return (
        bundle["model"],
        bundle["tfidf_char"],
        bundle["tfidf_word"],
        bundle["class_names"],
    )


def predict_language(
    file_path: Path,
    model: Any,
    tfidf_char: Any,
    tfidf_word: Any,
    class_names: List[str],
) -> Optional[str]:
    """Predict the programming language of a single file.

    Args:
        file_path: Path to the source file.
        model: Trained classifier.
        tfidf_char: Character‑level TF‑IDF transformer.
        tfidf_word: Word‑level TF‑IDF transformer.
        class_names: Mapping from numeric labels to language names.

    Returns:
        Predicted language name, or None if an error occurs.
    """
    try:
        with file_path.open("r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return None

    lines = len(content.splitlines())
    combined = f"{file_path.name.lower()}  {content}"

    x_char = tfidf_char.transform([combined])
    x_word = tfidf_word.transform([combined])
    x_lines = csr_matrix([[float(lines)]])

    if hasattr(model, "n_features_in_") and model.n_features_in_ == x_char.shape[1] + x_word.shape[1]:
        features = hstack([x_char, x_word])
    else:
        features = hstack([x_char, x_word, x_lines])

    try:
        pred_label = model.predict(features)[0]
        return class_names[pred_label]
    except Exception:
        return None


def write_python_file_list(python_files: List[str], output_path: str = PYTHON_FILES_OUTPUT) -> None:
    """Write the list of detected Python files to a text file.

    Args:
        python_files: List of file paths as strings.
        output_path: Destination file path.
    """
    with open(output_path, "w", encoding="utf-8") as pf:
        for path in python_files:
            pf.write(f"{path}\n")


def append_to_github_output(github_output: Optional[str], **kv_pairs: str) -> None:
    """Append key‑value pairs to the GitHub Actions output file.

    Args:
        github_output: Path to the GITHUB_OUTPUT file, or None.
        **kv_pairs: Arbitrary keyword arguments to write as `key=value` lines.
    """
    if not github_output:
        return
    with open(github_output, "a", encoding="utf-8") as f:
        for key, value in kv_pairs.items():
            f.write(f"{key}={value}\n")


def scan_repository(
    model: Any,
    tfidf_char: Any,
    tfidf_word: Any,
    class_names: List[str],
) -> Tuple[bool, bool, Set[str], List[str]]:
    """Traverse the repository and detect languages.

    Returns:
        A tuple containing:
            - python_detected (bool)
            - other_language_detected (bool)
            - set of other language names
            - list of Python file paths
    """
    python_detected = False
    other_language_detected = False
    other_languages: Set[str] = set()
    python_files: List[str] = []

    for file_path in Path(".").rglob("*"):
        if not file_path.is_file():
            continue
        if any(part.startswith(".") for part in file_path.parts):
            continue
        if "models" in file_path.parts:
            continue
        if file_path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        prediction = predict_language(file_path, model, tfidf_char, tfidf_word, class_names)
        if prediction is None:
            continue

        if prediction.lower() == "python":
            python_detected = True
            python_files.append(str(file_path))
        else:
            other_language_detected = True
            other_languages.add(str(prediction))
            print(f"Detected other language ({prediction}) in file: {file_path}")

    return python_detected, other_language_detected, other_languages, python_files


def main() -> None:
    """Entry point for language detection and GitHub Actions output handling."""
    try:
        model, tfidf_char, tfidf_word, class_names = load_model(MODEL_PATH)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    print("Scanning files for language detection...")
    python_detected, other_language_detected, other_languages, python_files = scan_repository(
        model, tfidf_char, tfidf_word, class_names
    )

    github_output_path = os.environ.get(GITHUB_OUTPUT_ENV)
    other_langs_str = ", ".join(other_languages)

    if other_language_detected and python_detected:
        print(f"Mixed repo detected. Non-Python languages found: {other_langs_str}")
        print(f"Proceeding with {len(python_files)} Python file(s) only. Skipping: {other_langs_str}")
        write_python_file_list(python_files)
        append_to_github_output(
            github_output_path,
            proceed="true",
            python_only_mode="true",
            skipped_languages=other_langs_str,
        )
    elif other_language_detected:
        print(f"Non-Python language detected ({other_langs_str}). No Python files found. Stopping system.")
        append_to_github_output(
            github_output_path,
            proceed="false",
            detected_languages=other_langs_str,
        )
        sys.exit(1)
    elif python_detected:
        print("Only Python files detected. Proceeding...")
        append_to_github_output(github_output_path, proceed="true")
    else:
        print("No readable source files found.")
        append_to_github_output(github_output_path, proceed="false")


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
