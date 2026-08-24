import os
import sys
import joblib
from pathlib import Path
from scipy.sparse import hstack, csr_matrix

import os
import sys
from pathlib import Path
from typing import List, Set, Tuple, Dict, Any

import joblib
from scipy.sparse import csr_matrix, hstack

MODEL_REL_PATH = Path("models/best_language_classifier.joblib")
VALID_EXTENSIONS = {".py", ".java", ".cs", ".js", ".php"}
PYTHON_FILES_LIST = "python_files.txt"


def load_model_bundle(model_path: Path) -> Tuple[Any, Any, Any, List[str]]:
    """Load the language classifier bundle.

    Args:
        model_path: Path to the joblib file containing the model bundle.

    Returns:
        A tuple containing the model, character TF‑IDF transformer,
        word TF‑IDF transformer, and list of class names.

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


def is_valid_source_file(file_path: Path) -> bool:
    """Determine whether a file should be inspected.

    Skips hidden folders, any path containing a folder named 'models',
    and files without a supported extension.

    Args:
        file_path: Path to the file being evaluated.

    Returns:
        True if the file is a candidate for language detection.
    """
    if not file_path.is_file():
        return False
    if any(part.startswith(".") for part in file_path.parts):
        return False
    if "models" in file_path.parts:
        return False
    return file_path.suffix.lower() in VALID_EXTENSIONS


def predict_language(
    file_path: Path,
    tfidf_char: Any,
    tfidf_word: Any,
    model: Any,
    class_names: List[str],
) -> str:
    """Predict the programming language of a file.

    Args:
        file_path: Path to the source file.
        tfidf_char: Character‑level TF‑IDF transformer.
        tfidf_word: Word‑level TF‑IDF transformer.
        model: Trained classifier.
        class_names: Mapping from label indices to language names.

    Returns:
        The predicted language name.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    lines = len(content.splitlines())
    combined = f"{file_path.name.lower()}  {content}"

    x_char = tfidf_char.transform([combined])
    x_word = tfidf_word.transform([combined])
    x_lines = csr_matrix([[float(lines)]])

    if hasattr(model, "n_features_in_") and model.n_features_in_ == x_char.shape[1] + x_word.shape[1]:
        features = hstack([x_char, x_word])
    else:
        features = hstack([x_char, x_word, x_lines])

    pred_label = model.predict(features)[0]
    return str(class_names[pred_label])


def scan_repository(
    tfidf_char: Any,
    tfidf_word: Any,
    model: Any,
    class_names: List[str],
) -> Tuple[bool, bool, Set[str], List[str]]:
    """Traverse the repository and collect language detection results.

    Returns:
        A tuple containing:
            - python_detected: Whether any Python file was found.
            - other_language_detected: Whether any non‑Python language was found.
            - detected_other_languages: Set of other detected language names.
            - python_file_paths: List of paths to detected Python files.
    """
    python_detected = False
    other_language_detected = False
    detected_other_languages: Set[str] = set()
    python_file_paths: List[str] = []

    for path in Path(".").rglob("*"):
        if not is_valid_source_file(path):
            continue
        try:
            prediction = predict_language(path, tfidf_char, tfidf_word, model, class_names)
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue

        if prediction.lower() == "python":
            python_detected = True
            python_file_paths.append(str(path))
        else:
            other_language_detected = True
            detected_other_languages.add(prediction)
            print(f"Detected other language ({prediction}) in file: {path}")

    return python_detected, other_language_detected, detected_other_languages, python_file_paths


def write_python_file_list(python_file_paths: List[str]) -> None:
    """Write the list of Python file paths to a file for downstream steps."""
    with open(PYTHON_FILES_LIST, "w", encoding="utf-8") as f:
        for path in python_file_paths:
            f.write(f"{path}\n")


def append_to_github_output(lines: List[str]) -> None:
    """Append key‑value lines to the GitHub output file if the environment variable is set."""
    github_output = os.environ.get("GITHUB_OUTPUT")
    if not github_output:
        return
    with open(github_output, "a", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def main() -> None:
    """Entry point for language detection and pipeline control.

    Detects whether the repository contains Python files, other languages,
    or a mixture, and writes appropriate signals for the CI pipeline.
    """
    print("Loading language classifier bundle...")
    try:
        model, tfidf_char, tfidf_word, class_names = load_model_bundle(MODEL_REL_PATH)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    print("Scanning files for language detection...")
    python_detected, other_language_detected, other_languages, python_paths = scan_repository(
        tfidf_char, tfidf_word, model, class_names
    )

    other_langs_str = ", ".join(other_languages)

    if other_language_detected and python_detected:
        print(f"Mixed repo detected. Non-Python languages found: {other_langs_str}")
        print(f"Proceeding with {len(python_paths)} Python file(s) only. Skipping: {other_langs_str}")
        write_python_file_list(python_paths)
        append_to_github_output(
            [
                "proceed=true",
                "python_only_mode=true",
                f"skipped_languages={other_langs_str}",
            ]
        )
    elif other_language_detected:
        print(f"Non-Python language detected ({other_langs_str}). No Python files found. Stopping system.")
        append_to_github_output(
            [
                "proceed=false",
                f"detected_languages={other_langs_str}",
            ]
        )
        sys.exit(1)
    elif python_detected:
        print("Only Python files detected. Proceeding...")
        append_to_github_output(["proceed=true"])
    else:
        print("No readable source files found.")
        append_to_github_output(["proceed=false"])


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
