import os
import sys
import joblib
from pathlib import Path
from scipy.sparse import hstack, csr_matrix

import os
import sys
from pathlib import Path
from typing import Any, List, Set, Tuple

import joblib
from scipy.sparse import csr_matrix, hstack

MODEL_PATH = Path("models/best_language_classifier.joblib")
VALID_EXTENSIONS = {".py", ".java", ".cs", ".js", ".php"}
PYTHON_FILES_LIST = "python_files.txt"


def _is_hidden_path(path: Path) -> bool:
    """Return True if any part of the path starts with a dot."""
    return any(part.startswith(".") for part in path.parts)


def _should_process_file(file_path: Path) -> bool:
    """Determine whether a file should be examined for language detection."""
    return (
        file_path.is_file()
        and not _is_hidden_path(file_path)
        and "models" not in file_path.parts
        and file_path.suffix.lower() in VALID_EXTENSIONS
    )


def _load_model_bundle(
    model_path: Path,
) -> Tuple[Any, Any, Any, List[str]]:
    """Load the ML model bundle and return its components.

    Args:
        model_path: Path to the serialized joblib bundle.

    Returns:
        A tuple containing (model, tfidf_char, tfidf_word, class_names).

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


def _predict_language(
    file_path: Path,
    tfidf_char: Any,
    tfidf_word: Any,
    model: Any,
    class_names: List[str],
) -> str | None:
    """Predict the programming language of a file.

    Returns:
        The predicted language name, or None if prediction fails.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    lines = len(content.splitlines())
    combined = f"{file_path.name.lower()}  {content}"

    x_char = tfidf_char.transform([combined])
    x_word = tfidf_word.transform([combined])
    x_lines = csr_matrix([[float(lines)]])

    if hasattr(model, "n_features_in_") and model.n_features_in_ == x_char.shape[1] + x_word.shape[1]:
        X = hstack([x_char, x_word])
    else:
        X = hstack([x_char, x_word, x_lines])

    try:
        pred_label = model.predict(X)[0]
        return class_names[pred_label]
    except Exception as e:
        print(f"Error predicting {file_path}: {e}")
        return None


def _write_python_file_list(python_files: List[str]) -> None:
    """Write the list of detected Python files to a text file."""
    with open(PYTHON_FILES_LIST, "w", encoding="utf-8") as pf:
        for path in python_files:
            pf.write(f"{path}\n")


def _append_github_output(lines: List[str]) -> None:
    """Append key‑value pairs to the GitHub Actions output file if set."""
    github_output = os.getenv("GITHUB_OUTPUT")
    if not github_output:
        return
    try:
        with open(github_output, "a", encoding="utf-8") as f:
            for line in lines:
                f.write(f"{line}\n")
    except Exception as e:
        print(f"Failed to write to GITHUB_OUTPUT: {e}")


def main() -> None:
    """Detect programming languages in the repository and decide further actions.

    The function loads a pre‑trained language classifier, scans source files,
    records detected languages, and writes decision flags for downstream CI steps.
    """
    try:
        model, tfidf_char, tfidf_word, class_names = _load_model_bundle(MODEL_PATH)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    python_detected = False
    other_language_detected = False
    detected_other_languages: Set[str] = set()
    python_files: List[str] = []

    print("Scanning files for language detection...")
    for file_path in Path(".").rglob("*"):
        if not _should_process_file(file_path):
            continue

        prediction = _predict_language(
            file_path, tfidf_char, tfidf_word, model, class_names
        )
        if prediction is None:
            continue

        if prediction.lower() == "python":
            python_detected = True
            python_files.append(str(file_path))
        else:
            other_language_detected = True
            detected_other_languages.add(str(prediction))
            print(f"Detected other language ({prediction}) in file: {file_path}")

    langs = ", ".join(detected_other_languages)

    if other_language_detected and python_detected:
        print(f"Mixed repo detected. Non-Python languages found: {langs}")
        print(f"Proceeding with {len(python_files)} Python file(s) only. Skipping: {langs}")
        _write_python_file_list(python_files)
        _append_github_output(
            [
                "proceed=true",
                "python_only_mode=true",
                f"skipped_languages={langs}",
            ]
        )
    elif other_language_detected:
        print(f"Non-Python language detected ({langs}). No Python files found. Stopping system.")
        _append_github_output(
            [
                "proceed=false",
                f"detected_languages={langs}",
            ]
        )
        sys.exit(1)
    elif python_detected:
        print("Only Python files detected. Proceeding...")
        _append_github_output(["proceed=true"])
    else:
        print("No readable source files found.")
        _append_github_output(["proceed=false"])


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
