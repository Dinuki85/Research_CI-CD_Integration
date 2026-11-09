import os
import sys
import joblib
from pathlib import Path
from scipy.sparse import hstack, csr_matrix

import os
import sys
from pathlib import Path
from typing import List, Set, Tuple, Optional

import joblib
from scipy.sparse import csr_matrix, hstack

MODEL_PATH = Path("models/best_language_classifier.joblib")
VALID_EXTENSIONS = {".py", ".java", ".cs", ".js", ".php"}
GITHUB_OUTPUT_ENV = "GITHUB_OUTPUT"


def load_model_bundle(
    model_path: Path,
) -> Tuple[object, object, object, List[str]]:
    """Load the language‑classifier bundle from *model_path*.

    Args:
        model_path: Path to the ``.joblib`` file containing the model.

    Returns:
        A tuple ``(model, tfidf_char, tfidf_word, class_names)``.
    """
    bundle = joblib.load(model_path)
    return (
        bundle["model"],
        bundle["tfidf_char"],
        bundle["tfidf_word"],
        bundle["class_names"],
    )


def is_valid_source_file(file_path: Path) -> bool:
    """Determine whether *file_path* should be processed.

    Skips hidden directories, the ``models`` directory and files with
    extensions not listed in :data:`VALID_EXTENSIONS`.

    Args:
        file_path: Path to evaluate.

    Returns:
        ``True`` if the file is a candidate for language detection.
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
    content: str,
    model: object,
    tfidf_char: object,
    tfidf_word: object,
    class_names: List[str],
) -> Tuple[bool, str]:
    """Predict the programming language of *content*.

    Args:
        file_path: Path of the source file (used for filename feature).
        content: File content.
        model: Trained classifier.
        tfidf_char: Character‑level TF‑IDF transformer.
        tfidf_word: Word‑level TF‑IDF transformer.
        class_names: Mapping from label indices to language names.

    Returns:
        ``(is_python, prediction)`` where *is_python* is ``True`` if the
        predicted language is Python.
    """
    lines = len(content.splitlines())
    combined = f"{file_path.name.lower()}  {content}"

    x_char = tfidf_char.transform([combined])
    x_word = tfidf_word.transform([combined])
    x_lines = csr_matrix([[float(lines)]])

    if hasattr(model, "n_features_in_") and model.n_features_in_ == x_char.shape[1] + x_word.shape[1]:
        X = hstack([x_char, x_word])
    else:
        X = hstack([x_char, x_word, x_lines])

    pred_label = model.predict(X)[0]
    prediction = class_names[pred_label]
    is_python = str(prediction).lower() == "python"
    return is_python, prediction


def write_python_file_list(python_files: List[str], output_path: Path = Path("python_files.txt")) -> None:
    """Write *python_files* to *output_path*, one per line."""
    with output_path.open("w", encoding="utf-8") as pf:
        for path in python_files:
            pf.write(f"{path}\n")


def append_to_github_output(lines: List[str]) -> None:
    """Append *lines* to the file pointed to by the ``GITHUB_OUTPUT`` env var."""
    github_output = os.environ.get(GITHUB_OUTPUT_ENV)
    if not github_output:
        return
    with open(github_output, "a", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def main() -> None:
    """Entry point for language detection.

    Scans the repository, determines whether Python files are present,
    and writes appropriate flags for downstream GitHub Actions steps.
    """
    if not MODEL_PATH.exists():
        print(f"Model not found at {MODEL_PATH}")
        sys.exit(1)

    print("Loading language classifier bundle...")
    model, tfidf_char, tfidf_word, class_names = load_model_bundle(MODEL_PATH)

    python_detected = False
    other_language_detected = False
    detected_other_languages: Set[str] = set()
    python_files: List[str] = []

    print("Scanning files for language detection...")
    for file_path in Path(".").rglob("*"):
        if not is_valid_source_file(file_path):
            continue
        try:
            with file_path.open("r", encoding="utf-8") as f:
                content = f.read()
            is_python, prediction = predict_language(
                file_path, content, model, tfidf_char, tfidf_word, class_names
            )
            if is_python:
                python_detected = True
                python_files.append(str(file_path))
            else:
                other_language_detected = True
                detected_other_languages.add(str(prediction))
                print(f"Detected other language ({prediction}) in file: {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    langs = ", ".join(detected_other_languages)

    if other_language_detected and python_detected:
        print(f"Mixed repo detected. Non-Python languages found: {langs}")
        print(f"Proceeding with {len(python_files)} Python file(s) only. Skipping: {langs}")
        write_python_file_list(python_files)
        append_to_github_output(
            [
                "proceed=true",
                "python_only_mode=true",
                f"skipped_languages={langs}",
            ]
        )
    elif other_language_detected:
        print(f"Non-Python language detected ({langs}). No Python files found. Stopping system.")
        append_to_github_output(
            [
                "proceed=false",
                f"detected_languages={langs}",
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
