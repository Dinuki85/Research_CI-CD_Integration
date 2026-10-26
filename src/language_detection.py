import os
import sys
import joblib
from pathlib import Path
from scipy.sparse import hstack, csr_matrix

import os
import sys
from pathlib import Path
from typing import List, Set, Tuple

import joblib
from scipy.sparse import csr_matrix, hstack

MODEL_RELATIVE_PATH = Path("models/best_language_classifier.joblib")
VALID_EXTENSIONS = {".py", ".java", ".cs", ".js", ".php"}
PYTHON_ONLY_MODE = "python_only_mode"
PROCEED_KEY = "proceed"
SKIPPED_LANGUAGES_KEY = "skipped_languages"
DETECTED_LANGUAGES_KEY = "detected_languages"


def _load_model_bundle(model_path: Path) -> Tuple[object, object, object, List[str]]:
    """Load the ML model bundle from the given path.

    Args:
        model_path: Path to the joblib file containing the model bundle.

    Returns:
        A tuple (model, tfidf_char, tfidf_word, class_names).

    Raises:
        SystemExit: If the model file does not exist.
    """
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        sys.exit(1)

    print("Loading language classifier bundle...")
    bundle = joblib.load(model_path)
    return (
        bundle["model"],
        bundle["tfidf_char"],
        bundle["tfidf_word"],
        bundle["class_names"],
    )


def _is_valid_source_file(file_path: Path) -> bool:
    """Determine whether a file should be processed.

    Args:
        file_path: Path object to evaluate.

    Returns:
        True if the file is a regular file, not hidden, not inside a 'models'
        directory, and has a valid source-code extension.
    """
    if not file_path.is_file():
        return False
    if any(part.startswith(".") for part in file_path.parts):
        return False
    if "models" in file_path.parts:
        return False
    return file_path.suffix.lower() in VALID_EXTENSIONS


def _predict_language(
    file_path: Path,
    content: str,
    model: object,
    tfidf_char,
    tfidf_word,
    class_names: List[str],
) -> str:
    """Predict the programming language for a given file's content.

    Args:
        file_path: Path of the processed file.
        content: File content as a string.
        model: Trained classification model.
        tfidf_char: Character-level TF‑IDF transformer.
        tfidf_word: Word-level TF‑IDF transformer.
        class_names: List mapping numeric predictions to language names.

    Returns:
        Predicted language name.
    """
    line_count = len(content.splitlines())
    combined = f"{file_path.name.lower()}  {content}"

    x_char = tfidf_char.transform([combined])
    x_word = tfidf_word.transform([combined])
    x_lines = csr_matrix([[float(line_count)]])

    if hasattr(model, "n_features_in_") and model.n_features_in_ == x_char.shape[1] + x_word.shape[1]:
        features = hstack([x_char, x_word])
    else:
        features = hstack([x_char, x_word, x_lines])

    pred_index = model.predict(features)[0]
    return class_names[pred_index]


def _write_python_file_list(python_files: List[str]) -> None:
    """Write the list of detected Python files to 'python_files.txt'."""
    with open("python_files.txt", "w", encoding="utf-8") as pf:
        for path in python_files:
            pf.write(f"{path}\n")


def _append_to_github_output(lines: List[str]) -> None:
    """Append given lines to the file referenced by the GITHUB_OUTPUT env var."""
    github_output = os.environ.get("GITHUB_OUTPUT")
    if not github_output:
        return
    with open(github_output, "a", encoding="utf-8") as out_file:
        for line in lines:
            out_file.write(f"{line}\n")


def _process_repository(
    model: object,
    tfidf_char,
    tfidf_word,
    class_names: List[str],
) -> Tuple[bool, bool, Set[str], List[str]]:
    """Scan the repository and collect language detection results.

    Returns:
        (python_detected, other_language_detected, detected_languages, python_file_paths)
    """
    python_detected = False
    other_language_detected = False
    detected_languages: Set[str] = set()
    python_file_paths: List[str] = []

    print("Scanning files for language detection...")
    for file_path in Path(".").rglob("*"):
        if not _is_valid_source_file(file_path):
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            prediction = _predict_language(
                file_path, content, model, tfidf_char, tfidf_word, class_names
            )
            if prediction.lower() == "python":
                python_detected = True
                python_file_paths.append(str(file_path))
            else:
                other_language_detected = True
                detected_languages.add(prediction)
                print(f"Detected other language ({prediction}) in file: {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    return python_detected, other_language_detected, detected_languages, python_file_paths


def main() -> None:
    """Entry point for language detection and workflow control.

    Detects programming languages in the repository, writes helper files, and
    updates GitHub Action outputs accordingly.
    """
    model, tfidf_char, tfidf_word, class_names = _load_model_bundle(MODEL_RELATIVE_PATH)

    (
        python_detected,
        other_language_detected,
        detected_languages,
        python_file_paths,
    ) = _process_repository(model, tfidf_char, tfidf_word, class_names)

    langs_str = ", ".join(detected_languages)

    if other_language_detected and python_detected:
        print(f"Mixed repo detected. Non-Python languages found: {langs_str}")
        print(f"Proceeding with {len(python_file_paths)} Python file(s) only. Skipping: {langs_str}")
        _write_python_file_list(python_file_paths)
        _append_to_github_output(
            [
                f"{PROCEED_KEY}=true",
                f"{PYTHON_ONLY_MODE}=true",
                f"{SKIPPED_LANGUAGES_KEY}={langs_str}",
            ]
        )
    elif other_language_detected:
        print(f"Non-Python language detected ({langs_str}). No Python files found. Stopping system.")
        _append_to_github_output(
            [
                f"{PROCEED_KEY}=false",
                f"{DETECTED_LANGUAGES_KEY}={langs_str}",
            ]
        )
        sys.exit(1)
    elif python_detected:
        print("Only Python files detected. Proceeding...")
        _append_to_github_output([f"{PROCEED_KEY}=true"])
    else:
        print("No readable source files found.")
        _append_to_github_output([f"{PROCEED_KEY}=false"])


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
