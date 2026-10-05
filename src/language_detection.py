import os
import sys
import joblib
from pathlib import Path
from scipy.sparse import hstack, csr_matrix

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import joblib
from scipy.sparse import csr_matrix, hstack

MODEL_PATH = Path("models/best_language_classifier.joblib")
VALID_EXTENSIONS = {".py", ".java", ".cs", ".js", ".php"}


def _load_model() -> Tuple[Any, Any, Any, List[str]]:
    """Load the language classifier model bundle.

    Returns:
        A tuple containing the classifier model, character TF‑IDF transformer,
        word TF‑IDF transformer, and ordered list of class names.

    Raises:
        SystemExit: If the model file does not exist.
    """
    if not MODEL_PATH.exists():
        print(f"Model not found at {MODEL_PATH}")
        sys.exit(1)

    print("Loading language classifier bundle...")
    bundle: Dict[str, Any] = joblib.load(MODEL_PATH)
    return (
        bundle["model"],
        bundle["tfidf_char"],
        bundle["tfidf_word"],
        bundle["class_names"],
    )


def _iter_source_files() -> List[Path]:
    """Yield all source files that should be inspected.

    Skips hidden directories, the `models` directory,
    and files whose extensions are not in :data:`VALID_EXTENSIONS`.

    Returns:
        A list of qualifying file paths.
    """
    source_files: List[Path] = []
    for path in Path(".").rglob("*"):
        if not path.is_file():
            continue
        if any(part.startswith(".") for part in path.parts):
            continue
        if "models" in path.parts:
            continue
        if path.suffix.lower() not in VALID_EXTENSIONS:
            continue
        source_files.append(path)
    return source_files


def _predict_language(
    file_path: Path,
    model: Any,
    tfidf_char: Any,
    tfidf_word: Any,
    class_names: List[str],
) -> str:
    """Predict the programming language of a given file.

    Args:
        file_path: Path to the source file.
        model: Trained classifier model.
        tfidf_char: Character‑level TF‑IDF transformer.
        tfidf_word: Word‑level TF‑IDF transformer.
        class_names: Mapping from numeric label to language name.

    Returns:
        Detected language name.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    line_count = len(content.splitlines())
    combined = f"{file_path.name.lower()}  {content}"

    x_char = tfidf_char.transform([combined])
    x_word = tfidf_word.transform([combined])
    x_lines = csr_matrix([[float(line_count)]])

    if hasattr(model, "n_features_in_") and model.n_features_in_ == x_char.shape[1] + x_word.shape[1]:
        features = hstack([x_char, x_word])
    else:
        features = hstack([x_char, x_word, x_lines])

    label_idx = model.predict(features)[0]
    return str(class_names[label_idx])


def _write_python_file_list(python_files: List[str]) -> None:
    """Write the list of Python files to ``python_files.txt``."""
    with open("python_files.txt", "w", encoding="utf-8") as out_file:
        for path in python_files:
            out_file.write(f"{path}\n")


def _append_to_github_output(lines: List[str]) -> None:
    """Append key‑value pairs to the GitHub actions output file if defined."""
    github_output = os.environ.get("GITHUB_OUTPUT")
    if not github_output:
        return
    with open(github_output, "a", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def _process_detection_results(
    python_files: List[str],
    other_languages: Set[str],
    python_detected: bool,
    other_detected: bool,
) -> None:
    """Handle the outcomes of the language detection step.

    Prints user‑friendly messages, writes auxiliary files and sets GitHub
    action outputs, and exits the script when appropriate.

    Args:
        python_files: Paths of files identified as Python.
        other_languages: Set of non‑Python language names detected.
        python_detected: ``True`` if any Python file was found.
        other_detected: ``True`` if any non‑Python language was found.
    """
    langs_str = ", ".join(sorted(other_languages))

    if other_detected and python_detected:
        print(f"Mixed repo detected. Non-Python languages found: {langs_str}")
        print(f"Proceeding with {len(python_files)} Python file(s) only. Skipping: {langs_str}")
        _write_python_file_list(python_files)
        _append_to_github_output(
            [
                "proceed=true",
                "python_only_mode=true",
                f"skipped_languages={langs_str}",
            ]
        )
        return

    if other_detected and not python_detected:
        print(f"Non-Python language detected ({langs_str}). No Python files found. Stopping system.")
        _append_to_github_output(
            [
                "proceed=false",
                f"detected_languages={langs_str}",
            ]
        )
        sys.exit(1)

    if python_detected:
        print("Only Python files detected. Proceeding...")
        _append_to_github_output(["proceed=true"])
        return

    print("No readable source files found.")
    _append_to_github_output(["proceed=false"])


def main() -> None:
    """Entry point for language detection in a repository.

    Scans source files, identifies Python files versus other languages,
    and communicates the decision to downstream GitHub Actions steps.
    """
    model, tfidf_char, tfidf_word, class_names = _load_model()

    python_files: List[str] = []
    other_languages: Set[str] = set()
    python_detected = False
    other_detected = False

    print("Scanning files for language detection...")
    for file_path in _iter_source_files():
        try:
            language = _predict_language(file_path, model, tfidf_char, tfidf_word, class_names)
            if language.lower() == "python":
                python_detected = True
                python_files.append(str(file_path))
            else:
                other_detected = True
                other_languages.add(language)
                print(f"Detected other language ({language}) in file: {file_path}")
        except Exception as exc:  # pragma: no cover
            print(f"Error processing {file_path}: {exc}")

    _process_detection_results(python_files, other_languages, python_detected, other_detected)


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
