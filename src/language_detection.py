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
GITHUB_OUTPUT_ENV = "GITHUB_OUTPUT"


def load_model(model_path: Path) -> Tuple[Any, Any, Any, List[str]]:
    """Load the language classifier model bundle.

    Args:
        model_path: Path to the joblib bundle containing the model and vectorizers.

    Returns:
        A tuple of (model, tfidf_char, tfidf_word, class_names).

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


def is_hidden_path(file_path: Path) -> bool:
    """Return True if any part of the path is a hidden directory."""
    return any(part.startswith(".") for part in file_path.parts)


def should_process_file(file_path: Path) -> bool:
    """Determine whether a file should be inspected."""
    return (
        file_path.is_file()
        and not is_hidden_path(file_path)
        and "models" not in file_path.parts
        and file_path.suffix.lower() in VALID_EXTENSIONS
    )


def predict_language(
    file_path: Path,
    content: str,
    model: Any,
    tfidf_char: Any,
    tfidf_word: Any,
    class_names: List[str],
) -> str:
    """Predict the programming language of a file's content.

    Returns:
        The predicted language name.
    """
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
    model: Any,
    tfidf_char: Any,
    tfidf_word: Any,
    class_names: List[str],
) -> Tuple[bool, bool, Set[str], List[str]]:
    """Walk the repository and detect Python and other languages.

    Returns:
        A tuple containing:
            - python_detected: True if any Python file is found.
            - other_language_detected: True if any non‑Python language is found.
            - detected_other_languages: Set of detected non‑Python language names.
            - python_files: List of paths to detected Python files.
    """
    python_detected = False
    other_language_detected = False
    detected_other_languages: Set[str] = set()
    python_files: List[str] = []

    print("Scanning files for language detection...")
    for file_path in Path(".").rglob("*"):
        if not should_process_file(file_path):
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        try:
            prediction = predict_language(
                file_path, content, model, tfidf_char, tfidf_word, class_names
            )
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

        if prediction.lower() == "python":
            python_detected = True
            python_files.append(str(file_path))
        else:
            other_language_detected = True
            detected_other_languages.add(prediction)
            print(f"Detected other language ({prediction}) in file: {file_path}")

    return python_detected, other_language_detected, detected_other_languages, python_files


def write_python_file_list(python_files: List[str], output_path: Path = Path("python_files.txt")) -> None:
    """Write the list of detected Python file paths to a file."""
    with open(output_path, "w", encoding="utf-8") as pf:
        for path in python_files:
            pf.write(f"{path}\n")


def append_to_github_output(lines: List[str]) -> None:
    """Append given lines to the GitHub Actions output file, if defined."""
    github_output = os.getenv(GITHUB_OUTPUT_ENV)
    if not github_output:
        return
    with open(github_output, "a", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def handle_detection_results(
    python_detected: bool,
    other_language_detected: bool,
    detected_other_languages: Set[str],
    python_files: List[str],
) -> None:
    """Decide the next steps based on detection results and update outputs."""
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


def main() -> None:
    """Execute language detection workflow for the current repository.

    The script loads a pre‑trained classifier, scans files,
    determines repository language composition, and writes
    results for downstream CI steps.
    """
    model, tfidf_char, tfidf_word, class_names = load_model(MODEL_REL_PATH)

    (python_detected, other_language_detected, detected_other_languages, python_files) = scan_repository(
        model, tfidf_char, tfidf_word, class_names
    )

    handle_detection_results(
        python_detected,
        other_language_detected,
        detected_other_languages,
        python_files,
    )


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
