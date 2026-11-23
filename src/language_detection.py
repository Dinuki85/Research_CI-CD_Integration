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

MODEL_REL_PATH = Path("models/best_language_classifier.joblib")
VALID_EXTENSIONS = {".py", ".java", ".cs", ".js", ".php"}
PYTHON_ONLY_MODE_KEY = "python_only_mode"
PROCEED_KEY = "proceed"
SKIPPED_LANGUAGES_KEY = "skipped_languages"
DETECTED_LANGUAGES_KEY = "detected_languages"


def load_model_bundle(model_path: Path) -> Tuple[object, object, object, List[str]]:
    """Load the language classifier bundle.

    Args:
        model_path: Path to the joblib file containing the model bundle.

    Returns:
        A tuple containing the model, character-level TF‑IDF transformer,
        word-level TF‑IDF transformer, and list of class names.

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


def predict_file_language(
    file_path: Path,
    model: object,
    tfidf_char: object,
    tfidf_word: object,
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
        features = hstack([x_char, x_word])
    else:
        features = hstack([x_char, x_word, x_lines])

    try:
        pred_label = model.predict(features)[0]
        return class_names[pred_label]
    except Exception as e:
        print(f"Error predicting {file_path}: {e}")
        return None


def scan_repository(
    model: object,
    tfidf_char: object,
    tfidf_word: object,
    class_names: List[str],
) -> Tuple[List[str], Set[str], bool, bool]:
    """Traverse the repository and collect language detection results.

    Returns:
        A tuple containing:
            - List of Python file paths.
            - Set of detected non‑Python languages.
            - Flag indicating if any Python file was found.
            - Flag indicating if any non‑Python language was found.
    """
    python_files: List[str] = []
    other_languages: Set[str] = set()
    python_found = False
    other_found = False

    for file_path in Path(".").rglob("*"):
        if not file_path.is_file():
            continue
        if any(part.startswith(".") for part in file_path.parts):
            continue
        if "models" in file_path.parts:
            continue
        if file_path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        prediction = predict_file_language(
            file_path, model, tfidf_char, tfidf_word, class_names
        )
        if prediction is None:
            continue

        if prediction.lower() == "python":
            python_found = True
            python_files.append(str(file_path))
        else:
            other_found = True
            other_languages.add(str(prediction))
            print(f"Detected other language ({prediction}) in file: {file_path}")

    return python_files, other_languages, python_found, other_found


def write_python_file_list(python_files: List[str], output_path: Path = Path("python_files.txt")) -> None:
    """Write the list of Python file paths to a text file."""
    with open(output_path, "w", encoding="utf-8") as pf:
        for path in python_files:
            pf.write(f"{path}\n")


def update_github_output(
    github_output_path: Optional[str],
    entries: dict,
) -> None:
    """Append key‑value pairs to the GitHub Actions output file."""
    if not github_output_path:
        return
    try:
        with open(github_output_path, "a", encoding="utf-8") as f:
            for key, value in entries.items():
                f.write(f"{key}={value}\n")
    except Exception as e:
        print(f"Error writing to GITHUB_OUTPUT: {e}")


def main() -> None:
    """Entry point for language detection and pipeline decision.

    The script loads a pre‑trained classifier, scans the repository for source
    files, determines the presence of Python versus other languages, and writes
    appropriate flags for downstream GitHub Actions steps.
    """
    try:
        model, tfidf_char, tfidf_word, class_names = load_model_bundle(MODEL_REL_PATH)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    print("Scanning files for language detection...")
    python_files, other_languages, python_detected, other_detected = scan_repository(
        model, tfidf_char, tfidf_word, class_names
    )

    github_output = os.getenv("GITHUB_OUTPUT")
    other_langs_str = ", ".join(other_languages)

    if other_detected and python_detected:
        print(f"Mixed repo detected. Non-Python languages found: {other_langs_str}")
        print(f"Proceeding with {len(python_files)} Python file(s) only. Skipping: {other_langs_str}")
        write_python_file_list(python_files)
        update_github_output(
            github_output,
            {
                PROCEED_KEY: "true",
                PYTHON_ONLY_MODE_KEY: "true",
                SKIPPED_LANGUAGES_KEY: other_langs_str,
            },
        )
        return

    if other_detected and not python_detected:
        print(f"Non-Python language detected ({other_langs_str}). No Python files found. Stopping system.")
        update_github_output(
            github_output,
            {
                PROCEED_KEY: "false",
                DETECTED_LANGUAGES_KEY: other_langs_str,
            },
        )
        sys.exit(1)

    if python_detected and not other_detected:
        print("Only Python files detected. Proceeding...")
        update_github_output(github_output, {PROCEED_KEY: "true"})
        return

    print("No readable source files found.")
    update_github_output(github_output, {PROCEED_KEY: "false"})


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
