import os
import sys
import joblib
from pathlib import Path
from scipy.sparse import hstack, csr_matrix

from pathlib import Path
import sys
import os
import joblib
from scipy.sparse import csr_matrix, hstack
from typing import Any, List, Set, Tuple

MODEL_PATH = Path("models/best_language_classifier.joblib")
VALID_EXTENSIONS = {".py", ".java", ".cs", ".js", ".php"}
PYTHON_LANG = "python"


def load_classifier(model_path: Path) -> Tuple[Any, Any, Any, List[str]]:
    """Load the language classifier bundle.

    Args:
        model_path: Path to the joblib file containing the model and vectorizers.

    Returns:
        A tuple containing the model, character TF‑IDF vectorizer,
        word TF‑IDF vectorizer, and list of class names.

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


def is_hidden_path(path: Path) -> bool:
    """Return True if any part of the path is hidden (starts with a dot)."""
    return any(part.startswith(".") for part in path.parts)


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
        tfidf_char: Character‑level TF‑IDF vectorizer.
        tfidf_word: Word‑level TF‑IDF vectorizer.
        model: Trained classifier.
        class_names: Mapping from numeric labels to language names.

    Returns:
        Predicted language name.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

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
    return str(class_names[pred_label])


def write_python_file_list(paths: List[str], filename: str = "python_files.txt") -> None:
    """Write the list of Python file paths to a text file."""
    with open(filename, "w", encoding="utf-8") as f:
        for p in paths:
            f.write(p + "\n")


def append_github_output(lines: List[str]) -> None:
    """Append key‑value lines to the GitHub Actions output file if set."""
    github_output = os.getenv("GITHUB_OUTPUT")
    if not github_output:
        return
    with open(github_output, "a", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def main() -> None:
    """Detect repository language composition and decide further processing.

    The function loads a pre‑trained classifier, scans source files,
    determines whether the repo contains Python, other languages, or both,
    and writes appropriate signals for downstream GitHub Actions steps.
    """
    model, tfidf_char, tfidf_word, class_names = load_classifier(MODEL_PATH)

    python_files: List[str] = []
    other_langs: Set[str] = set()
    python_detected = False
    other_detected = False

    print("Scanning files for language detection...")
    for path in Path(".").rglob("*"):
        if not path.is_file():
            continue
        if is_hidden_path(path) or "models" in path.parts:
            continue
        if path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        try:
            language = predict_language(path, tfidf_char, tfidf_word, model, class_names)
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue

        if language.lower() == PYTHON_LANG:
            python_detected = True
            python_files.append(str(path))
        else:
            other_detected = True
            other_langs.add(language)
            print(f"Detected other language ({language}) in file: {path}")

    langs_str = ", ".join(other_langs)

    if other_detected and python_detected:
        print(f"Mixed repo detected. Non-Python languages found: {langs_str}")
        print(f"Proceeding with {len(python_files)} Python file(s) only. Skipping: {langs_str}")
        write_python_file_list(python_files)
        append_github_output(
            [
                "proceed=true",
                "python_only_mode=true",
                f"skipped_languages={langs_str}",
            ]
        )
        return

    if other_detected and not python_detected:
        print(f"Non-Python language detected ({langs_str}). No Python files found. Stopping system.")
        append_github_output(
            [
                "proceed=false",
                f"detected_languages={langs_str}",
            ]
        )
        sys.exit(1)

    if python_detected:
        print("Only Python files detected. Proceeding...")
        append_github_output(["proceed=true"])
        return

    print("No readable source files found.")
    append_github_output(["proceed=false"])


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
