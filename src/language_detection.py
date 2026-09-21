import os
import sys
import joblib
from pathlib import Path
from scipy.sparse import hstack, csr_matrix

import sys
import os
from pathlib import Path
from typing import List, Set, Tuple

import joblib
from scipy.sparse import csr_matrix, hstack

MODEL_PATH = Path("models/best_language_classifier.joblib")
VALID_EXTENSIONS = {".py", ".java", ".cs", ".js", ".php"}
PYTHON_ONLY_OUTPUT = "python_files.txt"


def _load_classifier(model_path: Path) -> Tuple[object, object, object, List[str]]:
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


def _is_hidden(path: Path) -> bool:
    """Return True if any part of the path is a hidden directory."""
    return any(part.startswith(".") for part in path.parts)


def _predict_language(
    file_path: Path,
    content: str,
    model,
    tfidf_char,
    tfidf_word,
    class_names: List[str],
) -> str:
    """Predict the programming language of a file.

    Args:
        file_path: Path object of the file.
        content: Text content of the file.
        model: Trained classifier.
        tfidf_char: Character level TF‑IDF transformer.
        tfidf_word: Word level TF‑IDF transformer.
        class_names: Mapping from numeric label to language name.

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

    pred_label = model.predict(features)[0]
    return class_names[pred_label]


def _write_python_file_list(python_files: List[str], output_path: Path = Path(PYTHON_ONLY_OUTPUT)) -> None:
    """Write the list of detected Python files to a text file."""
    with output_path.open("w", encoding="utf-8") as f:
        for path in python_files:
            f.write(f"{path}\n")


def _append_to_github_output(lines: List[str]) -> None:
    """Append key‑value pairs to the GitHub Actions output file if set."""
    github_output = os.getenv("GITHUB_OUTPUT")
    if not github_output:
        return
    with Path(github_output).open("a", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def main() -> None:
    """Detect programming languages in the repository and decide pipeline flow.

    The function loads a pretrained language classifier, scans source files,
    records detected languages, and writes decision flags for GitHub Actions.
    """
    try:
        model, tfidf_char, tfidf_word, class_names = _load_classifier(MODEL_PATH)
    except FileNotFoundError as exc:
        print(exc)
        sys.exit(1)

    python_detected = False
    other_language_detected = False
    detected_other_languages: Set[str] = set()
    python_files: List[str] = []

    print("Scanning files for language detection...")
    for file_path in Path(".").rglob("*"):
        if not file_path.is_file():
            continue
        if _is_hidden(file_path) or "models" in file_path.parts:
            continue
        if file_path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        try:
            content = file_path.read_text(encoding="utf-8")
            prediction = _predict_language(
                file_path, content, model, tfidf_char, tfidf_word, class_names
            )
            if prediction.lower() == "python":
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
        _write_python_file_list(python_files)
        _append_to_github_output(
            [
                "proceed=true",
                "python_only_mode=true",
                f"skipped_languages={langs}",
            ]
        )
    elif other_language_detected:
        print(f"Non-Python language detected ({langs}). No Python files found. Stopping system.")
        _append_to_github_output(
            [
                "proceed=false",
                f"detected_languages={langs}",
            ]
        )
        sys.exit(1)
    elif python_detected:
        print("Only Python files detected. Proceeding...")
        _append_to_github_output(["proceed=true"])
    else:
        print("No readable source files found.")
        _append_to_github_output(["proceed=false"])


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
