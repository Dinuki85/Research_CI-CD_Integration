"""
Repository Analyzer
===================
Scans a local or remote Git repository for Python functions, extracts each
function's source code, and prepares them for code smell analysis.

Supports:
  - Local repository scanning (path)
  - Remote repository cloning (Git URL)
  - AST-based function extraction with source code preservation
  - Function metadata (name, file, line numbers, decorators)
"""

import ast
import os
import re
import sys
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Optional


def clone_repository(repo_url: str, target_dir: Optional[Path] = None) -> Path:
    """
    Clone a git repository to a local directory.

    Args:
        repo_url: Git repository URL (HTTPS or SSH).
        target_dir: Target directory. If None, creates a temp directory.

    Returns:
        Path to the cloned repository.
    """
    if target_dir is None:
        target_dir = Path(tempfile.mkdtemp(prefix="coderefactor-"))

    print(f"Cloning {repo_url} into {target_dir}...")

    result = subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(target_dir)],
        capture_output=True,
        text=True,
        timeout=300,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Failed to clone repository: {result.stderr}")

    return target_dir


from pathlib import Path
import os
from typing import List

_EXCLUDED_DIRS = {
    ".git",
    "__pycache__",
    "venv",
    ".venv",
    "env",
    ".env",
    "node_modules",
    "dist",
    "build",
    ".tox",
    ".eggs",
    "egg-info",
    ".mypy_cache",
    ".pytest_cache",
    ".coverage",
}


def _filter_dirs(dirs: List[str]) -> List[str]:
    """Return a list of directory names that are not excluded.

    Args:
        dirs: List of directory names from os.walk.

    Returns:
        Filtered list excluding known non-source directories and hidden folders.
    """
    return [
        d
        for d in dirs
        if d not in _EXCLUDED_DIRS and not d.startswith(".")
    ]


def find_python_files(repo_path: Path) -> List[Path]:
    """Recursively locate all Python files in a repository.

    Args:
        repo_path: Path to the root of the repository to search.

    Returns:
        A sorted list of Path objects pointing to ``.py`` files found.

    Raises:
        OSError: If the repository path cannot be accessed.
    """
    python_files: List[Path] = []
    for root, dirs, files in os.walk(str(repo_path)):
        dirs[:] = _filter_dirs(dirs)
        for filename in files:
            if filename.endswith(".py"):
                python_files.append(Path(root) / filename)

    return sorted(python_files)


class FunctionExtractor(ast.NodeVisitor):
    """
    AST visitor that extracts top-level and nested function definitions
    with their source code ranges.
    """

    def __init__(self, source_code: str):
        self.source_code = source_code
        self.source_lines = source_code.split("\n")
        self.functions: List[Dict] = []

import ast
from typing import Any, Dict, List, Optional


class YourClassName:
    """Class containing utilities for extracting function metadata."""

    _DEFAULT_START_LINE_OFFSET = 1  # Convert from 0‑index to 1‑index

    def _extract_function_info(
        self, node: ast.FunctionDef, parent_class: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract metadata and source code for a function definition.

        Args:
            node: The AST node representing the function.
            parent_class: Name of the enclosing class, if any.

        Returns:
            A dictionary with keys such as ``name``, ``full_name``, ``source_code``,
            ``start_line``, ``end_line``, ``params``, ``decorators``, ``is_method``,
            ``is_nested``, ``parent_class`` and ``docstring``.

        Raises:
            ValueError: If ``node`` does not contain line number information.
        """
        start_line = self._get_start_line(node)
        end_line = self._get_end_line(node, start_line)

        source_code = self._extract_source_code(start_line, end_line)
        decorators = self._extract_decorators(node)
        params = self._extract_params(node)

        is_method = bool(params and params[0] in ("self", "cls"))
        is_nested = parent_class is not None and not is_method
        full_name = f"{parent_class}.{node.name}" if parent_class else node.name

        return {
            "name": node.name,
            "full_name": full_name,
            "file_path": None,  # Set by caller
            "source_code": source_code,
            "start_line": start_line + self._DEFAULT_START_LINE_OFFSET,
            "end_line": end_line,
            "num_lines": end_line - start_line,
            "params": params,
            "num_params": len(params),
            "decorators": decorators,
            "is_method": is_method,
            "is_nested": is_nested,
            "parent_class": parent_class,
            "docstring": ast.get_docstring(node),
        }

    def _get_start_line(self, node: ast.FunctionDef) -> int:
        """Return the zero‑based start line of the function."""
        if not hasattr(node, "lineno"):
            raise ValueError("AST node lacks 'lineno' attribute.")
        return node.lineno - self._DEFAULT_START_LINE_OFFSET

    def _get_end_line(self, node: ast.FunctionDef, start_line: int) -> int:
        """Return the end line (inclusive) of the function."""
        if hasattr(node, "end_lineno"):
            return node.end_lineno
        return start_line + 1

    def _extract_source_code(self, start: int, end: int) -> str:
        """Join source lines between ``start`` and ``end`` into a string."""
        source_lines = self.source_lines[start:end]
        return "\n".join(source_lines)

    def _extract_decorators(self, node: ast.FunctionDef) -> List[str]:
        """Return a list of decorator representations for the function."""
        decorators: List[str] = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
            elif isinstance(decorator, ast.Attribute):
                decorators.append(ast.dump(decorator))
        return decorators

    def _extract_params(self, node: ast.FunctionDef) -> List[str]:
        """Return a list of parameter names for the function."""
        return [arg.arg for arg in node.args.args]

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # Skip __init__ and other dunder methods (they're standard)
        if node.name.startswith("__") and node.name.endswith("__") and node.name != "__init__":
            return

        info = self._extract_function_info(node)
        self.functions.append(info)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        info = self._extract_function_info(node)
        self.functions.append(info)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        """Visit class definitions and extract their methods."""
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                info = self._extract_function_info(item, parent_class=node.name)
                self.functions.append(info)
        # Don't visit nested classes recursively (for now)


def extract_functions_from_file(file_path: Path) -> List[Dict]:
    """
    Extract all function definitions from a single Python file.

    Returns:
        List of function info dicts with keys:
            name, full_name, file_path, source_code, start_line, end_line,
            num_lines, params, num_params, decorators, is_method, docstring
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            source_code = f.read()
    except Exception as e:
        print(f"  [WARN] Cannot read {file_path}: {e}", file=sys.stderr)
        return []

    # Skip empty or very short files
    if len(source_code.strip()) < 20:
        return []

    # Try to parse AST
    try:
        tree = ast.parse(source_code, filename=str(file_path))
    except SyntaxError as e:
        print(f"  [WARN] Syntax error in {file_path}: {e}", file=sys.stderr)
        return []

    extractor = FunctionExtractor(source_code)
    extractor.visit(tree)

    # Set file path for each function
    for func in extractor.functions:
        func["file_path"] = str(file_path)

    return extractor.functions


def analyze_repository(repo_path: Path) -> List[Dict]:
    """
    Analyze an entire repository and extract all function definitions.

    Args:
        repo_path: Path to the local repository.

    Returns:
        List of all function info dicts across all Python files.
    """
    repo_path = Path(repo_path).resolve()
    if not repo_path.exists():
        raise FileNotFoundError(f"Repository path does not exist: {repo_path}")

    python_files = find_python_files(repo_path)
    print(f"Found {len(python_files)} Python files in {repo_path}")

    all_functions = []
    for py_file in python_files:
        functions = extract_functions_from_file(py_file)
        all_functions.extend(functions)

    print(f"Extracted {len(all_functions)} functions total")

    return all_functions


from pathlib import Path
from typing import List, Dict, Any, Union


def _relative_path(file_path: Path, repo_root: Path) -> str:
    """Return the path of *file_path* relative to *repo_root*.

    If *file_path* is not located within *repo_root*, the original absolute
    path is returned.

    Args:
        file_path: Absolute path to the file.
        repo_root: Root directory of the repository.

    Returns:
        Relative path as a string, or the original absolute path if it cannot be
        made relative.
    """
    try:
        return str(file_path.relative_to(repo_root))
    except ValueError:
        return str(file_path)


def scan_functions(repo_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Scan a repository and return information about its functions.

    The function analyzes the repository located at *repo_path* and augments
    each discovered function dictionary with a relative file path for easier
    display.

    Args:
        repo_path: Path to the repository root, either as a string or a Path
            object.

    Returns:
        A list of dictionaries, each representing a function. Each dictionary
        includes the original keys returned by ``analyze_repository`` plus an
        additional ``rel_path`` key containing the file path relative to the
        repository root.

    Raises:
        Any exception raised by ``analyze_repository`` is propagated unchanged.
    """
    repo_root = Path(repo_path).resolve()
    functions = analyze_repository(repo_root)

    for function in functions:
        absolute_path = Path(function["file_path"])
        function["rel_path"] = _relative_path(absolute_path, repo_root)

    return functions


from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

_WARN_FILE_NOT_EXIST = "  [WARN] File does not exist: {path}"
_WARN_NOT_PYTHON = "  [WARN] Not a Python file: {path}"


def _validate_file(py_file: Path) -> bool:
    """Check that *py_file* exists and is a Python file.

    Args:
        py_file: Path to validate.

    Returns:
        True if the file exists and has a ``.py`` suffix, False otherwise.
    """
    if not py_file.exists():
        print(_WARN_FILE_NOT_EXIST.format(path=py_file), file=sys.stderr)
        return False
    if py_file.suffix != ".py":
        print(_WARN_NOT_PYTHON.format(path=py_file), file=sys.stderr)
        return False
    return True


def _add_relative_paths(
    functions: List[Dict[str, Any]], root: Optional[Path]
) -> None:
    """Populate ``rel_path`` for each function dict.

    Args:
        functions: List of function info dictionaries containing ``file_path``.
        root: Optional root directory to compute relative paths.
    """
    for func in functions:
        abs_path = Path(func["file_path"])
        if root:
            try:
                func["rel_path"] = str(abs_path.relative_to(root))
                continue
            except ValueError:
                pass
        func["rel_path"] = func["file_path"]


def scan_files(file_paths: List[Path], root: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Scan a list of Python files and extract function information.

    Args:
        file_paths: Paths to Python files to scan.
        root: Optional root path for computing relative paths.

    Returns:
        A list of dictionaries, each describing a discovered function.
    """
    resolved_root = Path(root).resolve() if root else None
    all_functions: List[Dict[str, Any]] = []

    for raw_path in file_paths:
        py_file = Path(raw_path).resolve()
        if not _validate_file(py_file):
            continue

        functions = extract_functions_from_file(py_file)
        _add_relative_paths(functions, resolved_root)
        all_functions.extend(functions)

    return all_functions
