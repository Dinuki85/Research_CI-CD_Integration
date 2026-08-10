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


def find_python_files(repo_path: Path) -> List[Path]:
    """
    Recursively find all Python files in a repository, excluding
    common non-source directories.

    Returns:
        List of paths to .py files.
    """
    excluded_dirs = {
        ".git", "__pycache__", "venv", ".venv", "env", ".env",
        "node_modules", "dist", "build", ".tox", ".eggs",
        "egg-info", ".mypy_cache", ".pytest_cache", ".coverage",
    }

    python_files = []
    for root, dirs, files in os.walk(str(repo_path)):
        # Modify dirs in-place to skip excluded directories
        dirs[:] = [d for d in dirs if d not in excluded_dirs and not d.startswith(".")]
        for f in files:
            if f.endswith(".py"):
                python_files.append(Path(root) / f)

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

from typing import Any, Dict, List, Optional
import ast


SELF = "self"
CLS = "cls"


def _extract_source_code(source_lines: List[str], start: int, end: int) -> str:
    """Return the source code slice between start and end lines."""
    return "\n".join(source_lines[start:end])


def _extract_decorators(decorator_list: List[ast.expr]) -> List[str]:
    """Return a list of decorator representations."""
    result: List[str] = []
    for decorator in decorator_list:
        if isinstance(decorator, ast.Name):
            result.append(decorator.id)
        elif isinstance(decorator, ast.Attribute):
            result.append(ast.dump(decorator))
    return result


def _extract_parameters(args: ast.arguments) -> List[str]:
    """Return a list of parameter names."""
    return [arg.arg for arg in args.args]


def _determine_flags(params: List[str], parent_class: Optional[str]) -> tuple[bool, bool]:
    """Return (is_method, is_nested) flags."""
    is_method = bool(params and params[0] in (SELF, CLS))
    is_nested = parent_class is not None and not is_method
    return is_method, is_nested


def _extract_function_info(
    self,
    node: ast.FunctionDef,
    parent_class: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract metadata and source code for a function definition.

    Args:
        node: AST node representing the function definition.
        parent_class: Name of the enclosing class, if any.

    Returns:
        Dictionary containing function metadata such as name, source code,
        line numbers, parameters, decorators, and classification flags.

    Raises:
        AttributeError: If required AST attributes are missing.
    """
    start_line = node.lineno - 1  # convert to 0‑indexed
    end_line = getattr(node, "end_lineno", start_line + 1)

    source_code = _extract_source_code(self.source_lines, start_line, end_line)
    decorators = _extract_decorators(node.decorator_list)
    params = _extract_parameters(node.args)
    is_method, is_nested = _determine_flags(params, parent_class)

    full_name = f"{parent_class}.{node.name}" if parent_class else node.name

    return {
        "name": node.name,
        "full_name": full_name,
        "file_path": None,  # Set by caller
        "source_code": source_code,
        "start_line": start_line + 1,  # back to 1‑indexed
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


def scan_functions(repo_path: Path) -> List[Dict]:
    """
    High-level wrapper: scan repo and return functions with their
    relative paths (for nicer display).
    """
    repo_path = Path(repo_path).resolve()
    functions = analyze_repository(repo_path)

    # Add relative file paths
    for func in functions:
        abs_path = Path(func["file_path"])
        try:
            func["rel_path"] = str(abs_path.relative_to(repo_path))
        except ValueError:
            func["rel_path"] = func["file_path"]

    return functions


from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

WARNING_PREFIX = "  [WARN]"


def _validate_file(file_path: Path) -> bool:
    """Check that the given path exists and is a Python file.

    Args:
        file_path: Path to validate.

    Returns:
        True if the file exists and has a ``.py`` suffix, False otherwise.
    """
    if not file_path.exists():
        print(f"{WARNING_PREFIX} File does not exist: {file_path}", file=sys.stderr)
        return False
    if file_path.suffix != ".py":
        print(f"{WARNING_PREFIX} Not a Python file: {file_path}", file=sys.stderr)
        return False
    return True


def _populate_relative_path(
    func_info: Dict[str, Any],
    root: Optional[Path],
) -> None:
    """Add ``rel_path`` to a function info dict based on an optional root.

    Args:
        func_info: Dictionary containing at least ``file_path``.
        root: Root directory used to compute the relative path. If ``None``,
            ``rel_path`` will be identical to ``file_path``.
    """
    absolute_path = Path(func_info["file_path"])
    if root:
        try:
            func_info["rel_path"] = str(absolute_path.relative_to(root))
            return
        except ValueError:
            pass
    func_info["rel_path"] = func_info["file_path"]


def scan_files(
    file_paths: List[Path],
    root: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Scan a list of Python files and extract function definitions.

    Args:
        file_paths: Paths to Python files to scan.
        root: Optional directory used to compute relative paths for each function.

    Returns:
        A list of dictionaries, each describing a discovered function. The
        dictionaries include the original ``file_path`` and an added
        ``rel_path`` key.

    Raises:
        None.
    """
    resolved_root = Path(root).resolve() if root else None
    all_functions: List[Dict[str, Any]] = []

    for raw_path in file_paths:
        py_file = Path(raw_path).resolve()
        if not _validate_file(py_file):
            continue

        functions = extract_functions_from_file(py_file)

        for func in functions:
            _populate_relative_path(func, resolved_root)

        all_functions.extend(functions)

    return all_functions

# Note: ``extract_functions_from_file`` is expected to be imported elsewhere.
