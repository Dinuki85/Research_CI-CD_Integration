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

_LINE_ADJUST = 1


def _extract_function_info(
    self,
    node: ast.FunctionDef,
    parent_class: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract metadata for a function definition node.

    Args:
        node: AST node representing a function definition.
        parent_class: Name of the enclosing class if the function is a method,
            otherwise ``None``.

    Returns:
        A dictionary containing function metadata such as name, full name,
        source code, line numbers, parameters, decorators, and flags indicating
        whether the function is a method or a nested function.

    Raises:
        ValueError: If ``node`` lacks required attributes.
    """
    start_line = node.lineno - _LINE_ADJUST
    end_line = getattr(node, "end_lineno", start_line + _LINE_ADJUST)

    source_code = self._extract_source(start_line, end_line)
    decorators = self._extract_decorators(node.decorator_list)
    params = self._extract_params(node.args.args)

    is_method = bool(params and params[0] in ("self", "cls"))
    is_nested = parent_class is not None and not is_method
    full_name = f"{parent_class}.{node.name}" if parent_class else node.name

    return {
        "name": node.name,
        "full_name": full_name,
        "file_path": None,
        "source_code": source_code,
        "start_line": start_line + _LINE_ADJUST,
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


def _extract_source(self, start: int, end: int) -> str:
    """Return the source code slice for the given line range."""
    lines = self.source_lines[start:end]
    return "\n".join(lines)


def _extract_decorators(self, decorator_nodes: List[ast.AST]) -> List[str]:
    """Convert decorator AST nodes to their string representation."""
    result: List[str] = []
    for dec in decorator_nodes:
        if isinstance(dec, ast.Name):
            result.append(dec.id)
        elif isinstance(dec, ast.Attribute):
            result.append(ast.dump(dec))
    return result


def _extract_params(self, arg_nodes: List[ast.arg]) -> List[str]:
    """Return a list of parameter names from argument nodes."""
    return [arg.arg for arg in arg_nodes]

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


from pathlib import Path
import sys
from typing import List, Dict, Optional, Any, Sequence

PYTHON_SUFFIX = ".py"


def _set_relative_path(func: Dict[str, Any], root: Optional[Path]) -> None:
    """Set the relative path for a function dictionary.

    Args:
        func: Dictionary containing at least a ``file_path`` key.
        root: Root directory to which the path should be made relative. If ``None``,
            the original absolute path is kept.
    """
    absolute_path = Path(func["file_path"])
    if root:
        try:
            func["rel_path"] = str(absolute_path.relative_to(root))
        except ValueError:
            func["rel_path"] = func["file_path"]
    else:
        func["rel_path"] = func["file_path"]


def scan_files(file_paths: Sequence[Path], root: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Scan a list of Python files and extract function information.

    Args:
        file_paths: Iterable of paths to Python files to scan.
        root: Optional root path for computing relative paths.

    Returns:
        A list of dictionaries, each representing a discovered function.

    Raises:
        None.
    """
    resolved_root = Path(root).resolve() if root else None
    all_functions: List[Dict[str, Any]] = []

    for py_file in file_paths:
        py_file = Path(py_file).resolve()
        if not py_file.exists():
            print(f"  [WARN] File does not exist: {py_file}", file=sys.stderr)
            continue
        if py_file.suffix != PYTHON_SUFFIX:
            print(f"  [WARN] Not a Python file: {py_file}", file=sys.stderr)
            continue

        functions = extract_functions_from_file(py_file)

        for func in functions:
            _set_relative_path(func, resolved_root)

        all_functions.extend(functions)

    return all_functions
