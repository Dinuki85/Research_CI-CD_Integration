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

import ast
from typing import Any, Dict, List, Optional, Tuple


ZERO_INDEX_OFFSET = 1
DEFAULT_LINE_INCREMENT = 1


def _get_source_code(self, node: ast.FunctionDef) -> str:
    """Return the source code for the given function node."""
    start = node.lineno - ZERO_INDEX_OFFSET
    end = getattr(node, "end_lineno", start + DEFAULT_LINE_INCREMENT)
    source_lines = self.source_lines[start:end]
    return "\n".join(source_lines)


def _extract_decorators(self, node: ast.FunctionDef) -> List[str]:
    """Extract decorator names from a function node."""
    decorators: List[str] = []
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Name):
            decorators.append(decorator.id)
        elif isinstance(decorator, ast.Attribute):
            decorators.append(ast.dump(decorator))
    return decorators


def _extract_params(self, node: ast.FunctionDef) -> List[str]:
    """Extract parameter names from a function node."""
    return [arg.arg for arg in node.args.args]


def _determine_flags(
    self,
    params: List[str],
    parent_class: Optional[str],
) -> Tuple[bool, bool, str]:
    """Determine method, nesting flags and full name."""
    is_method = bool(params and params[0] in ("self", "cls"))
    is_nested = parent_class is not None and not is_method
    full_name = f"{parent_class}.{params[0]}" if parent_class else params[0]
    return is_method, is_nested, full_name


def _extract_function_info(
    self,
    node: ast.FunctionDef,
    parent_class: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract metadata and source code for a function definition.

    Args:
        node: The AST node representing the function.
        parent_class: Name of the enclosing class, if any.

    Returns:
        A dictionary containing function metadata such as name, source code,
        line numbers, parameters, decorators, and flags indicating whether the
        function is a method or nested.
    """
    source_code = self._get_source_code(node)
    decorators = self._extract_decorators(node)
    params = self._extract_params(node)

    is_method, is_nested, full_name = self._determine_flags(params, parent_class)

    start_line = node.lineno
    end_line = getattr(node, "end_lineno", start_line + DEFAULT_LINE_INCREMENT)

    return {
        "name": node.name,
        "full_name": f"{parent_class}.{node.name}" if parent_class else node.name,
        "file_path": None,
        "source_code": source_code,
        "start_line": start_line,
        "end_line": end_line,
        "num_lines": end_line - start_line + 1,
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


import sys
from pathlib import Path
from typing import List, Dict, Optional, Any

PY_SUFFIX = ".py"


def _validate_file(py_file: Path) -> bool:
    """Check that a file exists and is a Python file.

    Args:
        py_file: Path to the file to validate.

    Returns:
        True if the file exists and has a .py suffix, False otherwise.
    """
    if not py_file.exists():
        print(f"  [WARN] File does not exist: {py_file}", file=sys.stderr)
        return False
    if py_file.suffix != PY_SUFFIX:
        print(f"  [WARN] Not a Python file: {py_file}", file=sys.stderr)
        return False
    return True


def _add_relative_path(func: Dict[str, Any], root: Optional[Path]) -> None:
    """Add a relative path entry to a function info dict.

    Args:
        func: Dictionary containing function information, must include
            a 'file_path' key.
        root: Root directory used to compute the relative path. If None,
            the absolute path is used.
    """
    absolute_path = Path(func["file_path"])
    if root:
        try:
            func["rel_path"] = str(absolute_path.relative_to(root))
        except ValueError:
            func["rel_path"] = func["file_path"]
    else:
        func["rel_path"] = func["file_path"]


def scan_files(
    file_paths: List[Path], root: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """Scan a list of Python files and extract function information.

    Args:
        file_paths: List of paths to Python files to scan.
        root: Optional root path for computing relative paths.

    Returns:
        A list of dictionaries, each describing a discovered function.
    """
    resolved_root = Path(root).resolve() if root else None
    all_functions: List[Dict[str, Any]] = []

    for path in file_paths:
        py_file = Path(path).resolve()
        if not _validate_file(py_file):
            continue

        functions = extract_functions_from_file(py_file)

        for func in functions:
            _add_relative_path(func, resolved_root)

        all_functions.extend(functions)

    return all_functions
