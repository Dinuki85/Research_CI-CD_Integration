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
from typing import List, Set

EXCLUDED_DIRS: Set[str] = {
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


def _is_excluded(dir_name: str) -> bool:
    """Return True if a directory should be excluded from the search."""
    return dir_name in EXCLUDED_DIRS or dir_name.startswith(".")


def find_python_files(repo_path: Path) -> List[Path]:
    """Recursively find all Python files in a repository, excluding common non‑source directories.

    Args:
        repo_path: Path to the root of the repository.

    Returns:
        Sorted list of Path objects pointing to ``.py`` files.

    Raises:
        None.
    """
    python_files: List[Path] = []
    for root, dirs, files in os.walk(str(repo_path)):
        dirs[:] = [d for d in dirs if not _is_excluded(d)]
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

from typing import Any, Dict, List, Optional
import ast

ZERO_INDEX_OFFSET = -1
DEFAULT_LINE_COUNT = 1


class FunctionInfoExtractor:
    def _extract_function_info(
        self, node: ast.FunctionDef, parent_class: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract metadata and source code for a function definition.

        Args:
            node: AST node representing the function definition.
            parent_class: Name of the enclosing class, if any.

        Returns:
            A dictionary containing function metadata such as name, source code,
            line numbers, parameters, decorators, and classification flags.

        Raises:
            ValueError: If the node does not contain required line information.
        """
        start_line = self._zero_index_line(node.lineno)
        end_line = self._resolve_end_line(node, start_line)

        source_code = self._extract_source_code(start_line, end_line)
        decorators = self._extract_decorators(node.decorator_list)
        params = self._extract_parameters(node.args.args)

        is_method, is_nested = self._classify_function(params, parent_class)

        full_name = f"{parent_class}.{node.name}" if parent_class else node.name

        return {
            "name": node.name,
            "full_name": full_name,
            "file_path": None,
            "source_code": source_code,
            "start_line": start_line + 1,
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

    def _zero_index_line(self, line_number: int) -> int:
        """Convert a 1‑based line number to 0‑based."""
        return line_number + ZERO_INDEX_OFFSET

    def _resolve_end_line(self, node: ast.FunctionDef, start_line: int) -> int:
        """Determine the ending line of a function definition."""
        if hasattr(node, "end_lineno"):
            return node.end_lineno
        return start_line + DEFAULT_LINE_COUNT

    def _extract_source_code(self, start: int, end: int) -> str:
        """Return the source code slice for the given line range."""
        lines = self.source_lines[start:end]
        return "\n".join(lines)

    def _extract_decorators(self, decorator_list: List[ast.AST]) -> List[str]:
        """Collect decorator identifiers from the AST."""
        result: List[str] = []
        for decorator in decorator_list:
            if isinstance(decorator, ast.Name):
                result.append(decorator.id)
            elif isinstance(decorator, ast.Attribute):
                result.append(ast.dump(decorator))
        return result

    def _extract_parameters(self, args: List[ast.arg]) -> List[str]:
        """Return a list of parameter names."""
        return [arg.arg for arg in args]

    def _classify_function(
        self, params: List[str], parent_class: Optional[str]
    ) -> tuple[bool, bool]:
        """Determine if the function is a method and/or nested."""
        is_method = bool(params and params[0] in ("self", "cls"))
        is_nested = parent_class is not None and not is_method
        return is_method, is_nested

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


def _add_relative_paths(functions: List[Dict[str, Any]], repo_root: Path) -> None:
    """Add a ``rel_path`` key to each function dict.

    The relative path is calculated with respect to ``repo_root``. If the
    absolute path cannot be made relative (e.g., it is outside the repository),
    the original absolute path is kept.

    Args:
        functions: List of dictionaries representing functions. Each dict must
            contain a ``file_path`` key with an absolute path string.
        repo_root: The root directory of the repository.
    """
    for func in functions:
        absolute_path = Path(func["file_path"])
        try:
            func["rel_path"] = str(absolute_path.relative_to(repo_root))
        except ValueError:
            func["rel_path"] = func["file_path"]


def scan_functions(repo_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Scan a repository and return information about its functions.

    The function delegates the heavy lifting to ``analyze_repository`` and
    then enriches each result with a path relative to the repository root.

    Args:
        repo_path: Path to the repository root. Can be a string or ``Path`` object.

    Returns:
        A list of dictionaries, each describing a function and containing at least
        ``file_path`` (absolute) and ``rel_path`` (relative to ``repo_path``).

    Raises:
        Any exception raised by ``analyze_repository`` is propagated unchanged.
    """
    repo_root = Path(repo_path).resolve()
    functions = analyze_repository(repo_root)
    _add_relative_paths(functions, repo_root)
    return functions


from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

WARN_FILE_NOT_EXIST = "  [WARN] File does not exist: {}"
WARN_NOT_PYTHON_FILE = "  [WARN] Not a Python file: {}"


def _validate_file(py_file: Path) -> bool:
    """Return True if *py_file* exists and is a Python file; otherwise log a warning."""
    if not py_file.exists():
        print(WARN_FILE_NOT_EXIST.format(py_file), file=sys.stderr)
        return False
    if py_file.suffix != ".py":
        print(WARN_NOT_PYTHON_FILE.format(py_file), file=sys.stderr)
        return False
    return True


def _set_relative_path(func: Dict[str, Any], root: Optional[Path]) -> None:
    """Add a ``rel_path`` key to *func* based on *root*."""
    abs_path = Path(func["file_path"])
    if root:
        try:
            func["rel_path"] = str(abs_path.relative_to(root))
            return
        except ValueError:
            pass
    func["rel_path"] = func["file_path"]


def scan_files(file_paths: List[Path], root: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Scan a list of Python files and extract function metadata.

    Args:
        file_paths: List of paths to Python files to scan.
        root: Optional root directory used to compute relative paths.

    Returns:
        A list of dictionaries, each containing information about a function
        found in the provided files.

    Raises:
        None.
    """
    root_path = Path(root).resolve() if root else None
    all_functions: List[Dict[str, Any]] = []

    for file_path in file_paths:
        py_file = Path(file_path).resolve()
        if not _validate_file(py_file):
            continue

        functions = extract_functions_from_file(py_file)

        for func in functions:
            _set_relative_path(func, root_path)

        all_functions.extend(functions)

    return all_functions
