"""Restricted Python code runner for harness code nodes."""

from __future__ import annotations

import ast
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd
import xarray as xr
from scipy import signal, stats


_BLOCKED_IMPORT_ROOTS = {
    "os",
    "sys",
    "subprocess",
    "pathlib",
    "shutil",
    "socket",
    "requests",
    "httpx",
    "urllib",
    "ftplib",
    "glob",
    "pickle",
    "builtins",
}

def _safe_import(name: str, globals=None, locals=None, fromlist=(), level: int = 0):
    root = name.split(".")[0]
    if root in {"numpy", "pandas", "xarray", "scipy"}:
        return __import__(name, globals, locals, fromlist, level)
    raise CodeSafetyError(f"Import is not allowed in code node: {name}")


_SAFE_BUILTINS = {
        "__import__": _safe_import,
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "getattr": getattr,
        "hasattr": hasattr,
        "int": int,
        "isinstance": isinstance,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "range": range,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
}


class CodeSafetyError(ValueError):
    pass


def run_code_node(code: str, inputs: Mapping[str, Any], params: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    """Run a code node that defines ``run(inputs, params) -> dict``."""

    _validate_code_safety(code)
    globals_dict: Dict[str, Any] = {
        "__builtins__": _SAFE_BUILTINS,
        "np": np,
        "pd": pd,
        "xr": xr,
        "stats": stats,
        "signal": signal,
    }
    locals_dict: Dict[str, Any] = {}
    exec(compile(code, "<ocean_harness_code_node>", "exec"), globals_dict, locals_dict)
    run_func = locals_dict.get("run") or globals_dict.get("run")
    if not callable(run_func):
        raise CodeSafetyError("Code node must define run(inputs, params) -> dict")
    result = run_func(dict(inputs), dict(params or {}))
    if not isinstance(result, dict):
        raise CodeSafetyError("Code node run() must return a dict")
    return result


def _validate_code_safety(code: str) -> None:
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in {"numpy", "np", "pandas", "pd", "xarray", "xr", "scipy"}:
                    raise CodeSafetyError(f"Import is not allowed in code node: {alias.name}")
                if root in _BLOCKED_IMPORT_ROOTS:
                    raise CodeSafetyError(f"Blocked import in code node: {alias.name}")
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in {"open", "exec", "eval", "compile", "__import__"}:
                raise CodeSafetyError(f"Blocked call in code node: {func.id}")
            if isinstance(func, ast.Attribute) and func.attr in {"system", "popen", "remove", "unlink", "rmdir"}:
                raise CodeSafetyError(f"Blocked attribute call in code node: {func.attr}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise CodeSafetyError("Dunder attribute access is not allowed in code nodes")
