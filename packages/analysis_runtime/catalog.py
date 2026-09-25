"""Search actual ocean Python tools without skill routing or DSL references."""

from __future__ import annotations

import ast
from functools import lru_cache
import inspect
import textwrap
from typing import Any, Callable, get_args, get_origin, get_type_hints

from packages.tool_loader.introspect import discover_tools


@lru_cache(maxsize=1)
def _tools() -> dict[str, Callable[..., Any]]:
    return discover_tools()


def _default(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (tuple, list)):
        return list(value)
    return repr(value)[:120]


def _dict_return_keys(function: Callable[..., Any]) -> list[str]:
    """Expose literal keys in direct dict returns, when source is available."""
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    except (OSError, TypeError, SyntaxError):
        return []
    keys: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            for key in node.value.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str) and key.value not in keys:
                    keys.append(key.value)
    return keys[:20]


def _json_type(annotation: Any) -> str:
    if annotation is inspect.Parameter.empty:
        return "object"
    origin = get_origin(annotation)
    if origin is not None:
        members = [member for member in get_args(annotation) if member is not type(None)]
        if members and origin not in (list, tuple, dict):
            return _json_type(members[0])
        annotation = origin
    return {
        str: "string", int: "integer", float: "number", bool: "boolean",
        list: "array", tuple: "array", dict: "object",
    }.get(annotation, "object")


def _parameter_docs(doc: str) -> dict[str, str]:
    descriptions: dict[str, str] = {}
    in_args = False
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped in {"Args:", "Arguments:", "Parameters:"}:
            in_args = True
            continue
        if in_args and stripped.endswith(":") and ":" not in stripped[:-1]:
            break
        if in_args and ":" in stripped:
            name, description = stripped.split(":", 1)
            descriptions[name.strip().split(" (")[0]] = description.strip()
    return descriptions


def _describe(key: str, function: Callable[..., Any]) -> dict[str, Any]:
    signature = inspect.signature(function)
    doc = inspect.getdoc(function) or ""
    try:
        hints = get_type_hints(function)
    except Exception:
        hints = {}
    parameter_docs = _parameter_docs(doc)
    returns = doc.split("Returns:", 1)[1].split("\n\n", 1)[0].strip() if "Returns:" in doc else ""
    return {
        "name": function.__name__,
        "discovery_name": key,
        "module": function.__module__,
        "description": doc.split("\n\n", 1)[0][:300] if doc else function.__name__,
        "parameters": [
            {
                "name": name,
                "type": _json_type(hints.get(name, param.annotation)),
                "required": param.default is inspect.Parameter.empty,
                "default": None if param.default is inspect.Parameter.empty else _default(param.default),
                "description": parameter_docs.get(name, "")[:240],
            }
            for name, param in signature.parameters.items()
            if name not in {"self", "cls"}
        ],
        "python_return_type": inspect.formatannotation(signature.return_annotation)
        if signature.return_annotation is not inspect.Signature.empty else None,
        "returns_description": returns[:300],
        "literal_dict_return_keys": _dict_return_keys(function),
    }


def find_tools(query: str = "", *, limit: int = 8) -> dict[str, Any]:
    """Find callable ocean functions by name or description; no skill whitelist."""
    if not 1 <= limit <= 20:
        raise ValueError("limit must be between 1 and 20")
    needle = query.strip().lower()
    terms = needle.replace("_", " ").split()
    matches = [
        (key, function) for key, function in _tools().items()
        if all(term in (key.replace("_", " ") + " " + (inspect.getdoc(function) or "")).lower()
               for term in terms)
    ]
    matches.sort(key=lambda pair: (
        0 if pair[1].__name__.lower() == needle else 1 if needle in pair[1].__name__.lower() else 2,
        pair[0],
    ))
    return {
        "query": query,
        "total_matches": len(matches),
        "tools": [_describe(key, function) for key, function in matches[:limit]],
        "truncated": len(matches) > limit,
    }
