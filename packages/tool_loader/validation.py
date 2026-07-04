"""
Tool validation - 计划参数校验

基于 registry 中的工具契约，在执行前给出轻量校验结果。
"""

from typing import Any, Dict, List, Optional

from packages.tool_loader.registry import get_param_contract, get_tool_contract


def validate_tool_params(tool_name: str, params: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    校验工具参数是否符合编排契约。

    返回 issue 列表，每项结构：
    {
        "level": "warning" | "error",
        "param": "timeseries",
        "message": "..."
    }
    """
    contract = get_tool_contract(tool_name)
    if not contract:
        return []

    issues: List[Dict[str, str]] = []

    # First check for undefined parameters
    defined_params = set(contract.get("inputs", {}).keys())
    for param_name in params.keys():
        if param_name not in defined_params:
            issues.append({
                "level": "error",
                "param": param_name,
                "message": f"Parameter '{param_name}' is not defined in tool contract for '{tool_name}'. Valid parameters: {', '.join(sorted(defined_params))}"
            })

    # Then validate defined parameters
    for param_name, value in params.items():
        param_contract = get_param_contract(tool_name, param_name)
        if not param_contract:
            continue
        issues.extend(_validate_param_value(tool_name, param_name, value, param_contract))

    return issues


def _validate_param_value(
    tool_name: str,
    param_name: str,
    value: Any,
    param_contract: Dict[str, Any]
) -> List[Dict[str, str]]:
    issues: List[Dict[str, str]] = []
    kind = param_contract.get("kind")
    expected = param_contract.get("expected", "")

    numeric_issue = _validate_numeric_range(tool_name, param_name, value, param_contract)
    if numeric_issue is not None:
        issues.append(numeric_issue)

    if isinstance(value, dict):
        return issues

    if kind == "literal":
        if _is_reference_string(value):
            issues.append({
                "level": "warning",
                "param": param_name,
                "message": f"Parameter '{param_name}' is declared as literal but received a reference.",
            })
        return issues

    if kind == "literal_or_ref":
        return issues

    if kind == "ref_result":
        if not _is_reference_string(value):
            issues.append({
                "level": "warning",
                "param": param_name,
                "message": (
                    f"Parameter '{param_name}' expects a result reference"
                    f" ({expected}) but received a literal value."
                ),
            })
            return issues

        if isinstance(value, str) and value.startswith("$ref:") and "." in value[5:]:
            issues.append({
                "level": "warning",
                "param": param_name,
                "message": (
                    f"Parameter '{param_name}' expects the whole upstream result"
                    f" ({expected}), but received a field path reference '{value}'."
                ),
            })
        return issues

    if kind == "ref_field":
        if not _is_reference_string(value):
            issues.append({
                "level": "warning",
                "param": param_name,
                "message": (
                    f"Parameter '{param_name}' expects a field reference"
                    f" ({expected}) but received a literal value."
                ),
            })
            return issues

        if isinstance(value, str) and value.startswith("$"):
            if value.startswith("$ref:"):
                path = value[5:]
                expected_field = expected.split(".", 1)[1] if "." in expected else ""
                if expected_field and not path.endswith(expected_field):
                    issues.append({
                        "level": "warning",
                        "param": param_name,
                        "message": (
                            f"Parameter '{param_name}' usually expects field '{expected_field}',"
                            f" but received '{value}'."
                        ),
                    })
            else:
                expected_field = expected.split(".", 1)[1] if "." in expected else ""
                if expected_field and expected_field != "data":
                    issues.append({
                        "level": "warning",
                        "param": param_name,
                        "message": (
                            f"Legacy reference '{value}' may be ambiguous for parameter '{param_name}'."
                            f" Prefer '$ref:<result_id>.{expected_field}'."
                        ),
                    })

    return issues


def normalize_tool_param_value(
    tool_name: str,
    param_name: str,
    value: Any,
    param_contract: Dict[str, Any],
    *,
    use_default: bool = False,
) -> Any:
    param_type = param_contract.get("type")
    if param_type not in {"number", "integer"}:
        return value

    normalized = _coerce_numeric(value)
    if normalized is None:
        return value

    if param_contract.get("normalize_percentile_to_fraction") and normalized > 1.0:
        normalized = normalized / 100.0

    if param_contract.get("mirror_above_half") and 0.5 < normalized < 1.0:
        normalized = 1.0 - normalized

    if _is_numeric_in_range(normalized, param_contract):
        return _restore_numeric_type(normalized, param_type)

    if use_default:
        default = param_contract.get("default")
        default_numeric = _coerce_numeric(default)
        if default_numeric is not None and _is_numeric_in_range(default_numeric, param_contract):
            return _restore_numeric_type(default_numeric, param_type)

    return value


def _validate_numeric_range(
    tool_name: str,
    param_name: str,
    value: Any,
    param_contract: Dict[str, Any],
) -> Optional[Dict[str, str]]:
    if param_contract.get("type") not in {"number", "integer"}:
        return None

    has_bounds = any(
        key in param_contract
        for key in ("minimum", "maximum", "exclusive_minimum", "exclusive_maximum")
    )
    if not has_bounds:
        return None

    normalized = normalize_tool_param_value(
        tool_name,
        param_name,
        value,
        param_contract,
        use_default=False,
    )
    numeric_value = _coerce_numeric(normalized)
    if numeric_value is None:
        return None

    if _is_numeric_in_range(numeric_value, param_contract):
        return None

    return {
        "level": "error",
        "param": param_name,
        "message": (
            f"Parameter '{param_name}' for '{tool_name}' must be {_describe_numeric_range(param_contract)}; "
            f"received {value!r}."
        ),
    }


def _coerce_numeric(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _restore_numeric_type(value: float, param_type: Any) -> Any:
    if param_type == "integer":
        return int(round(value))
    return float(value)


def _is_numeric_in_range(value: float, param_contract: Dict[str, Any]) -> bool:
    minimum = _coerce_numeric(param_contract.get("minimum"))
    maximum = _coerce_numeric(param_contract.get("maximum"))
    exclusive_minimum = bool(param_contract.get("exclusive_minimum"))
    exclusive_maximum = bool(param_contract.get("exclusive_maximum"))

    if minimum is not None:
        if exclusive_minimum:
            if value <= minimum:
                return False
        elif value < minimum:
            return False

    if maximum is not None:
        if exclusive_maximum:
            if value >= maximum:
                return False
        elif value > maximum:
            return False

    return True


def _describe_numeric_range(param_contract: Dict[str, Any]) -> str:
    minimum = _coerce_numeric(param_contract.get("minimum"))
    maximum = _coerce_numeric(param_contract.get("maximum"))
    exclusive_minimum = bool(param_contract.get("exclusive_minimum"))
    exclusive_maximum = bool(param_contract.get("exclusive_maximum"))

    if minimum is not None and maximum is not None:
        left = "(" if exclusive_minimum else "["
        right = ")" if exclusive_maximum else "]"
        return f"in the range {left}{minimum}, {maximum}{right}"
    if minimum is not None:
        comparator = ">" if exclusive_minimum else ">="
        return f"{comparator} {minimum}"
    if maximum is not None:
        comparator = "<" if exclusive_maximum else "<="
        return f"{comparator} {maximum}"
    return "within the allowed numeric range"


def _is_reference_string(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("$")
