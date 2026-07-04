"""
Tool Loader - 工具自动发现和Schema生成

通过introspection自动发现domain/ocean下的所有函数，并生成LLM可用的schema
"""

import inspect
import importlib
import pkgutil
from typing import Dict, Callable, Any, get_type_hints, get_origin, get_args
from pathlib import Path

from packages.tool_loader.registry import (
    build_param_reference_template,
    build_result_reference_examples,
    get_param_contract,
    get_tool_contract,
    get_tool_output_type,
    get_tool_planner_contract,
)


def discover_tools(package_name: str = "domain.ocean") -> Dict[str, Callable]:
    """
    自动发现指定包下的所有函数

    Args:
        package_name: 包名（默认'domain.ocean'）

    Returns:
        {函数名: 函数对象} 的字典

    Example:
        >>> tools = discover_tools()
        >>> print(list(tools.keys()))
        ['load_dataset', 'extract_regional_mean', ...]
    """
    tools = {}

    try:
        # 导入包
        package = importlib.import_module(package_name)
    except ImportError as e:
        print(f"Warning: Failed to import {package_name}: {e}")
        return tools

    # 遍历所有子模块
    for importer, modname, ispkg in pkgutil.walk_packages(
        path=package.__path__,
        prefix=package.__name__ + '.',
        onerror=lambda x: None
    ):
        try:
            module = importlib.import_module(modname)

            # 获取模块中的所有函数
            for name, obj in inspect.getmembers(module, inspect.isfunction):
                # 跳过私有函数
                if name.startswith('_'):
                    continue

                # 只包含在当前模块定义的函数（避免重复）
                if obj.__module__ == modname:
                    # 使用完整路径作为键，避免重名
                    tool_name = f"{modname.split('.')[-1]}.{name}"
                    tools[tool_name] = obj

        except Exception as e:
            print(f"Warning: Failed to import {modname}: {e}")

    return tools


def get_tool_schema(func: Callable) -> Dict[str, Any]:
    """
    从函数签名自动生成LLM tool schema

    Args:
        func: Python函数

    Returns:
        LLM tool calling格式的schema

    Example:
        >>> schema = get_tool_schema(extract_regional_mean)
        >>> print(schema['name'])
        'extract_regional_mean'
    """
    sig = inspect.signature(func)

    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}

    doc = inspect.getdoc(func) or ""

    # 解析docstring
    param_docs = _parse_docstring(doc)
    description = doc.split('\n\n')[0] if doc else func.__name__

    # 构建parameters schema
    parameters = {
        "type": "object",
        "properties": {},
        "required": []
    }

    for param_name, param in sig.parameters.items():
        # 跳过self和特殊参数
        if param_name in ('self', 'cls', 'kwargs'):
            continue

        param_type = hints.get(param_name, type(None))

        parameters["properties"][param_name] = {
            "type": _python_type_to_json_type(param_type),
            "description": param_docs.get(param_name, "")
        }
        input_contract = get_param_contract(func.__name__, param_name)
        if input_contract:
            parameters["properties"][param_name]["x-input-binding"] = {
                **input_contract,
                "reference_template": build_param_reference_template(func.__name__, param_name),
            }

        # 处理复杂类型（如Tuple, Optional等）
        if get_origin(param_type) is tuple:
            args = get_args(param_type)
            if args:
                parameters["properties"][param_name]["items"] = {
                    "type": _python_type_to_json_type(args[0])
                }

        # 判断是否必需
        if param.default == inspect.Parameter.empty:
            parameters["required"].append(param_name)

    return {
        "name": func.__name__,
        "description": description,
        "parameters": parameters,
        "module": func.__module__,
        "x-orchestration": {
            **(get_tool_contract(func.__name__) or {}),
            "result_reference_examples": build_result_reference_examples(
                get_tool_output_type(func.__name__)
            ),
        }
    }


def get_all_tool_schemas(tools: Dict[str, Callable]) -> Dict[str, Dict]:
    """
    获取所有工具的schema

    Args:
        tools: discover_tools的输出

    Returns:
        {工具名: schema} 的字典

    Example:
        >>> tools = discover_tools()
        >>> schemas = get_all_tool_schemas(tools)
    """
    return {
        name: get_tool_schema(func)
        for name, func in tools.items()
    }


def get_planner_tool_spec(func: Callable) -> Dict[str, Any]:
    """
    生成给 planner 使用的轻量工具规格。

    与 `get_tool_schema()` 不同，这里重点是工具编排信息。
    """
    schema = get_tool_schema(func)
    planner_contract = get_tool_planner_contract(func.__name__)

    return {
        "name": func.__name__,
        "module": func.__module__,
        "description": schema["description"],
        "parameters": planner_contract.get("params", {}),
        "output_type": planner_contract.get("output_type"),
        "result_reference_examples": planner_contract.get("result_reference_examples", {}),
    }


def get_all_planner_tool_specs(tools: Dict[str, Callable]) -> Dict[str, Dict[str, Any]]:
    """批量生成 planner tool specs。"""
    return {
        name: get_planner_tool_spec(func)
        for name, func in tools.items()
    }


def _python_type_to_json_type(py_type: Any) -> str:
    """
    Python类型转JSON Schema类型

    Args:
        py_type: Python类型

    Returns:
        JSON Schema类型字符串
    """
    # 处理Optional
    origin = get_origin(py_type)
    if origin is type(None):
        return "null"

    # 处理Union (包括Optional)
    if origin is not None:
        args = get_args(py_type)
        if args:
            # Optional[X] 返回X的类型
            non_none_types = [t for t in args if t is not type(None)]
            if non_none_types:
                return _python_type_to_json_type(non_none_types[0])

    xarray_types = _get_xarray_type_mapping()

    # 基本类型映射
    type_mapping = {
        int: "integer",
        float: "number",
        str: "string",
        bool: "boolean",
        list: "array",
        dict: "object",
        tuple: "array",
    }
    type_mapping.update(xarray_types)

    # 处理泛型
    if origin is not None:
        return type_mapping.get(origin, "object")

    return type_mapping.get(py_type, "string")


def _get_xarray_type_mapping() -> Dict[Any, str]:
    try:
        import xarray as xr
    except ImportError:
        return {}
    return {
        xr.DataArray: "object",
        xr.Dataset: "object",
    }


def _parse_docstring(doc: str) -> Dict[str, str]:
    """
    解析Google风格的docstring，提取参数描述

    Args:
        doc: docstring文本

    Returns:
        {参数名: 描述} 的字典
    """
    param_docs = {}

    if not doc:
        return param_docs

    lines = doc.split('\n')
    in_args_section = False

    for line in lines:
        line_stripped = line.strip()

        # 检测Args section
        if line_stripped == 'Args:':
            in_args_section = True
            continue

        # 检测其他section（结束Args section）
        if in_args_section and line_stripped.endswith(':') and not ':' in line_stripped[:-1]:
            break

        # 解析参数行
        if in_args_section and ':' in line_stripped:
            parts = line_stripped.split(':', 1)
            if len(parts) == 2:
                param_name = parts[0].strip()
                param_desc = parts[1].strip()
                param_docs[param_name] = param_desc

    return param_docs


def get_tool_by_name(name: str, tools: Dict[str, Callable]) -> Callable:
    """
    根据名称获取工具函数

    Args:
        name: 工具名称
        tools: discover_tools的输出

    Returns:
        工具函数

    Raises:
        ValueError: 如果工具不存在
    """
    # 精确匹配
    if name in tools:
        return tools[name]

    # 模糊匹配（只匹配函数名部分）
    for tool_name, func in tools.items():
        if tool_name.endswith(f".{name}"):
            return func

    raise ValueError(f"Tool not found: {name}")


# 缓存已发现的工具
_cached_tools: Dict[str, Callable] = {}
_cached_schemas: Dict[str, Dict] = {}
_cached_planner_specs: Dict[str, Dict] = {}


def get_tools_cached() -> Dict[str, Callable]:
    """获取缓存的工具列表"""
    global _cached_tools
    if not _cached_tools:
        _cached_tools = discover_tools()
    return _cached_tools


def get_schemas_cached() -> Dict[str, Dict]:
    """获取缓存的schema列表"""
    global _cached_schemas
    if not _cached_schemas:
        tools = get_tools_cached()
        _cached_schemas = get_all_tool_schemas(tools)
    return _cached_schemas


def get_planner_specs_cached() -> Dict[str, Dict]:
    """获取缓存的 planner tool specs。"""
    global _cached_planner_specs
    if not _cached_planner_specs:
        tools = get_tools_cached()
        _cached_planner_specs = get_all_planner_tool_specs(tools)
    return _cached_planner_specs


def reload_tools():
    """重新加载工具（清除缓存）"""
    global _cached_tools, _cached_schemas, _cached_planner_specs
    _cached_tools = {}
    _cached_schemas = {}
    _cached_planner_specs = {}
