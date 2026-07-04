"""Skill loading for the OceanMind harness planner.

The files under ``skills/*/SKILL.md`` are the source of truth for retrieval,
defaults, composition hints, and safe tool-call workflow templates.  Markdown
workflow blocks are parsed as a small DSL; they are not executed as Python.
"""

from __future__ import annotations

import ast
import io
import json
import re
import tokenize
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import yaml


@dataclass(frozen=True)
class ToolCallTemplate:
    title: str
    tool: str
    params_template: Mapping[str, Any]
    save_as: str
    param_docs: Mapping[str, str] = field(default_factory=dict)
    input_artifacts: Mapping[str, Any] = field(default_factory=dict)
    output_artifact: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkflowTemplate:
    workflow_id: str
    skill_id: str
    intents: Tuple[str, ...]
    event_type: Optional[str] = None
    mode: str = "tool_workflow"
    steps: Tuple[ToolCallTemplate, ...] = ()
    planner_parameters: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    analysis_steps: Tuple[Mapping[str, Any], ...] = ()
    final_artifacts: Tuple[str, ...] = ()
    defaults: Mapping[str, Any] = field(default_factory=dict)
    required_inputs: Mapping[str, Any] = field(default_factory=dict)
    output_policy: Mapping[str, Any] = field(default_factory=dict)
    validation_rules: Tuple[str, ...] = ()

    @property
    def recipe_id(self) -> str:
        return self.workflow_id

    @property
    def manual_id(self) -> str:
        return self.skill_id

    @property
    def execution_mode(self) -> str:
        return "manual_context_only" if self.mode == "manual_context_only" else "tool_recipe"

    @property
    def default_params(self) -> Mapping[str, Any]:
        return self.defaults


@dataclass(frozen=True)
class SkillHeader:
    skill_id: str
    title: str
    category: str = ""
    description: str = ""
    input_intent: str = ""
    output_intent: str = ""
    avoid_when: Tuple[str, ...] = ()
    intents: Tuple[str, ...] = ()
    variables: Tuple[str, ...] = ()
    requires: Tuple[str, ...] = ()
    defaults: Mapping[str, Any] = field(default_factory=dict)
    produces: Tuple[Mapping[str, Any], ...] = ()
    mask_support: Mapping[str, Any] = field(default_factory=dict)
    composes_with: Tuple[str, ...] = ()
    default_skill: bool = False


@dataclass(frozen=True)
class SkillSpec:
    skill_id: str
    title: str
    markdown: str
    path: str
    workflow: WorkflowTemplate
    intents: Tuple[str, ...] = ()
    header: Optional[SkillHeader] = None

    @property
    def manual_id(self) -> str:
        return self.skill_id

    @property
    def name(self) -> str:
        return self.title

    @property
    def recipes(self) -> Tuple[WorkflowTemplate, ...]:
        return (self.workflow,)

    @property
    def workflows(self) -> Tuple[WorkflowTemplate, ...]:
        return (self.workflow,)


# Backward-compatible names for code that still imports the old manual/recipe
# vocabulary.  The main harness path now uses SkillSpec/WorkflowTemplate.
ManualStepSpec = ToolCallTemplate
RecipeSpec = WorkflowTemplate
AnalysisManual = SkillSpec


def load_skill_specs(skills_root: str | Path = "skills") -> Dict[str, SkillSpec]:
    root = Path(skills_root)
    if not root.exists():
        return {}

    skills: Dict[str, SkillSpec] = {}
    for skill_file in sorted(root.glob("*/SKILL.md")):
        try:
            markdown = skill_file.read_text(encoding="utf-8")
        except OSError:
            continue
        fallback_id = skill_file.parent.name
        header = _parse_skill_header(markdown, fallback_id=fallback_id)
        skill_id = header.skill_id if header is not None else (_metadata_value(markdown, "ID") or fallback_id)
        title = header.title if header is not None else (_metadata_value(markdown, "Name") or skill_id.replace("_", " ").title())
        intents = _skill_intents(skill_id, markdown, header=header)
        workflow = _workflow_for_skill(skill_id, markdown, intents=intents, header=header)
        skills[skill_id] = SkillSpec(
            skill_id=skill_id,
            title=title,
            markdown=markdown,
            path=str(skill_file),
            workflow=workflow,
            intents=intents,
            header=header,
        )
    return skills


def load_analysis_manuals(skills_root: str | Path = "skills") -> Dict[str, SkillSpec]:
    """Compatibility wrapper. Prefer ``load_skill_specs``."""
    return load_skill_specs(skills_root)


def retrieve_skill_specs(
    query: str,
    *,
    skills_root: str | Path = "skills",
    limit: int = 4,
) -> Dict[str, SkillSpec]:
    skills = load_skill_specs(skills_root)
    if not skills:
        return {}
    query_tokens = _tokens(query)
    scored: List[Tuple[int, str]] = []
    for skill_id, skill in skills.items():
        header_text = ""
        if skill.header is not None:
            header_text = " ".join(
                [
                    skill.header.title,
                    skill.header.category,
                    skill.header.description,
                    skill.header.input_intent,
                    skill.header.output_intent,
                    " ".join(skill.header.avoid_when),
                    " ".join(skill.header.intents),
                    " ".join(skill.header.variables),
                    " ".join(skill.header.composes_with),
                ]
            )
        haystack = _tokens(
            skill_id.replace("_", " ")
            + " "
            + " ".join(skill.intents)
            + " "
            + header_text
            + " "
            + skill.markdown[:5000]
        )
        overlap = len(query_tokens & haystack)
        intent_bonus = 3 if _workflow_intent_matches(query, skill.workflow) else 0
        score = overlap + intent_bonus
        if score:
            scored.append((score, skill_id))
    scored.sort(reverse=True)
    return {skill_id: skills[skill_id] for _, skill_id in scored[: max(1, limit)]}


def retrieve_analysis_manuals(
    query: str,
    *,
    skills_root: str | Path = "skills",
    limit: int = 4,
) -> Dict[str, SkillSpec]:
    """Compatibility wrapper. Prefer ``retrieve_skill_specs``."""
    return retrieve_skill_specs(query, skills_root=skills_root, limit=limit)


def select_skill_workflow(
    query: str,
    *,
    skills: Optional[Mapping[str, SkillSpec]] = None,
    skills_root: str | Path = "skills",
) -> Optional[WorkflowTemplate]:
    available = dict(skills or load_skill_specs(skills_root))
    if not available:
        return None
    full_available: Optional[Dict[str, SkillSpec]] = None

    def get_skill(skill_id: str) -> Optional[SkillSpec]:
        nonlocal full_available
        skill = available.get(skill_id)
        if skill is not None:
            return skill
        if full_available is None:
            full_available = load_skill_specs(skills_root)
        return full_available.get(skill_id)

    lowered = query.lower()
    for skill_id in sorted(available):
        if re.search(rf"\b{re.escape(skill_id.lower())}\b", lowered):
            return available[skill_id].workflow

    if _looks_like_watermass_event_association_query(lowered):
        skill = get_skill("ocean_watermass_event_association")
        if skill is not None:
            return skill.workflow

    if _looks_like_event_frequency_query(lowered):
        event_type = infer_event_type(query)
        skill = get_skill("ocean_event_frequency_map")
        if skill is not None and event_type is not None:
            return _event_frequency_recipe(event_type)

    if _looks_like_lag_correlation_query(lowered):
        skill = get_skill("ocean_lag_correlation")
        if skill is not None:
            return skill.workflow

    if _looks_like_heatwave_query(lowered):
        skill = get_skill("ocean_heatwave_detection")
        if skill is not None:
            return skill.workflow

    if _looks_like_hypoxia_detection_query(lowered):
        skill = get_skill("ocean_hypoxia_detection")
        if skill is not None:
            return skill.workflow

    if _looks_like_bloom_detection_query(lowered):
        skill = get_skill("ocean_bloom_detection")
        if skill is not None:
            return skill.workflow

    for skill in retrieve_skill_specs(query, skills_root=skills_root).values():
        if _workflow_has_strong_executable_match(query, skill.workflow):
            return skill.workflow
    return None


def select_manual_recipe(
    query: str,
    *,
    manuals: Optional[Mapping[str, SkillSpec]] = None,
    skills_root: str | Path = "skills",
) -> Optional[WorkflowTemplate]:
    """Compatibility wrapper. Prefer ``select_skill_workflow``."""
    return select_skill_workflow(query, skills=manuals, skills_root=skills_root)


def infer_event_type(query: str) -> Optional[str]:
    lowered = query.lower()
    if re.search(r"\b(heatwave|marine heatwave|warm event)\b|热浪|高温事件", lowered):
        return "heatwave"
    if re.search(r"\b(hypoxia|hypoxic|low oxygen|oxygen depletion)\b|缺氧|低氧|溶解氧不足", lowered):
        return "hypoxia"
    if re.search(r"\b(algal bloom|bloom|chlorophyll|chla)\b|藻华|水华|叶绿素", lowered):
        return "algal_bloom"
    if re.search(r"\b(upwelling)\b|上升流", lowered):
        return "upwelling"
    if re.search(r"\b(eutrophication|eutrophic)\b|富营养", lowered):
        return "eutrophication"
    return None


def parse_skill_frontmatter(markdown: str, *, fallback_id: str = "") -> Optional[SkillHeader]:
    return _parse_skill_header(markdown, fallback_id=fallback_id)


def parse_workflow_steps(
    markdown: str,
    *,
    skill_id: str = "",
    manual_id: str = "",
) -> Tuple[ToolCallTemplate, ...]:
    steps, _planner_parameters = _parse_workflow_blocks(markdown, skill_id=skill_id, manual_id=manual_id)
    return steps


def parse_workflow_planner_parameters(
    markdown: str,
    *,
    skill_id: str = "",
    manual_id: str = "",
) -> Mapping[str, Mapping[str, Any]]:
    _steps, planner_parameters = _parse_workflow_blocks(markdown, skill_id=skill_id, manual_id=manual_id)
    return planner_parameters


def _parse_workflow_blocks(
    markdown: str,
    *,
    skill_id: str = "",
    manual_id: str = "",
) -> Tuple[Tuple[ToolCallTemplate, ...], Mapping[str, Mapping[str, Any]]]:
    context_id = skill_id or manual_id
    steps: List[ToolCallTemplate] = []
    known_artifacts: set[str] = set()
    planner_symbols: Dict[str, Any] = {}
    planner_parameters: Dict[str, Mapping[str, Any]] = {}
    for block in _python_code_blocks(markdown):
        try:
            module = ast.parse(block)
        except SyntaxError as exc:
            raise ValueError(f"Invalid workflow Python block in {context_id or 'skill'}: {exc}") from exc
        for stmt in module.body:
            if isinstance(stmt, ast.Assign):
                if not _is_workflow_tool_assignment(stmt):
                    name, parameter = _workflow_assign_to_planner_parameter(
                        stmt,
                        known_artifacts=known_artifacts,
                        planner_symbols=planner_symbols,
                        skill_id=context_id,
                        source_block=block,
                    )
                    planner_symbols[name] = parameter.get("template")
                    planner_parameters[name] = parameter
                    continue
                step = _workflow_assign_to_step(
                    stmt,
                    known_artifacts=known_artifacts,
                    planner_symbols=planner_symbols,
                    skill_id=context_id,
                    source_block=block,
                )
                steps.append(step)
                known_artifacts.add(step.save_as)
                continue
            if isinstance(stmt, ast.Pass):
                continue
            raise ValueError(
                f"Workflow blocks may only contain assignment-style tool calls; "
                f"planner parameter declarations; got {type(stmt).__name__} in {context_id or 'skill'}"
            )
    return tuple(steps), planner_parameters


def _parse_skill_header(markdown: str, *, fallback_id: str = "") -> Optional[SkillHeader]:
    frontmatter, _body = _split_frontmatter(markdown)
    if frontmatter is None:
        return None
    try:
        payload = yaml.safe_load(frontmatter) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid skill frontmatter for {fallback_id or 'skill'}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"Skill frontmatter must be a mapping for {fallback_id or 'skill'}")

    skill_id = str(payload.get("skill_id") or payload.get("id") or fallback_id).strip()
    title = str(payload.get("title") or skill_id.replace("_", " ").title()).strip()
    mask_support = dict(payload.get("mask_support") or {}) if isinstance(payload.get("mask_support") or {}, Mapping) else {}
    mask_support.setdefault("accepts_analysis_mask", False)
    mask_support.setdefault("can_build_masks", [])
    return SkillHeader(
        skill_id=skill_id,
        title=title,
        category=str(payload.get("category") or "").strip(),
        description=str(payload.get("description") or "").strip(),
        input_intent=str(payload.get("input_intent") or "").strip(),
        output_intent=str(payload.get("output_intent") or "").strip(),
        avoid_when=tuple(_string_list(payload.get("avoid_when"))),
        intents=tuple(_string_list(payload.get("intents"))),
        variables=tuple(_string_list(payload.get("variables"))),
        requires=tuple(_string_list(payload.get("requires"))),
        defaults=dict(payload.get("defaults") or {}) if isinstance(payload.get("defaults") or {}, Mapping) else {},
        produces=tuple(item for item in _mapping_list(payload.get("produces"))),
        mask_support=mask_support,
        composes_with=tuple(_string_list(payload.get("composes_with"))),
        default_skill=bool(payload.get("default_skill", False)),
    )


def _split_frontmatter(markdown: str) -> Tuple[Optional[str], str]:
    if not markdown.startswith("---"):
        return None, markdown
    match = re.match(r"^---\s*\n(?P<head>.*?)\n---\s*(?P<body>.*)\Z", markdown, flags=re.DOTALL)
    if not match:
        raise ValueError("Skill frontmatter starts with --- but is not closed")
    return match.group("head"), match.group("body")


def _python_code_blocks(markdown: str) -> Iterable[str]:
    for match in re.finditer(r"```(?:python|py)\s*\n(?P<code>.*?)\n```", markdown, flags=re.IGNORECASE | re.DOTALL):
        code = match.group("code").strip()
        if code:
            yield code


def _workflow_assign_to_step(
    stmt: ast.Assign,
    *,
    known_artifacts: set[str],
    planner_symbols: Mapping[str, Any],
    skill_id: str,
    source_block: str,
) -> ToolCallTemplate:
    if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
        raise ValueError(f"Workflow assignment target must be one artifact name in {skill_id or 'skill'}")
    save_as = stmt.targets[0].id
    if not isinstance(stmt.value, ast.Call) or not isinstance(stmt.value.func, ast.Name):
        raise ValueError(f"Workflow assignment must call a simple tool function in {skill_id or 'skill'}")
    tool = stmt.value.func.id
    if not _is_safe_tool_name(tool):
        raise ValueError(f"Invalid workflow tool name: {tool}")
    params: Dict[str, Any] = {}
    param_docs: Dict[str, str] = {}
    if stmt.value.args:
        raise ValueError(f"Workflow tool calls must use keyword arguments only: {tool}")
    for keyword in stmt.value.keywords:
        if keyword.arg is None:
            raise ValueError("Workflow tool calls do not allow **kwargs")
        params[keyword.arg] = _ast_value_to_template(
            keyword.value,
            known_artifacts=known_artifacts,
            planner_symbols=planner_symbols,
        )
        comment = _workflow_keyword_comment(source_block, keyword)
        if comment:
            param_docs[keyword.arg] = comment
    return ToolCallTemplate(
        title=save_as.replace("_", " ").title(),
        tool=tool,
        params_template=params,
        save_as=save_as,
        param_docs=param_docs,
        output_artifact=_workflow_output_artifact(tool, save_as),
    )


def _is_workflow_tool_assignment(stmt: ast.Assign) -> bool:
    return (
        len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
        and isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.func, ast.Name)
    )


def _workflow_assign_to_planner_parameter(
    stmt: ast.Assign,
    *,
    known_artifacts: set[str],
    planner_symbols: Mapping[str, Any],
    skill_id: str,
    source_block: str,
) -> Tuple[str, Mapping[str, Any]]:
    if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
        raise ValueError(f"Workflow planner parameter target must be one name in {skill_id or 'skill'}")
    name = stmt.targets[0].id
    if not _is_safe_tool_name(name):
        raise ValueError(f"Invalid workflow planner parameter name: {name}")
    comment = _workflow_assignment_comment(source_block, stmt)
    template = _planner_parameter_template(
        name,
        stmt.value,
        known_artifacts=known_artifacts,
        planner_symbols=planner_symbols,
        comment=comment,
    )
    parameter: Dict[str, Any] = {"template": template}
    default = _literal_parameter_default(stmt.value)
    if default is not _NO_PARAMETER_DEFAULT:
        parameter["default"] = default
    if comment:
        parameter["doc"] = comment
    return name, parameter


_NO_PARAMETER_DEFAULT = object()


def _planner_parameter_template(
    name: str,
    node: ast.AST,
    *,
    known_artifacts: set[str],
    planner_symbols: Mapping[str, Any],
    comment: str = "",
) -> Any:
    if _planner_comment_marks_modifiable(comment):
        placeholder = _modifiable_parameter_placeholder(name, node)
        if placeholder is not None:
            return placeholder
    if isinstance(node, ast.Constant) and node.value is None:
        return _placeholder_for_name(name)
    return _ast_value_to_template(node, known_artifacts=known_artifacts, planner_symbols=planner_symbols)


def _planner_comment_marks_modifiable(comment: str) -> bool:
    return "modify" in comment.lower()


def _modifiable_parameter_placeholder(name: str, node: ast.AST) -> Optional[Any]:
    if name == "variables":
        length = len(node.elts) if isinstance(node, (ast.List, ast.Tuple)) else 1
        length = max(length, 1)
        if length == 1:
            return ["{variable}"]
        return ["{" + f"variable{index + 1}" + "}" for index in range(length)]
    if name in {
        "lon_range",
        "lat_range",
        "time_range",
        "depth_range",
        "depth_value",
        "vertical_mode",
        "depth_aggregation",
        "time_aggregation",
        "season_filter",
        "diagnostic_type",
        "field_type",
        "output_mode",
        "mask_polygon",
        "mask_isobath_depth",
        "mask_isobath_comparison",
        "diagram_type",
        "aggregate_dim",
        "spatial_weighting",
        "weighting",
        "regional_gauge",
        "transect_points",
        "n_samples",
        "method",
        "layer_bounds",
        "transport_type",
        "rho0",
        "cp",
        "s_ref",
        "normalize",
        "period",
        "climatology_time_range",
        "max_lag",
        "threshold",
        "oxygen_threshold",
        "severe_threshold",
        "percentile_threshold",
        "min_duration_days",
        "min_area_km2",
        "resolution_deg",
    }:
        return _placeholder_for_name(name)
    if isinstance(node, ast.Constant) and node.value is None:
        return _placeholder_for_name(name)
    return None


def _literal_parameter_default(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        values = [_literal_parameter_default(item) for item in node.elts]
        if all(value is not _NO_PARAMETER_DEFAULT for value in values):
            return values
    if isinstance(node, ast.Tuple):
        values = [_literal_parameter_default(item) for item in node.elts]
        if all(value is not _NO_PARAMETER_DEFAULT for value in values):
            return values
    return _NO_PARAMETER_DEFAULT


def _workflow_assignment_comment(source_block: str, stmt: ast.Assign) -> str:
    start_line = getattr(stmt, "lineno", 0)
    end_line = getattr(stmt, "end_lineno", start_line)
    comments: List[str] = []

    try:
        tokens = tokenize.generate_tokens(io.StringIO(source_block).readline)
        for token in tokens:
            if token.type != tokenize.COMMENT:
                continue
            line_no = token.start[0]
            if start_line <= line_no <= end_line:
                text = token.string.lstrip("#").strip()
                if text:
                    comments.append(text)
    except tokenize.TokenError:
        comments = []

    if comments:
        return " ".join(comments)

    lines = source_block.splitlines()
    preceding: List[str] = []
    index = start_line - 2
    while index >= 0:
        stripped = lines[index].strip()
        if not stripped:
            break
        if not stripped.startswith("#"):
            break
        preceding.append(stripped.lstrip("#").strip())
        index -= 1
    preceding.reverse()
    return " ".join(item for item in preceding if item)


def _workflow_keyword_comment(source_block: str, keyword: ast.keyword) -> str:
    start_line = getattr(keyword, "lineno", 0)
    end_line = getattr(keyword, "end_lineno", start_line)
    comments: List[str] = []

    try:
        tokens = tokenize.generate_tokens(io.StringIO(source_block).readline)
        for token in tokens:
            if token.type != tokenize.COMMENT:
                continue
            line_no = token.start[0]
            if start_line <= line_no <= end_line:
                text = token.string.lstrip("#").strip()
                if text:
                    comments.append(text)
    except tokenize.TokenError:
        comments = []

    if comments:
        return " ".join(comments)

    lines = source_block.splitlines()
    preceding: List[str] = []
    index = start_line - 2
    while index >= 0:
        stripped = lines[index].strip()
        if not stripped:
            break
        if not stripped.startswith("#"):
            break
        preceding.append(stripped.lstrip("#").strip())
        index -= 1
    preceding.reverse()
    return " ".join(item for item in preceding if item)


def _ast_value_to_template(
    node: ast.AST,
    *,
    known_artifacts: set[str],
    planner_symbols: Optional[Mapping[str, Any]] = None,
) -> Any:
    planner_symbols = planner_symbols or {}
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in known_artifacts:
            return f"$ref:{node.id}"
        if node.id in planner_symbols:
            return planner_symbols[node.id]
        return _placeholder_for_name(node.id)
    if isinstance(node, ast.Attribute):
        path = _attribute_path(node)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+", path):
            raise ValueError(f"Only artifact field access is allowed in workflow refs: {path}")
        return f"$ref:{path}"
    if isinstance(node, ast.Subscript):
        indexed = _indexed_planner_value(node, known_artifacts=known_artifacts, planner_symbols=planner_symbols)
        if indexed is not _NO_INDEXED_VALUE:
            return indexed
    if isinstance(node, ast.List):
        return [
            _ast_value_to_template(item, known_artifacts=known_artifacts, planner_symbols=planner_symbols)
            for item in node.elts
        ]
    if isinstance(node, ast.Tuple):
        return [
            _ast_value_to_template(item, known_artifacts=known_artifacts, planner_symbols=planner_symbols)
            for item in node.elts
        ]
    if isinstance(node, ast.Dict):
        result: Dict[str, Any] = {}
        for key_node, value_node in zip(node.keys, node.values):
            if key_node is None:
                raise ValueError("Workflow dict unpacking is not allowed")
            key = _ast_value_to_template(key_node, known_artifacts=known_artifacts, planner_symbols=planner_symbols)
            if not isinstance(key, str):
                raise ValueError("Workflow dict keys must be strings")
            result[key] = _ast_value_to_template(
                value_node,
                known_artifacts=known_artifacts,
                planner_symbols=planner_symbols,
            )
        return result
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _ast_value_to_template(node.operand, known_artifacts=known_artifacts, planner_symbols=planner_symbols)
        if isinstance(value, (int, float)):
            return -value
    raise ValueError(f"Unsupported workflow parameter syntax: {type(node).__name__}")


_NO_INDEXED_VALUE = object()


def _indexed_planner_value(
    node: ast.Subscript,
    *,
    known_artifacts: set[str],
    planner_symbols: Mapping[str, Any],
) -> Any:
    if not isinstance(node.value, ast.Name):
        return _NO_INDEXED_VALUE
    index = _constant_subscript_index(node.slice)
    if not isinstance(index, int):
        return _NO_INDEXED_VALUE

    name = node.value.id
    if name in planner_symbols:
        values = planner_symbols[name]
        if isinstance(values, (list, tuple)) and 0 <= index < len(values):
            return values[index]
    if name == "variables":
        if index == 0:
            return "{variable}"
        return "{" + f"variable{index + 1}" + "}"
    if name.endswith("_ranges"):
        return "{" + f"{name}[{index}]" + "}"
    return _NO_INDEXED_VALUE


def _constant_subscript_index(node: ast.AST) -> Optional[int]:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    return None


def _attribute_path(node: ast.Attribute) -> str:
    parts: List[str] = []
    current: ast.AST = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        raise ValueError("Workflow refs only support artifact.field access")
    parts.append(current.id)
    return ".".join(reversed(parts))


def _placeholder_for_name(name: str) -> str:
    if name == "lon_range":
        return "{region.lon_range}"
    if name == "lat_range":
        return "{region.lat_range}"
    if name in {
        "time_range",
        "depth_range",
        "depth_value",
        "vertical_mode",
        "depth_aggregation",
        "variable",
        "variable1",
        "variable2",
        "depth_range1",
        "depth_range2",
        "max_lag",
        "threshold",
        "oxygen_threshold",
        "severe_threshold",
        "percentile_threshold",
        "min_duration_days",
        "min_area_km2",
        "resolution_deg",
        "normalize",
    }:
        return "{" + name + "}"
    return "{" + name + "}"


def _workflow_output_artifact(tool: str, save_as: str) -> Mapping[str, Any]:
    kind = "generic"
    dims: List[str] = []
    frontend_type = "generic_result"
    if tool == "load_dataset":
        kind, dims, frontend_type = "field", ["time", "depth", "lat", "lon"], "data_container_result"
    elif tool == "select_vertical":
        kind, dims, frontend_type = "field", ["time", "lat", "lon"], "data_container_result"
    elif tool in {"build_threshold_mask", "build_condition_mask"}:
        kind, dims, frontend_type = "mask", ["time", "lat", "lon"], "data_container_result"
    elif tool in {"build_polygon_mask", "build_isobath_mask", "combine_masks"}:
        kind, dims, frontend_type = "mask", ["lat", "lon"], "data_container_result"
    elif tool in {
        "assemble_dataset",
        "apply_mask",
        "compute_speed_from_uv",
        "compute_density",
        "compute_derived_field",
        "compute_stratification_index",
        "compute_brunt_vaisala_frequency",
        "compute_field_anomaly",
        "compute_field_climatology",
        "compute_layer_mean",
        "compute_mixed_layer_mean",
        "compute_local_tendency",
        "compute_horizontal_advection",
        "compute_vertical_advection",
        "compute_front_proximity_index",
        "compute_eddy_influence_mask",
        "compute_tracer_gradient_alignment",
        "compute_mesoscale_background_separation",
        "compute_flow_structure_context",
        "filter_mesoscale_component",
        "replace_field_with_climatology",
    }:
        kind, dims, frontend_type = "field", ["time", "depth", "lat", "lon"], "data_container_result"
    elif tool.startswith("detect_"):
        kind, dims, frontend_type = "table", [], "generic_result"
    elif tool == "compute_watermass_event_association":
        kind, dims, frontend_type = "table", [], "generic_result"
    elif tool in {
        "compute_event_summary_map",
        "compute_event_frequency_map",
        "compute_spatial_field",
        "compute_spatial_vorticity_map",
        "compute_transport_streamfunction_map",
        "build_watermass_tile_map",
    }:
        kind, dims, frontend_type = "map", ["lat", "lon"], "spatial_field_result"
    elif tool in {
        "extract_timeseries",
        "extract_regional_mean",
        "compute_masked_mean_timeseries",
        "compute_masked_area_fraction_timeseries",
        "compute_tracer_horizontal_advection_timeseries",
        "compute_vertical_stability_timeseries",
        "compute_budget_residual",
    }:
        kind, dims, frontend_type = "series", ["time"], "timeseries_result"
    elif tool == "compare_budget_term_magnitudes":
        kind, dims, frontend_type = "table", [], "generic_result"
    elif tool in {"compute_hovmoller", "compute_transect_normal_flux_hovmoller"}:
        kind, dims, frontend_type = "hovmoller", ["time", "depth"], "hovmoller_result"
    elif tool == "compute_spectrum":
        kind, dims, frontend_type = "spectrum", ["frequency"], "spectrum_result"
    elif tool == "build_watermass_ts_diagram":
        kind, dims, frontend_type = "generic", [], "generic_result"
    elif tool == "compute_trend":
        kind, dims, frontend_type = "table", [], "trend_result"
    elif tool == "compute_field_trend":
        kind, dims, frontend_type = "map", ["lat", "lon"], "field_trend_result"
    return {"id": save_as, "kind": kind, "dims": dims, "frontend_type": frontend_type}


def _event_type_from_header(header: SkillHeader) -> Optional[str]:
    text = " ".join([header.skill_id, header.title, header.category, header.description, " ".join(header.intents)]).lower()
    if "heatwave" in text or "热浪" in text:
        return "heatwave"
    if "hypoxia" in text or "缺氧" in text or "low oxygen" in text:
        return "hypoxia"
    if "bloom" in text or "藻华" in text:
        return "algal_bloom"
    if "upwelling" in text or "上升流" in text:
        return "upwelling"
    if "eutrophication" in text or "富营养" in text:
        return "eutrophication"
    return None


def _final_artifacts_from_header(header: SkillHeader, steps: Tuple[ToolCallTemplate, ...]) -> Tuple[str, ...]:
    produced = tuple(str(item.get("id")) for item in header.produces if isinstance(item.get("id"), str))
    if produced:
        return produced
    return tuple(step.save_as for step in steps[-3:])


def _walk_template_values(value: Any) -> Iterable[Any]:
    yield value
    if isinstance(value, Mapping):
        for nested in value.values():
            yield from _walk_template_values(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _walk_template_values(nested)


def _workflow_mentions_template(steps: Tuple[ToolCallTemplate, ...], marker: str) -> bool:
    return any(
        isinstance(value, str) and marker in value
        for step in steps
        for value in _walk_template_values(step.params_template)
    )


def _workflow_variables_from_steps(steps: Tuple[ToolCallTemplate, ...]) -> List[str]:
    variables: List[str] = []
    for step in steps:
        if step.tool == "load_dataset":
            value = step.params_template.get("variable")
            if isinstance(value, str) and value and not value.startswith("{") and value not in variables:
                variables.append(value)
        for tool, variable in (
            ("detect_heatwaves", "temp"),
            ("detect_hypoxia", "oxygen"),
            ("detect_algal_blooms", "chlorophyll"),
            ("detect_upwelling", "temp"),
            ("detect_eutrophication", "chlorophyll"),
        ):
            if step.tool == tool and variable not in variables:
                variables.append(variable)
    return variables


def _defaults_from_workflow(header: SkillHeader, steps: Tuple[ToolCallTemplate, ...]) -> Mapping[str, Any]:
    defaults: Dict[str, Any] = dict(header.defaults)
    tools = {step.tool for step in steps}
    if "depth_aggregation" not in defaults and any(tool.startswith("detect_") for tool in tools):
        defaults["depth_aggregation"] = "mean"
    if "vertical_mode" not in defaults:
        if "detect_hypoxia" in tools:
            defaults["vertical_mode"] = "bottom"
        elif tools.intersection({"detect_heatwaves", "detect_algal_blooms", "detect_upwelling", "detect_eutrophication"}):
            defaults["vertical_mode"] = "surface"
            defaults.setdefault("depth_range", [0, 0])
    if "compute_event_frequency_map" in tools:
        defaults.setdefault("normalize", False)
    return defaults


def _required_inputs_from_header(header: SkillHeader, steps: Tuple[ToolCallTemplate, ...]) -> Mapping[str, Any]:
    variables = list(header.variables) or _workflow_variables_from_steps(steps)
    defaults = _defaults_from_workflow(header, steps)
    required: Dict[str, Any] = {"variables": variables}
    if any(item in header.requires for item in ("lon_range", "lat_range", "region")) or (
        _workflow_mentions_template(steps, "{region.lon_range}") and _workflow_mentions_template(steps, "{region.lat_range}")
    ):
        required["region"] = "required"
    if "time_range" in header.requires or _workflow_mentions_template(steps, "{time_range}"):
        required["time_range"] = "required"
    if defaults.get("vertical_mode") is not None:
        required["vertical"] = defaults.get("vertical_mode")
    return required


def _output_policy_from_header(header: SkillHeader, steps: Tuple[ToolCallTemplate, ...]) -> Mapping[str, Any]:
    artifacts = _final_artifacts_from_header(header, steps)
    if not artifacts:
        return {}
    return {"primary": artifacts[-1], "secondary": list(artifacts[:-1])}


def _validation_rules_from_header(header: SkillHeader, steps: Tuple[ToolCallTemplate, ...]) -> Tuple[str, ...]:
    rules: List[str] = []
    if any(item in header.requires for item in ("lon_range", "lat_range", "region")) or (
        _workflow_mentions_template(steps, "{region.lon_range}") and _workflow_mentions_template(steps, "{region.lat_range}")
    ):
        rules.append("requires_region")
    if "time_range" in header.requires or _workflow_mentions_template(steps, "{time_range}"):
        rules.append("requires_time_dimension")
    if header.variables or _workflow_variables_from_steps(steps):
        rules.append("requires_variable")
    return tuple(rules)


def _string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item is not None]
    return []


def _mapping_list(value: Any) -> List[Mapping[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _is_safe_tool_name(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value))


def _workflow_for_skill(
    skill_id: str,
    markdown: str,
    *,
    intents: Tuple[str, ...],
    header: Optional[SkillHeader] = None,
) -> WorkflowTemplate:
    workflows = tuple(_workflows_for_skill(skill_id, markdown, intents=intents, header=header))
    if workflows:
        return workflows[0]
    return WorkflowTemplate(
        workflow_id=f"{skill_id}_workflow",
        skill_id=skill_id,
        intents=intents,
        mode="manual_context_only",
        steps=(),
    )


def _workflows_for_skill(
    skill_id: str,
    markdown: str,
    *,
    intents: Tuple[str, ...],
    header: Optional[SkillHeader] = None,
) -> Iterable[WorkflowTemplate]:
    if header is not None:
        if skill_id == "ocean_lag_correlation":
            yield _lag_correlation_recipe()
            return

        steps, planner_parameters = _parse_workflow_blocks(markdown, skill_id=skill_id)
        if skill_id == "ocean_heatwave_detection" and not steps:
            steps = _heatwave_steps()
        elif skill_id == "ocean_hypoxia_detection" and not steps:
            steps = _hypoxia_steps()

        mode = "tool_workflow" if steps else "manual_context_only"
        yield WorkflowTemplate(
            workflow_id=f"{skill_id}_workflow",
            skill_id=skill_id,
            intents=tuple(header.intents or intents),
            event_type=_event_type_from_header(header),
            mode=mode,
            steps=tuple(steps),
            planner_parameters=dict(planner_parameters),
            final_artifacts=_final_artifacts_from_header(header, steps),
            defaults=_defaults_from_workflow(header, tuple(steps)),
            required_inputs=_required_inputs_from_header(header, tuple(steps)),
            output_policy=_output_policy_from_header(header, tuple(steps)),
            validation_rules=_validation_rules_from_header(header, tuple(steps)),
        )
        return

    structured = _parse_analysis_manual_block(skill_id, markdown)
    if structured:
        yield from structured
        return

    if skill_id == "ocean_heatwave_detection":
        steps = _parse_execution_steps(markdown) or _heatwave_steps()
        yield WorkflowTemplate(
            workflow_id="marine_heatwave_detection",
            skill_id=skill_id,
            intents=("detect_heatwaves", "marine_heatwave_summary", "heatwave_burden"),
            event_type="heatwave",
            steps=tuple(steps),
            final_artifacts=("marine_heatwave_burden", "marine_heatwave_days", "heatwave_detection"),
            defaults={
                "vertical_mode": "surface",
                "depth_range": [0, 0],
                "depth_aggregation": "mean",
            },
            required_inputs={
                "variables": ["temp"],
                "region": "required",
                "time_range": "required",
                "vertical": "surface by default; user semantic override allowed",
            },
            output_policy={"primary": "marine_heatwave_burden", "secondary": ["marine_heatwave_days"]},
            validation_rules=("requires_time_dimension", "requires_region", "requires_temperature"),
        )
    elif skill_id == "ocean_hypoxia_detection":
        steps = _parse_execution_steps(markdown) or _hypoxia_steps()
        yield WorkflowTemplate(
            workflow_id="hypoxia_detection",
            skill_id=skill_id,
            intents=("detect_hypoxia", "hypoxia_summary", "hypoxia_burden"),
            event_type="hypoxia",
            steps=tuple(steps),
            final_artifacts=("hypoxia_oxygen_deficit_burden", "hypoxic_days", "hypoxia_detection"),
            defaults={
                "vertical_mode": "bottom",
                "depth_aggregation": "mean",
            },
            required_inputs={
                "variables": ["oxygen"],
                "region": "required",
                "time_range": "required",
                "vertical": "bottom by default; local per-cell bottom semantics",
            },
            output_policy={"primary": "hypoxia_oxygen_deficit_burden", "secondary": ["hypoxic_days"]},
            validation_rules=("requires_time_or_single_snapshot", "requires_region", "requires_oxygen"),
        )
    elif skill_id == "ocean_event_frequency_map":
        for event_type in ("heatwave", "hypoxia", "algal_bloom", "upwelling", "eutrophication"):
            yield _event_frequency_recipe(event_type)
    elif skill_id == "ocean_lag_correlation":
        yield _lag_correlation_recipe()


def _recipes_for_manual(
    manual_id: str,
    markdown: str,
    *,
    intents: Tuple[str, ...],
    header: Optional[SkillHeader] = None,
) -> Iterable[WorkflowTemplate]:
    """Compatibility wrapper for the previous internal helper name."""
    yield from _workflows_for_skill(manual_id, markdown, intents=intents, header=header)


def _parse_execution_steps(markdown: str) -> Tuple[ToolCallTemplate, ...]:
    pattern = re.compile(
        r"###\s+Step\s+[^\n:]*:?\s*(?P<title>[^\n]+).*?"
        r"\*\*Tool\*\*:\s*`(?P<tool>[^`]+)`.*?"
        r"(?:\*\*Parameters\*\*:\s*)?```json\s*(?P<params>.*?)\s*```.*?"
        r"\*\*Save As\*\*:\s*`(?P<save_as>[^`]+)`",
        flags=re.IGNORECASE | re.DOTALL,
    )
    steps: List[ToolCallTemplate] = []
    for match in pattern.finditer(markdown):
        tool = match.group("tool").strip()
        save_as = match.group("save_as").strip()
        if "{" in tool or "}" in tool or "," in save_as:
            continue
        try:
            params = json.loads(match.group("params"))
        except json.JSONDecodeError:
            continue
        if not isinstance(params, dict):
            continue
        steps.append(
            ToolCallTemplate(
                title=match.group("title").strip(),
                tool=tool,
                params_template=params,
                save_as=save_as,
            )
        )
    return tuple(steps)


def _parse_analysis_manual_block(manual_id: str, markdown: str) -> Tuple[WorkflowTemplate, ...]:
    match = re.search(
        r"##\s+Analysis Manual\s*```json\s*(?P<payload>.*?)\s*```",
        markdown,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return ()
    try:
        payload = json.loads(match.group("payload"))
    except json.JSONDecodeError:
        return ()
    workflows_payload = payload.get("recipes")
    if not isinstance(workflows_payload, list):
        return ()

    workflows: List[WorkflowTemplate] = []
    for workflow_payload in workflows_payload:
        if not isinstance(workflow_payload, Mapping):
            continue
        steps_payload = workflow_payload.get("steps")
        if not isinstance(steps_payload, list):
            steps_payload = []
        steps: List[ToolCallTemplate] = []
        for step_payload in steps_payload:
            if not isinstance(step_payload, Mapping):
                continue
            tool = step_payload.get("tool")
            save_as = step_payload.get("save_as")
            if not isinstance(tool, str) or not isinstance(save_as, str):
                continue
            params = step_payload.get("params") if isinstance(step_payload.get("params"), Mapping) else {}
            steps.append(
                ToolCallTemplate(
                    title=str(step_payload.get("title") or save_as),
                    tool=tool,
                    params_template=dict(params),
                    save_as=save_as,
                    input_artifacts=dict(step_payload.get("input_artifacts") or {}),
                    output_artifact=dict(step_payload.get("output_artifact") or {}),
                )
            )
        execution_mode = str(workflow_payload.get("execution_mode") or "tool_recipe")
        if not steps and execution_mode != "manual_context_only":
            continue
        workflows.append(
            WorkflowTemplate(
                workflow_id=str(workflow_payload.get("recipe_id") or f"{manual_id}_recipe"),
                skill_id=str(workflow_payload.get("manual_id") or manual_id),
                intents=tuple(str(item) for item in workflow_payload.get("intents", []) if isinstance(item, str)),
                event_type=workflow_payload.get("event_type") if isinstance(workflow_payload.get("event_type"), str) else None,
                mode=execution_mode,
                steps=tuple(steps),
                analysis_steps=tuple(
                    item
                    for item in workflow_payload.get("analysis_steps", [])
                    if isinstance(item, Mapping)
                ),
                final_artifacts=tuple(str(item) for item in workflow_payload.get("final_artifacts", []) if isinstance(item, str)),
                defaults=dict(workflow_payload.get("default_params") or {}),
                required_inputs=dict(workflow_payload.get("required_inputs") or {}),
                output_policy=dict(workflow_payload.get("output_policy") or {}),
                validation_rules=tuple(str(item) for item in workflow_payload.get("validation_rules", []) if isinstance(item, str)),
            )
        )
    return tuple(workflows)


def _lag_correlation_recipe() -> WorkflowTemplate:
    steps = (
        ToolCallTemplate(
            "Load first variable",
            "load_dataset",
            {
                "variable": "{variable1}",
                "lon_range": "{region.lon_range}",
                "lat_range": "{region.lat_range}",
                "time_range": "{time_range}",
                "depth_range": "{depth_range1}",
            },
            "raw_data1",
        ),
        ToolCallTemplate(
            "Load second variable",
            "load_dataset",
            {
                "variable": "{variable2}",
                "lon_range": "{region.lon_range}",
                "lat_range": "{region.lat_range}",
                "time_range": "{time_range}",
                "depth_range": "{depth_range2}",
            },
            "raw_data2",
        ),
        ToolCallTemplate(
            "Extract regional mean time series for variable 1",
            "extract_timeseries",
            {
                "data": "$ref:raw_data1.data",
                "lon_range": "{region.lon_range}",
                "lat_range": "{region.lat_range}",
                "spatial_aggregation": "mean",
                "depth_aggregation": "mean",
            },
            "timeseries1",
        ),
        ToolCallTemplate(
            "Extract regional mean time series for variable 2",
            "extract_timeseries",
            {
                "data": "$ref:raw_data2.data",
                "lon_range": "{region.lon_range}",
                "lat_range": "{region.lat_range}",
                "spatial_aggregation": "mean",
                "depth_aggregation": "mean",
            },
            "timeseries2",
        ),
        ToolCallTemplate(
            "Raw lag correlation",
            "compute_lag_correlation",
            {
                "timeseries1": "$ref:timeseries1",
                "timeseries2": "$ref:timeseries2",
                "max_lag": "{max_lag}",
                "confidence_level": 0.95,
            },
            "lag_correlation_raw",
        ),
    )
    return WorkflowTemplate(
        workflow_id="lag_correlation_raw",
        skill_id="ocean_lag_correlation",
        intents=("lag correlation", "lead lag", "time lag", "cross correlation", "related with time lags"),
        steps=steps,
        final_artifacts=("timeseries1", "timeseries2", "lag_correlation_raw"),
        defaults={"max_lag": 12, "seasonality_mode": "raw"},
        required_inputs={
            "variables": ["variable1", "variable2"],
            "region": "required",
            "time_range": "required",
            "vertical": "per variable from user semantics",
        },
        output_policy={"primary": "lag_correlation_raw", "secondary": ["timeseries1", "timeseries2"]},
        validation_rules=("requires_region", "requires_time_dimension", "requires_two_variables"),
    )


def _event_frequency_recipe(event_type: str) -> WorkflowTemplate:
    variable, detect_tool, field_param, defaults = _event_recipe_parts(event_type)
    return WorkflowTemplate(
        workflow_id=f"{event_type}_frequency_map",
        skill_id="ocean_event_frequency_map",
        intents=("event_frequency_map", f"{event_type}_hotspot", f"{event_type}_frequency"),
        event_type=event_type,
        steps=(
            ToolCallTemplate(
                title="Load event field",
                tool="load_dataset",
                params_template={
                    "variable": variable,
                    "lon_range": "{region.lon_range}",
                    "lat_range": "{region.lat_range}",
                    "time_range": "{time_range}",
                    "vertical_mode": "{vertical_mode}",
                    "depth_value": "{depth_value}",
                    "depth_range": "{depth_range}",
                },
                save_as="event_field",
            ),
            ToolCallTemplate(
                title=f"Detect {event_type}",
                tool=detect_tool,
                params_template={
                    field_param: "$ref:event_field.data",
                    "vertical_mode": "{vertical_mode}",
                    "depth_value": "{depth_value}",
                    "depth_range": "{depth_range}",
                    "depth_aggregation": "{depth_aggregation}",
                },
                save_as="event_detection",
            ),
            ToolCallTemplate(
                title="Compute event frequency map",
                tool="compute_event_frequency_map",
                params_template={
                    "event_detection": "$ref:event_detection",
                    "data": "$ref:event_field.data",
                    "lon_range": "{region.lon_range}",
                    "lat_range": "{region.lat_range}",
                    "normalize": "{normalize}",
                },
                save_as="event_frequency_map",
            ),
        ),
        final_artifacts=("event_frequency_map", "event_detection"),
        defaults={**defaults, "normalize": False, "depth_aggregation": "mean"},
        required_inputs={
            "variables": [variable],
            "region": "required",
            "time_range": "required",
            "vertical": defaults.get("vertical_mode", "user semantic"),
        },
        output_policy={"primary": "event_frequency_map"},
        validation_rules=("requires_time_dimension", "requires_region", "requires_event_variable"),
    )


def _event_recipe_parts(event_type: str) -> Tuple[str, str, str, Mapping[str, Any]]:
    if event_type == "hypoxia":
        return "oxygen", "detect_hypoxia", "oxygen", {"vertical_mode": "bottom"}
    if event_type == "algal_bloom":
        return "chlorophyll", "detect_algal_blooms", "chlorophyll", {"vertical_mode": "surface", "depth_range": [0, 0]}
    if event_type == "upwelling":
        return "temp", "detect_upwelling", "temp", {"vertical_mode": "surface", "depth_range": [0, 0]}
    if event_type == "eutrophication":
        return "chlorophyll", "detect_eutrophication", "chlorophyll", {"vertical_mode": "surface", "depth_range": [0, 0]}
    return "temp", "detect_heatwaves", "temp", {"vertical_mode": "surface", "depth_range": [0, 0]}


def _heatwave_steps() -> Tuple[ToolCallTemplate, ...]:
    return (
        ToolCallTemplate(
            "Load temperature",
            "load_dataset",
            {
                "variable": "temp",
                "lon_range": "{region.lon_range}",
                "lat_range": "{region.lat_range}",
                "time_range": "{time_range}",
                "vertical_mode": "{vertical_mode}",
                "depth_value": "{depth_value}",
                "depth_range": "{depth_range}",
            },
            "temperature_field",
        ),
        ToolCallTemplate(
            "Detect heatwaves",
            "detect_heatwaves",
            {
                "temp": "$ref:temperature_field.data",
                "percentile_threshold": "{percentile_threshold}",
                "min_duration_days": "{min_duration_days}",
                "min_area_km2": "{min_area_km2}",
                "vertical_mode": "{vertical_mode}",
                "depth_value": "{depth_value}",
                "depth_range": "{depth_range}",
                "depth_aggregation": "{depth_aggregation}",
            },
            "heatwave_detection",
        ),
        ToolCallTemplate(
            "Summarize heatwave days",
            "compute_event_summary_map",
            {
                "event_detection": "$ref:heatwave_detection",
                "data": "$ref:temperature_field.data",
                "summary_mode": "event_days",
            },
            "marine_heatwave_days",
        ),
        ToolCallTemplate(
            "Summarize heatwave burden",
            "compute_event_summary_map",
            {
                "event_detection": "$ref:heatwave_detection",
                "data": "$ref:temperature_field.data",
                "summary_mode": "burden",
            },
            "marine_heatwave_burden",
        ),
    )


def _hypoxia_steps() -> Tuple[ToolCallTemplate, ...]:
    return (
        ToolCallTemplate(
            "Load oxygen",
            "load_dataset",
            {
                "variable": "oxygen",
                "lon_range": "{region.lon_range}",
                "lat_range": "{region.lat_range}",
                "time_range": "{time_range}",
                "vertical_mode": "{vertical_mode}",
                "depth_value": "{depth_value}",
                "depth_range": "{depth_range}",
            },
            "oxygen_field",
        ),
        ToolCallTemplate(
            "Detect hypoxia",
            "detect_hypoxia",
            {
                "oxygen": "$ref:oxygen_field.data",
                "oxygen_threshold": "{oxygen_threshold}",
                "severe_threshold": "{severe_threshold}",
                "min_area_km2": "{min_area_km2}",
                "min_duration_days": "{min_duration_days}",
                "vertical_mode": "{vertical_mode}",
                "depth_value": "{depth_value}",
                "depth_range": "{depth_range}",
                "depth_aggregation": "{depth_aggregation}",
            },
            "hypoxia_detection",
        ),
        ToolCallTemplate(
            "Summarize hypoxic days",
            "compute_event_summary_map",
            {
                "event_detection": "$ref:hypoxia_detection",
                "data": "$ref:oxygen_field.data",
                "summary_mode": "event_days",
            },
            "hypoxic_days",
        ),
        ToolCallTemplate(
            "Summarize hypoxia burden",
            "compute_event_summary_map",
            {
                "event_detection": "$ref:hypoxia_detection",
                "data": "$ref:oxygen_field.data",
                "summary_mode": "burden",
            },
            "hypoxia_oxygen_deficit_burden",
        ),
    )


def _metadata_value(markdown: str, key: str) -> Optional[str]:
    match = re.search(rf"-\s+\*\*{re.escape(key)}\*\*:\s*(.+)", markdown, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def _skill_intents(skill_id: str, markdown: str, *, header: Optional[SkillHeader] = None) -> Tuple[str, ...]:
    intents: List[str] = [skill_id.replace("_", " ")]
    if header is not None:
        intents.extend(
            [
                header.title,
                header.category,
                header.description,
                header.input_intent,
                header.output_intent,
                *header.avoid_when,
                *header.intents,
                *header.variables,
                *header.composes_with,
            ]
        )
    name = _metadata_value(markdown, "Name")
    category = _metadata_value(markdown, "Category")
    if name:
        intents.append(name)
    if category:
        intents.append(category)

    scenario_match = re.search(
        r"##\s+Applicable Scenarios\s*(?P<body>.*?)(?:\n##\s+|\Z)",
        markdown,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if scenario_match:
        for line in scenario_match.group("body").splitlines():
            stripped = line.strip()
            if stripped.startswith("-"):
                intents.append(stripped.lstrip("- ").strip())

    description_match = re.search(
        r"##\s+Description\s*(?P<body>.*?)(?:\n##\s+|\Z)",
        markdown,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if description_match:
        description = " ".join(description_match.group("body").strip().split())
        if description:
            intents.append(description[:500])
    return tuple(dict.fromkeys(intent for intent in intents if intent))


def _workflow_intent_matches(query: str, workflow: WorkflowTemplate) -> bool:
    tokens = _tokens(query)
    intent_tokens = set()
    for intent in workflow.intents:
        intent_tokens.update(_tokens(intent.replace("_", " ")))
    if workflow.event_type:
        intent_tokens.update(_tokens(workflow.event_type.replace("_", " ")))
    return bool(tokens & intent_tokens)


def _workflow_has_strong_executable_match(query: str, workflow: WorkflowTemplate) -> bool:
    """Conservative fallback for executable recipes after explicit predicates.

    Skills are allowed to be broad context, but choosing a deterministic tool
    workflow must be narrow; otherwise generic requests like "oxygen time series"
    can accidentally become lag-correlation plans because they share weak words
    such as "time".
    """

    if workflow.execution_mode == "manual_context_only" or not workflow.steps:
        return False

    lowered = query.lower()
    if workflow.skill_id == "ocean_lag_correlation":
        return _looks_like_lag_correlation_query(lowered)
    if workflow.skill_id == "ocean_heatwave_detection":
        return _looks_like_heatwave_query(lowered)
    if workflow.skill_id == "ocean_hypoxia_detection":
        return _looks_like_hypoxia_detection_query(lowered)
    if workflow.skill_id == "ocean_bloom_detection":
        return _looks_like_bloom_detection_query(lowered)
    if workflow.skill_id == "ocean_event_frequency_map":
        return _looks_like_event_frequency_query(lowered) and infer_event_type(query) == workflow.event_type
    if workflow.skill_id == "ocean_watermass_event_association":
        return _looks_like_watermass_event_association_query(lowered)
    return False


_manual_intents = _skill_intents
_recipe_intent_matches = _workflow_intent_matches
_recipe_has_strong_executable_match = _workflow_has_strong_executable_match


def _looks_like_event_frequency_query(lowered: str) -> bool:
    return bool(
        re.search(
            r"\b(frequency|hotspot|hot spot|density|most often|occur most|spatial density)\b|频率|高发|热点|空间密度",
            lowered,
        )
    )


def _looks_like_watermass_event_association_query(lowered: str) -> bool:
    has_watermass = bool(re.search(r"\bwater\s*-?\s*masses?\b|\bwatermass(?:es)?\b|水团", lowered))
    has_event = bool(
        re.search(
            r"\b(hotspot|hotspots|hot spot|event|events|bloom|chlorophyll|hypoxia|hypoxic|heatwave|upwelling)\b|"
            r"热点|事件|藻华|叶绿素|缺氧|低氧|热浪|上升流",
            lowered,
        )
    )
    has_association = bool(
        re.search(
            r"\b(associate|associated|association|organized|linked|compare|background|dominant|tile|tiles|grid)\b|"
            r"关联|相关|比较|背景|主导|优势|格网|网格|分区",
            lowered,
        )
    )
    return has_watermass and has_event and has_association


def _looks_like_heatwave_query(lowered: str) -> bool:
    return bool(re.search(r"\b(heatwave|marine heatwave|warm event)\b|热浪|海洋热浪|高温事件", lowered))


def _looks_like_hypoxia_detection_query(lowered: str) -> bool:
    has_hypoxia = bool(re.search(r"\b(hypoxia|hypoxic|low oxygen|oxygen depletion)\b|缺氧|低氧", lowered))
    has_driver_analysis = bool(re.search(r"\b(trend|spectrum|correlation|relationship|driver)\b|趋势|功率谱|相关|关系", lowered))
    return has_hypoxia and not has_driver_analysis


def _looks_like_bloom_detection_query(lowered: str) -> bool:
    return bool(re.search(r"\b(algal bloom|bloom|chlorophyll bloom|chla bloom)\b|藻华|水华|叶绿素.*事件", lowered))


def _looks_like_lag_correlation_query(lowered: str) -> bool:
    has_lag = bool(re.search(r"\b(lag|lead|lags|cross[- ]correlation|time lag|delayed)\b|滞后|领先", lowered))
    has_relationship = bool(re.search(r"\b(related|relationship|correlation|coupling|between|with)\b|相关|关系|联系|耦合", lowered))
    return has_lag and has_relationship


def _tokens(text: str) -> set[str]:
    normalized = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return {token for token in normalized.split() if len(token) >= 2}
