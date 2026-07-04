"""Shape-first harness primitives for OceanMind planning and execution."""

from packages.harness.artifact_store import ArtifactRecord, ArtifactStore
from packages.harness.contracts import (
    ValidationIssue,
    ValidationResult,
    infer_frontend_type,
    spec_from_xarray,
    validate_artifact_spec,
    validate_mask_for_artifact,
    validate_task_graph,
    validate_task_graph_contracts,
    validate_tool_runtime_result,
    validate_xarray_matches_spec,
)
from packages.harness.ir import ExecutionSpec, ExecutionStrategy, NodeType, TaskGraph, TaskNode, task_graph_from_nodes
from packages.harness.llm_contract import (
    LLMContractError,
    XMLIOContract,
    parse_json_from_xml_output,
    render_xml_io_contract,
)
from packages.harness.analysis_dag import AnalysisDAGBuilder, AnalysisDAGDecision, PlanRoute
from packages.harness.code_agent import CodeAgent, GeneratedCodePlan
from packages.harness.data_scope import DataScope, DataScopeResolver
from packages.harness.manual_loader import (
    AnalysisManual,
    ManualStepSpec,
    RecipeSpec,
    SkillHeader,
    SkillSpec,
    ToolCallTemplate,
    WorkflowTemplate,
    load_analysis_manuals,
    load_skill_specs,
    parse_skill_frontmatter,
    parse_workflow_steps,
    retrieve_skill_specs,
    select_skill_workflow,
)
from packages.harness.manual_retriever import ManualRetrievalResult, ManualRetriever, SkillRetrievalResult, SkillRetriever
from packages.harness.masks import MaskClass, MaskSpec, mask_spec_from_dims
from packages.harness.tool_contracts import OutputSpec, PortSpec, ToolShapeContract, get_tool_shape_contract
from packages.harness.pipeline import HarnessPlanningPipeline
from packages.harness.result_gatherer import ArtifactGatherer, SynthesisPacketSpec
from packages.harness.semantic_graph import (
    ArtifactContract,
    DataRequirementContract,
    MaskRequirementContract,
    SemanticTask,
    SemanticTaskGraph,
    semantic_graph_from_task_graph,
    semantic_graph_to_dict,
)
from packages.harness.executor import OceanHarnessExecutor
from packages.harness.planner import OceanHarnessPlanner
from packages.harness.planner_agent import PlannerAgent
from packages.harness.code_runner import CodeSafetyError, run_code_node
from packages.harness.shapes import ShapeClass, ShapeSpec, classify_dims, shape_spec_from_dims
from packages.harness.specs import (
    ArtifactKind,
    ArtifactSpec,
    DataBundle,
    FrontendType,
    ReadSpec,
    VerticalSpec,
)

__all__ = [
    "ArtifactKind",
    "ArtifactRecord",
    "ArtifactSpec",
    "ArtifactContract",
    "ArtifactStore",
    "AnalysisDAGBuilder",
    "AnalysisDAGDecision",
    "AnalysisManual",
    "ArtifactGatherer",
    "CodeSafetyError",
    "CodeAgent",
    "DataBundle",
    "DataScope",
    "DataScopeResolver",
    "DataRequirementContract",
    "ExecutionSpec",
    "ExecutionStrategy",
    "FrontendType",
    "GeneratedCodePlan",
    "LLMContractError",
    "MaskClass",
    "MaskRequirementContract",
    "MaskSpec",
    "ManualStepSpec",
    "ManualRetrievalResult",
    "ManualRetriever",
    "NodeType",
    "OceanHarnessExecutor",
    "OceanHarnessPlanner",
    "PlannerAgent",
    "OutputSpec",
    "HarnessPlanningPipeline",
    "PlanRoute",
    "PortSpec",
    "ReadSpec",
    "RecipeSpec",
    "ShapeClass",
    "ShapeSpec",
    "SkillHeader",
    "SkillRetrievalResult",
    "SkillRetriever",
    "SkillSpec",
    "SemanticTask",
    "SemanticTaskGraph",
    "SynthesisPacketSpec",
    "TaskGraph",
    "TaskNode",
    "ToolCallTemplate",
    "ToolShapeContract",
    "ValidationIssue",
    "ValidationResult",
    "VerticalSpec",
    "WorkflowTemplate",
    "XMLIOContract",
    "classify_dims",
    "get_tool_shape_contract",
    "infer_frontend_type",
    "load_analysis_manuals",
    "load_skill_specs",
    "mask_spec_from_dims",
    "parse_json_from_xml_output",
    "parse_skill_frontmatter",
    "parse_workflow_steps",
    "retrieve_skill_specs",
    "render_xml_io_contract",
    "shape_spec_from_dims",
    "spec_from_xarray",
    "task_graph_from_nodes",
    "run_code_node",
    "semantic_graph_from_task_graph",
    "semantic_graph_to_dict",
    "select_skill_workflow",
    "validate_artifact_spec",
    "validate_mask_for_artifact",
    "validate_task_graph",
    "validate_task_graph_contracts",
    "validate_tool_runtime_result",
    "validate_xarray_matches_spec",
]
