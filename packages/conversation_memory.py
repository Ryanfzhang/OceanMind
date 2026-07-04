from __future__ import annotations

import copy
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from packages.llm_gateway.config import load_model_name
from packages.llm_gateway.openai_compatible_client import OpenAICompatibleClientAdapter


logger = logging.getLogger(__name__)

MEMORY_STATE_KEY = "conversation_memory"
MAX_MEMORY_ENTRIES = 48
MAX_MEMORY_INDEX_ENTRIES = 96
MAX_RECALL_CANDIDATES = 24
MAX_ENTRY_TEXT_CHARS = 700
MAX_ENTRY_DATA_CHARS = 1200
MAX_INDEX_TEXT_CHARS = 220

MEMORY_SCOPES = {
    "conversation",
    "project",
    "reference",
}

MEMORY_KINDS = {
    "region",
    "mask",
    "preference",
    "fact",
    "artifact",
    "workflow_pattern",
    "feedback",
    "capability_note",
    "turn_summary",
}

ROLE_ALLOWED_KINDS = {
    "router": {"region", "mask", "preference", "fact", "turn_summary"},
    "planner": {
        "region",
        "mask",
        "preference",
        "fact",
        "artifact",
        "workflow_pattern",
        "feedback",
        "capability_note",
        "turn_summary",
    },
    "synthesizer": {"preference", "fact", "artifact", "feedback", "capability_note", "turn_summary"},
    "general_answer": {"region", "preference", "fact", "feedback", "turn_summary"},
}

ROLE_ALLOWED_SCOPES = {
    "router": {"conversation", "project", "reference"},
    "planner": {"conversation", "project", "reference"},
    "synthesizer": {"conversation", "project", "reference"},
    "general_answer": {"conversation", "project", "reference"},
}

ROLES = tuple(ROLE_ALLOWED_KINDS)


def get_memory_state(conversation_state: Dict[str, Any]) -> Dict[str, Any]:
    memory = conversation_state.setdefault(MEMORY_STATE_KEY, {})
    if not isinstance(memory, dict):
        memory = {}
        conversation_state[MEMORY_STATE_KEY] = memory
    entries = memory.setdefault("entries", [])
    if not isinstance(entries, list):
        memory["entries"] = []
    memory.setdefault("next_id", 1)
    return memory


def build_memory_index(entries: Iterable[Mapping[str, Any]], *, limit: int = MAX_MEMORY_INDEX_ENTRIES) -> List[Dict[str, Any]]:
    """Build a compact index that can be shown to the selector cheaply."""
    rows: List[Dict[str, Any]] = []
    for entry in entries:
        entry_id = str(entry.get("id") or "").strip()
        kind = str(entry.get("kind") or "").strip()
        scope = _coerce_scope(entry.get("scope"), default="conversation", allowed_scopes=MEMORY_SCOPES)
        text = str(entry.get("text") or "").strip()
        if not entry_id or kind not in MEMORY_KINDS or scope not in MEMORY_SCOPES or not text:
            continue
        data = entry.get("data") if isinstance(entry.get("data"), Mapping) else {}
        row: Dict[str, Any] = {
            "id": entry_id,
            "scope": scope,
            "kind": kind,
            "match_key": str(entry.get("match_key") or "").strip(),
            "index_text": _truncate_text(text, MAX_INDEX_TEXT_CHARS),
            "tags": _string_list(data.get("tags") or entry.get("tags")),
            "variables": _string_list(data.get("variables") or data.get("variable") or entry.get("variables")),
            "confidence": _coerce_score(entry.get("confidence")),
            "salience": _coerce_score(entry.get("salience")),
            "updated_at": entry.get("updated_at"),
        }
        region = _memory_region_hint(data)
        if region:
            row["region"] = region
        time_range = _memory_time_range_hint(data)
        if time_range:
            row["time_range"] = time_range
        artifact_refs = _string_list(data.get("artifact_refs") or data.get("result_id") or data.get("result_ids"))
        if artifact_refs:
            row["artifact_refs"] = artifact_refs
        rows.append(row)

    rows.sort(key=lambda row: (float(row.get("salience") or 0.0), str(row.get("updated_at") or "")), reverse=True)
    return rows[:limit]


def rank_memory_index(query: str, index_rows: Iterable[Mapping[str, Any]], *, limit: int = MAX_RECALL_CANDIDATES) -> List[Dict[str, Any]]:
    """Cheap lexical prefilter before the LLM selector hydrates full memory."""
    query_tokens = _tokens(query)
    ranked: List[tuple[float, Dict[str, Any]]] = []
    for raw_row in index_rows:
        row = dict(raw_row)
        haystack = " ".join(
            [
                str(row.get("id") or ""),
                str(row.get("scope") or ""),
                str(row.get("kind") or ""),
                str(row.get("match_key") or ""),
                str(row.get("index_text") or ""),
                " ".join(_string_list(row.get("tags"))),
                " ".join(_string_list(row.get("variables"))),
            ]
        )
        row_tokens = _tokens(haystack)
        overlap = len(query_tokens & row_tokens)
        query_lower = query.lower()
        haystack_lower = haystack.lower()
        substring_bonus = 1 if any(token and token in query_lower for token in row_tokens) else 0
        if not substring_bonus:
            substring_bonus = 1 if any(token and token in haystack_lower for token in query_tokens) else 0
        explicit_reference_bonus = 0
        if _looks_like_memory_reference(query):
            explicit_reference_bonus = 2 if row.get("scope") == "conversation" else 1
        base_score = overlap + substring_bonus + explicit_reference_bonus
        if base_score <= 0:
            continue
        score = base_score + float(row.get("salience") or 0.0) * 0.5
        ranked.append((score, row))

    ranked.sort(key=lambda item: (item[0], float(item[1].get("salience") or 0.0)), reverse=True)
    return [row for _, row in ranked[:limit]]


def _collect_scoped_entries(
    *,
    conversation_state: Dict[str, Any],
    project_memory_state: Optional[Dict[str, Any]],
    reference_memory_state: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    entries.extend(_public_entries(get_memory_state(conversation_state).get("entries", []), default_scope="conversation"))
    if project_memory_state is not None:
        entries.extend(_public_entries(get_memory_state(project_memory_state).get("entries", []), default_scope="project"))
    if reference_memory_state is not None:
        entries.extend(_public_entries(get_memory_state(reference_memory_state).get("entries", []), default_scope="reference"))
    return entries


class MemoryAgent:
    """Owns recall and write decisions for conversation/project/reference memory."""

    def recall(
        self,
        *,
        conversation_state: Dict[str, Any],
        query: str,
        planner: Any,
        role: Optional[str] = None,
        project_memory_state: Optional[Dict[str, Any]] = None,
        reference_memory_state: Optional[Dict[str, Any]] = None,
        max_entries_per_role: int = 8,
    ) -> Dict[str, Any]:
        entries = _collect_scoped_entries(
            conversation_state=conversation_state,
            project_memory_state=project_memory_state,
            reference_memory_state=reference_memory_state,
        )
        if not entries:
            packets = _empty_role_packets("No conversation memory has been recorded yet.")
            return packets.get(role, packets) if role else packets

        memory_index = build_memory_index(entries)
        candidate_index = rank_memory_index(query, memory_index, limit=MAX_RECALL_CANDIDATES)
        if not candidate_index:
            packets = _empty_role_packets("No relevant memory index entries matched the current query.")
            return packets.get(role, packets) if role else packets

        try:
            selection = _select_memory_with_llm(
                planner=planner,
                query=query,
                entries=candidate_index,
                max_entries_per_role=max_entries_per_role,
            )
        except Exception:
            logger.exception("Conversation memory selector failed; continuing without selected memory.")
            packets = _empty_role_packets("Memory selection failed, so no memory was used.")
            return packets.get(role, packets) if role else packets

        packets = _build_role_packets(
            entries=entries,
            selection=selection,
            max_entries_per_role=max_entries_per_role,
        )
        return packets.get(role, packets) if role else packets

    def record(
        self,
        *,
        conversation_state: Dict[str, Any],
        turn_packet: Dict[str, Any],
        planner: Any,
        default_scope: str = "conversation",
        allowed_write_scopes: Sequence[str] = ("conversation",),
    ) -> List[Dict[str, Any]]:
        memory = get_memory_state(conversation_state)
        entries = _public_entries(memory.get("entries", []), default_scope=default_scope)
        structured_operations = self._structured_operations_from_turn(
            turn_packet,
            default_scope=default_scope,
            allowed_write_scopes=allowed_write_scopes,
        )
        if structured_operations and self._turn_is_structured_memory_definition(turn_packet):
            _apply_memory_operations(
                memory,
                structured_operations,
                source_turn_id=turn_packet.get("turn_id"),
                default_scope=default_scope,
                allowed_scopes=allowed_write_scopes,
            )
            return structured_operations

        try:
            llm_operations = _distill_memory_operations_with_llm(
                planner=planner,
                current_entries=entries,
                turn_packet=turn_packet,
                allowed_write_scopes=allowed_write_scopes,
            )
        except Exception:
            logger.exception("Conversation memory distillation failed; using structured memory operations only.")
            llm_operations = []

        operations = [*structured_operations, *llm_operations]
        _apply_memory_operations(
            memory,
            operations,
            source_turn_id=turn_packet.get("turn_id"),
            default_scope=default_scope,
            allowed_scopes=allowed_write_scopes,
        )
        return operations

    def _structured_operations_from_turn(
        self,
        turn_packet: Mapping[str, Any],
        *,
        default_scope: str,
        allowed_write_scopes: Sequence[str],
    ) -> List[Dict[str, Any]]:
        if default_scope not in allowed_write_scopes:
            return []
        queries = [
            str(turn_packet.get("user_query") or ""),
            str(turn_packet.get("effective_query") or ""),
        ]
        operations: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for query in queries:
            parsed_region = parse_named_region_definition(query)
            if parsed_region is None:
                continue
            match_key = f"region:{normalize_memory_name(parsed_region['name'])}"
            if match_key in seen:
                continue
            seen.add(match_key)
            operations.append(
                {
                    "op": "upsert",
                    "scope": default_scope,
                    "kind": "region",
                    "match_key": match_key,
                    "text": (
                        f"User defined named region '{parsed_region['name']}' as "
                        f"lon_range={parsed_region['lon_range']} and lat_range={parsed_region['lat_range']}."
                    ),
                    "data": {
                        "name": parsed_region["name"],
                        "aliases": [parsed_region["name"]],
                        "lon_range": parsed_region["lon_range"],
                        "lat_range": parsed_region["lat_range"],
                        "tags": ["named_region", parsed_region["name"]],
                    },
                    "confidence": 1.0,
                    "salience": 0.95,
                }
            )
        return operations

    @staticmethod
    def _turn_is_structured_memory_definition(turn_packet: Mapping[str, Any]) -> bool:
        if str(turn_packet.get("routing_mode") or "") != "general_answer":
            return False
        if turn_packet.get("skill_id") or turn_packet.get("skills_used"):
            return False
        plan = turn_packet.get("plan")
        if isinstance(plan, Mapping) and plan:
            return False
        result_summaries = turn_packet.get("result_summaries")
        return not (isinstance(result_summaries, list) and result_summaries)


def retrieve_memory_packets(
    *,
    conversation_state: Dict[str, Any],
    query: str,
    planner: Any,
    project_memory_state: Optional[Dict[str, Any]] = None,
    reference_memory_state: Optional[Dict[str, Any]] = None,
    max_entries_per_role: int = 8,
) -> Dict[str, Dict[str, Any]]:
    return MemoryAgent().recall(
        conversation_state=conversation_state,
        query=query,
        planner=planner,
        project_memory_state=project_memory_state,
        reference_memory_state=reference_memory_state,
        max_entries_per_role=max_entries_per_role,
    )


def record_turn_memory(
    *,
    conversation_state: Dict[str, Any],
    turn_packet: Dict[str, Any],
    planner: Any,
    default_scope: str = "conversation",
    allowed_write_scopes: Sequence[str] = ("conversation",),
) -> None:
    MemoryAgent().record(
        conversation_state=conversation_state,
        turn_packet=turn_packet,
        planner=planner,
        default_scope=default_scope,
        allowed_write_scopes=allowed_write_scopes,
    )


def remember_memory_entry(
    memory_state: Dict[str, Any],
    *,
    kind: str,
    text: str,
    scope: str = "conversation",
    match_key: str = "",
    data: Optional[Mapping[str, Any]] = None,
    confidence: float = 0.8,
    salience: float = 0.6,
    source_turn_id: Any = None,
    op: str = "upsert",
) -> Optional[Dict[str, Any]]:
    """Add or update one structured memory entry without invoking an LLM.

    This is useful for seeding project/reference memory from deterministic
    application events such as named regions, reusable masks, or curated
    workflow patterns.
    """
    memory = get_memory_state(memory_state)
    _apply_memory_operations(
        memory,
        [
            {
                "op": op,
                "scope": scope,
                "kind": kind,
                "match_key": match_key,
                "text": text,
                "data": dict(data or {}),
                "confidence": confidence,
                "salience": salience,
            }
        ],
        source_turn_id=source_turn_id,
        default_scope=scope,
        allowed_scopes=MEMORY_SCOPES,
    )
    public = _public_entries(memory.get("entries", []), default_scope=scope)
    if match_key:
        return next((entry for entry in public if entry.get("match_key") == match_key), None)
    return public[0] if public else None


def list_memory_entries(memory_state: Dict[str, Any], *, default_scope: str = "conversation") -> List[Dict[str, Any]]:
    return _public_entries(get_memory_state(memory_state).get("entries", []), default_scope=default_scope)


def parse_named_region_definition(query: str) -> Optional[Dict[str, Any]]:
    text = str(query or "").strip()
    if not text:
        return None
    if not re.search(r"叫做|叫作|称为|命名为|定义为|记为|设为|\bcall(?:ed)?\b|\bname(?:d)?\b|\bdefine(?:d)?\s+as\b", text, re.IGNORECASE):
        return None

    bounds = _extract_lon_lat_bounds_from_text(text)
    if bounds is None:
        return None
    name = _extract_region_definition_name(text[bounds["end"] :])
    if not name or len(name) > 80:
        return None
    return {
        "name": name,
        "lon_range": bounds["lon_range"],
        "lat_range": bounds["lat_range"],
    }


def resolved_entities_from_memory_entries(entries: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    matches: List[Dict[str, Any]] = []
    for entry in entries:
        if entry.get("kind") != "region":
            continue
        data = entry.get("data") if isinstance(entry.get("data"), Mapping) else {}
        lon_range = coerce_memory_range(data.get("lon_range"))
        lat_range = coerce_memory_range(data.get("lat_range"))
        if lon_range is None or lat_range is None:
            continue
        names = _memory_region_names(entry, data)
        display_name = str(data.get("name") or entry.get("match_key") or (names[0] if names else "memory region"))
        display_name = display_name.replace("region:", "").strip()
        matches.append(
            {
                "name": display_name,
                "lon_range": lon_range,
                "lat_range": lat_range,
                "scope": entry.get("scope"),
                "memory_id": entry.get("id"),
            }
        )

    if not matches:
        return {}
    primary = matches[0]
    known_bounds = {
        str(match["name"]): {
            "lon_range": list(match["lon_range"]),
            "lat_range": list(match["lat_range"]),
            "source": f"memory:{match.get('scope') or 'conversation'}",
            "memory_id": match.get("memory_id"),
        }
        for match in matches
    }
    return {
        "region_name": primary["name"],
        "region_source": "conversation_memory",
        "named_regions": [str(match["name"]) for match in matches],
        "known_named_region_bounds": known_bounds,
        "lon_range": list(primary["lon_range"]),
        "lat_range": list(primary["lat_range"]),
        "region_bounds": {
            "lon": list(primary["lon_range"]),
            "lat": list(primary["lat_range"]),
            "source": "conversation_memory",
            "name": primary["name"],
        },
        "region": {
            "name": primary["name"],
            "lon_range": list(primary["lon_range"]),
            "lat_range": list(primary["lat_range"]),
            "source": "conversation_memory",
        },
    }


def normalize_memory_name(value: Any) -> str:
    return re.sub(r"[\s\"'“”‘’。.!！?？,，、;；:_-]+", "", str(value or "").lower())


def coerce_memory_range(value: Any) -> Optional[List[float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return sorted([float(value[0]), float(value[1])])
    except (TypeError, ValueError):
        return None


def _empty_role_packets(reason: str) -> Dict[str, Dict[str, Any]]:
    return {
        role: {
            "role": role,
            "entries": [],
            "selection_reason": reason,
            "resolved_entities": {},
            "usage_policy": _role_usage_policy(role),
        }
        for role in ROLES
    }


def _role_usage_policy(role: str) -> str:
    if role == "router":
        return (
            "Use selected memory only to resolve references, user-defined region names, or lightweight preferences. "
            "Do not infer that the active dataset is needed solely because a previous turn used it."
        )
    if role == "planner":
        return (
            "Use selected memory as scoped defaults or reusable analysis context only when relevant to the current query. "
            "The current user query overrides memory."
        )
    if role == "synthesizer":
        return (
            "Use selected memory only to distinguish current evidence from relevant earlier conversation context."
        )
    return (
        "Use selected memory only for user preferences or general conversation facts. "
        "Do not use dataset analysis artifacts unless the current query explicitly asks about previous analysis."
    )


def _select_memory_with_llm(
    *,
    planner: Any,
    query: str,
    entries: List[Dict[str, Any]],
    max_entries_per_role: int,
) -> Dict[str, Any]:
    helper = _memory_helper(planner)
    client = helper._get_client()
    payload = {
        "current_query": query,
        "memory_index": entries,
        "roles": list(ROLES),
        "max_entries_per_role": max_entries_per_role,
        "selection_contract": {
            "router": "Select only memory needed to resolve references or user-defined names in the current query.",
            "planner": "Select memory that can provide defaults, named definitions, prior artifacts, or capability limits relevant to the current query.",
            "synthesizer": "Select memory useful for explaining current results relative to relevant prior context.",
            "general_answer": "Select only non-dataset preferences or general facts relevant to the current query.",
        },
    }
    system = (
        "You are a conservative memory selector for OceanMind.\n"
        "Your job is to decide which compact memory-index entries are relevant to the CURRENT query for each role. "
        "The runtime will hydrate the full memory content only after you select ids.\n"
        "Avoid misuse: do not select memory merely because it is recent. Select an entry only when the current query "
        "contains a reference, name, preference, comparison, or omitted parameter that the entry directly helps resolve.\n"
        "If the current query is standalone or unrelated to prior memory, return empty arrays for all roles.\n"
        "For general_answer, never select dataset analysis artifacts or capability notes unless the user explicitly asks "
        "about a previous analysis result. For router, do not select artifacts unless the current query explicitly refers "
        "to a previous result. Project/reference memory is reusable context, not a user profile. Current query always "
        "overrides memory.\n"
        "Return JSON only with keys router, planner, synthesizer, general_answer. Each value is a list of objects "
        "with id and reason."
    )
    response = helper._create_message(
        client=client,
        max_tokens=1400,
        temperature=0.0,
        system=system,
        messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)}],
        request_name="conversation_memory_select",
        json_response=True,
    )
    return helper._parse_json_response(helper._extract_response_text(response))


def _distill_memory_operations_with_llm(
    *,
    planner: Any,
    current_entries: List[Dict[str, Any]],
    turn_packet: Dict[str, Any],
    allowed_write_scopes: Sequence[str],
) -> List[Dict[str, Any]]:
    helper = _memory_helper(planner)
    client = helper._get_client()
    payload = {
        "current_memory_entries": current_entries,
        "turn": turn_packet,
        "allowed_kinds": sorted(MEMORY_KINDS),
        "allowed_scopes": sorted(scope for scope in allowed_write_scopes if scope in MEMORY_SCOPES),
        "operation_schema": {
            "operations": [
                {
                    "op": "upsert | append | delete",
                    "scope": "conversation | project | reference",
                    "kind": "region | mask | preference | fact | artifact | workflow_pattern | feedback | capability_note | turn_summary",
                    "match_key": "stable key for upsert/delete, such as region:philippine_sea",
                    "text": "concise memory text",
                    "data": {},
                    "confidence": 0.0,
                    "salience": 0.0,
                }
            ]
        },
    }
    system = (
        "You maintain typed memory for OceanMind. Extract only information from this turn that is likely to help "
        "later turns within the allowed memory scopes.\n"
        "Do not use regex-style guessing and do not invent facts. User-stated information and executed results are stronger "
        "than model inferences. Preserve uncertainty when needed.\n"
        "Good memory includes user-defined regions, reusable masks, stable preferences, reusable analysis artifacts, "
        "workflow patterns, user corrections/feedback, capability boundaries, and short turn summaries. Do not store "
        "trivial chatter, one-off wording, or hidden implementation details.\n"
        "There is currently no user/account memory scope. Use only the allowed scopes in the payload; normally this is "
        "conversation scope unless the application explicitly enables project or reference writes.\n"
        "Use upsert when this turn updates an existing definition or preference. Use append for independent turn summaries "
        "or artifacts. Use delete only if the user explicitly corrects or withdraws a memory.\n"
        "If nothing is worth remembering, return {\"operations\": []}. Return JSON only."
    )
    response = helper._create_message(
        client=client,
        max_tokens=1800,
        temperature=0.0,
        system=system,
        messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)}],
        request_name="conversation_memory_distill",
        json_response=True,
    )
    parsed = helper._parse_json_response(helper._extract_response_text(response))
    operations = parsed.get("operations")
    return operations if isinstance(operations, list) else []


def _build_role_packets(
    *,
    entries: List[Dict[str, Any]],
    selection: Dict[str, Any],
    max_entries_per_role: int,
) -> Dict[str, Dict[str, Any]]:
    if not isinstance(selection, dict):
        return _empty_role_packets("Memory selector returned an invalid payload, so no memory was used.")
    by_id = {str(entry.get("id")): entry for entry in entries if entry.get("id")}
    packets: Dict[str, Dict[str, Any]] = {}
    for role in ROLES:
        allowed_kinds = ROLE_ALLOWED_KINDS[role]
        allowed_scopes = ROLE_ALLOWED_SCOPES[role]
        selected_items = selection.get(role)
        selected_entries: List[Dict[str, Any]] = []
        reasons: Dict[str, str] = {}
        if isinstance(selected_items, list):
            for item in selected_items:
                if not isinstance(item, dict):
                    continue
                entry_id = str(item.get("id") or "").strip()
                reason = str(item.get("reason") or "").strip()
                if not entry_id or not reason:
                    continue
                entry = by_id.get(entry_id)
                if not entry or entry.get("kind") not in allowed_kinds or entry.get("scope") not in allowed_scopes:
                    continue
                selected_entries.append(entry)
                reasons[entry_id] = reason
                if len(selected_entries) >= max_entries_per_role:
                    break
        packets[role] = {
            "role": role,
            "entries": selected_entries,
            "selection_reasons": reasons,
            "resolved_entities": resolved_entities_from_memory_entries(selected_entries),
            "usage_policy": _role_usage_policy(role),
        }
    return packets


def _apply_memory_operations(
    memory: Dict[str, Any],
    operations: List[Dict[str, Any]],
    *,
    source_turn_id: Any,
    default_scope: str = "conversation",
    allowed_scopes: Iterable[str] = ("conversation",),
) -> None:
    entries = memory.setdefault("entries", [])
    if not isinstance(entries, list):
        entries = []
        memory["entries"] = entries
    now = datetime.now(timezone.utc).isoformat()

    for operation in operations:
        normalized = _normalize_operation(operation, default_scope=default_scope, allowed_scopes=allowed_scopes)
        if not normalized:
            continue
        op = normalized["op"]
        scope = normalized["scope"]
        match_key = normalized.get("match_key")
        if op == "delete":
            if match_key:
                memory["entries"] = [
                    entry
                    for entry in entries
                    if not (
                        str(entry.get("match_key") or "") == match_key
                        and _coerce_scope(entry.get("scope"), default=default_scope, allowed_scopes=MEMORY_SCOPES) == scope
                    )
                ]
                entries = memory["entries"]
            continue

        if op == "upsert" and match_key:
            existing = next(
                (
                    entry
                    for entry in entries
                    if entry.get("match_key") == match_key
                    and _coerce_scope(entry.get("scope"), default=default_scope, allowed_scopes=MEMORY_SCOPES) == scope
                ),
                None,
            )
            if isinstance(existing, dict):
                existing.update(
                    {
                        "scope": scope,
                        "kind": normalized["kind"],
                        "text": normalized["text"],
                        "data": normalized.get("data", {}),
                        "confidence": normalized["confidence"],
                        "salience": normalized["salience"],
                        "updated_at": now,
                        "source_turn_id": source_turn_id,
                    }
                )
                continue

        next_id = int(memory.get("next_id") or 1)
        memory["next_id"] = next_id + 1
        entries.append(
            {
                "id": f"mem_{next_id:04d}",
                "scope": scope,
                "kind": normalized["kind"],
                "match_key": match_key or f"{normalized['kind']}:mem_{next_id:04d}",
                "text": normalized["text"],
                "data": normalized.get("data", {}),
                "confidence": normalized["confidence"],
                "salience": normalized["salience"],
                "source_turn_id": source_turn_id,
                "created_at": now,
                "updated_at": now,
            }
        )

    entries.sort(key=lambda entry: float(entry.get("salience") or 0.0), reverse=True)
    del entries[MAX_MEMORY_ENTRIES:]


def _normalize_operation(
    operation: Any,
    *,
    default_scope: str = "conversation",
    allowed_scopes: Iterable[str] = ("conversation",),
) -> Optional[Dict[str, Any]]:
    if not isinstance(operation, dict):
        return None
    op = str(operation.get("op") or "").strip().lower()
    if op not in {"upsert", "append", "delete"}:
        return None
    scope = _coerce_scope(operation.get("scope"), default=default_scope, allowed_scopes=allowed_scopes)
    kind = str(operation.get("kind") or "").strip()
    if kind not in MEMORY_KINDS:
        return None
    match_key = str(operation.get("match_key") or "").strip()
    if op == "delete":
        return {"op": op, "scope": scope, "kind": kind, "match_key": match_key}

    text = _truncate_text(str(operation.get("text") or "").strip(), MAX_ENTRY_TEXT_CHARS)
    if not text:
        return None
    data = _safe_data(operation.get("data"))
    return {
        "op": op,
        "scope": scope,
        "kind": kind,
        "match_key": match_key,
        "text": text,
        "data": data,
        "confidence": _coerce_score(operation.get("confidence")),
        "salience": _coerce_score(operation.get("salience")),
    }


def _public_entries(entries: Any, *, default_scope: str = "conversation") -> List[Dict[str, Any]]:
    if not isinstance(entries, list):
        return []
    public: List[Dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        raw_id = str(entry.get("id") or "").strip()
        scope = _coerce_scope(entry.get("scope"), default=default_scope, allowed_scopes=MEMORY_SCOPES)
        kind = str(entry.get("kind") or "").strip()
        text = str(entry.get("text") or "").strip()
        if not raw_id or kind not in MEMORY_KINDS or scope not in MEMORY_SCOPES or not text:
            continue
        public.append(
            {
                "id": _public_memory_id(scope, raw_id),
                "raw_id": raw_id,
                "scope": scope,
                "kind": kind,
                "match_key": str(entry.get("match_key") or "").strip(),
                "text": _truncate_text(text, MAX_ENTRY_TEXT_CHARS),
                "data": _safe_data(entry.get("data")),
                "confidence": _coerce_score(entry.get("confidence")),
                "salience": _coerce_score(entry.get("salience")),
                "source_turn_id": entry.get("source_turn_id"),
                "updated_at": entry.get("updated_at"),
            }
        )
    public.sort(key=lambda entry: float(entry.get("salience") or 0.0), reverse=True)
    return public[:MAX_MEMORY_ENTRIES]


def _coerce_scope(value: Any, *, default: str, allowed_scopes: Iterable[str]) -> str:
    allowed = {str(scope) for scope in allowed_scopes if str(scope) in MEMORY_SCOPES}
    fallback = default if default in MEMORY_SCOPES else "conversation"
    scope = str(value or fallback).strip().lower()
    if scope in allowed:
        return scope
    if fallback in allowed:
        return fallback
    return sorted(allowed)[0] if allowed else "conversation"


def _public_memory_id(scope: str, raw_id: str) -> str:
    if raw_id.startswith(f"{scope}:"):
        return raw_id
    return f"{scope}:{raw_id}"


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-zA-Z0-9_\u4e00-\u9fff]+", text.lower()) if len(token) > 1}


def _looks_like_memory_reference(query: str) -> bool:
    lowered = query.lower()
    return bool(
        re.search(
            r"\b(previous|last|same|that|this|again|continue|reuse|named|defined)\b|"
            r"上次|刚才|之前|这个|那个|同样|继续|复用|定义的|命名",
            lowered,
        )
    )


def _string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, Mapping):
        return [str(key) for key in value.keys()]
    if isinstance(value, Iterable):
        result: List[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                result.append(text)
        return result
    text = str(value).strip()
    return [text] if text else []


def _memory_region_hint(data: Mapping[str, Any]) -> Dict[str, Any]:
    region: Dict[str, Any] = {}
    lon_range = data.get("lon_range")
    lat_range = data.get("lat_range")
    polygon = data.get("polygon") or data.get("polygon_points")
    if isinstance(lon_range, list) and len(lon_range) == 2:
        region["lon_range"] = lon_range
    if isinstance(lat_range, list) and len(lat_range) == 2:
        region["lat_range"] = lat_range
    if isinstance(polygon, list) and polygon:
        region["polygon_points_count"] = len(polygon)
    return region


def _memory_time_range_hint(data: Mapping[str, Any]) -> List[Any]:
    time_range = data.get("time_range")
    if isinstance(time_range, list) and len(time_range) == 2:
        return time_range
    if isinstance(time_range, tuple) and len(time_range) == 2:
        return list(time_range)
    return []


def _extract_lon_lat_bounds_from_text(text: str) -> Optional[Dict[str, Any]]:
    range_sep = r"(?:-|–|—|~|至|到)"
    coord_sep = r"(?:\s*,\s*|\s*，\s*|\s*、\s*|\s*;\s*|\s*；\s*|\s+)"
    number = r"[-+]?\d+(?:\.\d+)?"
    pattern = re.compile(
        rf"(?P<lon1>{number})\s*°?\s*(?P<lon1_dir>[EeWw东西])?\s*{range_sep}\s*"
        rf"(?P<lon2>{number})\s*°?\s*(?P<lon2_dir>[EeWw东西])"
        rf"{coord_sep}"
        rf"(?P<lat1>{number})\s*°?\s*(?P<lat1_dir>[NnSs南北])?\s*{range_sep}\s*"
        rf"(?P<lat2>{number})\s*°?\s*(?P<lat2_dir>[NnSs南北])",
    )
    match = pattern.search(text)
    if not match:
        return None
    lon1 = _signed_coordinate(match.group("lon1"), match.group("lon1_dir") or match.group("lon2_dir"))
    lon2 = _signed_coordinate(match.group("lon2"), match.group("lon2_dir"))
    lat1 = _signed_coordinate(match.group("lat1"), match.group("lat1_dir") or match.group("lat2_dir"))
    lat2 = _signed_coordinate(match.group("lat2"), match.group("lat2_dir"))
    if None in {lon1, lon2, lat1, lat2}:
        return None
    lon_range = sorted([float(lon1), float(lon2)])
    lat_range = sorted([float(lat1), float(lat2)])
    return {"lon_range": lon_range, "lat_range": lat_range, "end": match.end()}


def _signed_coordinate(value: Any, direction: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    direction_text = str(direction or "").strip().lower()
    if direction_text in {"w", "west", "西", "s", "south", "南"}:
        return -abs(number)
    return number


def _extract_region_definition_name(text_after_bounds: str) -> Optional[str]:
    match = re.search(
        r"(?:叫做|叫作|称为|命名为|定义为|记为|设为|\bcalled\s+|\bcall\s+|\bname\s+|\bnamed\s+|\bdefine(?:d)?\s+as\s+)(?P<name>.+)$",
        text_after_bounds,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    name = str(match.group("name") or "").strip()
    name = re.sub(r"^[\"'“‘\s]+|[\"'”’。.!！?？\s]+$", "", name)
    name = re.sub(r"^(?:it|this|the region|as)\s+", "", name, flags=re.IGNORECASE).strip()
    return name or None


def _memory_region_names(entry: Mapping[str, Any], data: Mapping[str, Any]) -> List[str]:
    raw_names: List[Any] = [data.get("name"), entry.get("match_key")]
    aliases = data.get("aliases")
    if isinstance(aliases, list):
        raw_names.extend(aliases)
    names: List[str] = []
    for raw_name in raw_names:
        text = str(raw_name or "").replace("region:", "").strip()
        normalized = normalize_memory_name(text)
        if normalized and normalized not in names:
            names.append(normalized)
    return names


def _safe_data(value: Any) -> Any:
    try:
        copied = json.loads(json.dumps(value if value is not None else {}, ensure_ascii=False, default=str))
    except Exception:
        copied = {}
    text = json.dumps(copied, ensure_ascii=False, default=str)
    if len(text) <= MAX_ENTRY_DATA_CHARS:
        return copied
    return {"summary": _truncate_text(text, MAX_ENTRY_DATA_CHARS)}


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "..."


def _coerce_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, score))


def _memory_helper(planner: Any) -> Any:
    if planner is not None:
        return planner
    from packages.llm_gateway.skill_planner import SkillPlanner

    return SkillPlanner(model=load_model_name("MEMORY_MODEL", default=SkillPlanner.DEFAULT_MODEL))


def clone_memory_packet(packet: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(packet, dict):
        return {"entries": []}
    return copy.deepcopy(packet)
