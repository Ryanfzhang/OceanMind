from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


NAMED_REGION_PLANNING_RULES = (
    "\n"
    "REGION PARAMETER RULES:\n"
    "- Treat named geographic regions in the user request as explicit spatial intent.\n"
    "- During executable-plan generation, if planner_contract_packet.known_named_region_bounds contains a matching "
    "region, copy that region's lon_range and lat_range into every load_dataset step that serves the request.\n"
    "- If no known bounds are provided but the named region is a common, well-established ocean region, infer a "
    "tight approximate lon_range/lat_range from ocean-domain knowledge.\n"
    "- If multiple named regions are requested, use the relevant bounds per branch or step; when one shared domain "
    "is required, use the tight union of those named-region bounds.\n"
    "- Do NOT use workspace_context.region_bounds, workspace_context.current_region_bounds, or dataset spatial_extent "
    "when the user names a different geographic region.\n"
    "- Use workspace bounds only for phrases such as 'current region', 'this box', 'selected region', "
    "'drawn area', 'drawn polygon', or follow-ups that clearly refer to prior geometry.\n"
    "- If you cannot confidently resolve a named geographic region to approximate bounds, return "
    "clarification_needed instead of silently using the full dataset extent.\n"
)

KNOWN_NAMED_REGION_ALIASES: Dict[str, List[str]] = {
    "yellow sea": ["yellow sea", "黄海"],
    "bohai sea": ["bohai sea", "bohai", "渤海"],
    "east china sea": ["east china sea", "东海"],
    "south china sea": ["south china sea", "南海"],
    "china coastal seas": [
        "china coastal seas",
        "china's coastal seas",
        "chinese coastal seas",
        "china seas",
        "chinese marginal seas",
        "中国海",
        "中国近海",
        "中国沿海",
        "中国海域",
    ],
    "northern south china sea": ["northern south china sea", "northern scs", "南海北部", "北部南海"],
    "luzon strait": ["luzon strait", "吕宋海峡"],
    "taiwan strait": ["taiwan strait", "台湾海峡"],
    "karimata strait": ["karimata strait", "卡里马塔海峡"],
}

KNOWN_NAMED_REGION_BOUNDS: Dict[str, Dict[str, List[float]]] = {
    "yellow sea": {"lon_range": [119.0, 126.0], "lat_range": [33.0, 39.0]},
    "bohai sea": {"lon_range": [117.5, 121.5], "lat_range": [37.0, 41.0]},
    "east china sea": {"lon_range": [121.0, 126.0], "lat_range": [27.0, 32.0]},
    "south china sea": {"lon_range": [105.0, 122.0], "lat_range": [5.0, 23.0]},
    "china coastal seas": {"lon_range": [105.0, 126.0], "lat_range": [5.0, 42.0]},
    "northern south china sea": {"lon_range": [110.0, 120.0], "lat_range": [18.0, 23.0]},
    "luzon strait": {"lon_range": [119.0, 123.0], "lat_range": [18.0, 22.5]},
    "taiwan strait": {"lon_range": [118.0, 122.5], "lat_range": [22.0, 26.5]},
    "karimata strait": {"lon_range": [105.0, 111.0], "lat_range": [-4.0, 1.5]},
}

_GENERIC_NAMED_REGION_PATTERN = re.compile(
    r"\b([A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*){0,4}\s+"
    r"(?:Sea|Seas|Strait|Gulf|Bay|Basin))\b"
)


def extract_named_region_mentions(user_request: str) -> List[str]:
    request = user_request or ""
    lowered = request.lower()
    mentions: List[str] = []

    ordered_aliases = sorted(
        KNOWN_NAMED_REGION_ALIASES.items(),
        key=lambda item: max(len(alias) for alias in item[1]),
        reverse=True,
    )
    for canonical, aliases in ordered_aliases:
        if any(region_alias_matches(lowered, alias) for alias in aliases):
            if any(canonical in existing for existing in mentions):
                continue
            mentions.append(canonical)

    known_aliases = {
        alias
        for aliases in KNOWN_NAMED_REGION_ALIASES.values()
        for alias in aliases
    }
    for match in _GENERIC_NAMED_REGION_PATTERN.findall(request):
        normalized = re.sub(r"\s+", " ", match).strip().lower()
        if not normalized or normalized in known_aliases or normalized in mentions:
            continue
        mentions.append(normalized)

    return mentions


def resolve_named_region_entities(user_request: str) -> Dict[str, Any]:
    """Resolve named ocean regions into router-level structured entities."""
    named_regions = extract_named_region_mentions(user_request)
    known_bounds = {
        region_name: {
            "lon_range": list(KNOWN_NAMED_REGION_BOUNDS[region_name]["lon_range"]),
            "lat_range": list(KNOWN_NAMED_REGION_BOUNDS[region_name]["lat_range"]),
        }
        for region_name in named_regions
        if region_name in KNOWN_NAMED_REGION_BOUNDS
    }
    if not known_bounds:
        return {"named_regions": named_regions} if named_regions else {}

    lon_min = min(bounds["lon_range"][0] for bounds in known_bounds.values())
    lon_max = max(bounds["lon_range"][1] for bounds in known_bounds.values())
    lat_min = min(bounds["lat_range"][0] for bounds in known_bounds.values())
    lat_max = max(bounds["lat_range"][1] for bounds in known_bounds.values())
    primary_region = next(iter(known_bounds))
    resolved = {
        "region_name": primary_region,
        "region_source": "router_named_region",
        "named_regions": named_regions,
        "known_named_region_bounds": known_bounds,
        "lon_range": [lon_min, lon_max],
        "lat_range": [lat_min, lat_max],
        "region_bounds": {
            "lon": [lon_min, lon_max],
            "lat": [lat_min, lat_max],
            "source": "router_named_region",
            "name": primary_region,
        },
    }
    if len(known_bounds) == 1:
        resolved["region"] = {
            "name": primary_region,
            "lon_range": list(known_bounds[primary_region]["lon_range"]),
            "lat_range": list(known_bounds[primary_region]["lat_range"]),
            "source": "router_named_region",
        }
    else:
        resolved["region"] = {
            "name": "union of named regions",
            "region_names": list(known_bounds),
            "lon_range": [lon_min, lon_max],
            "lat_range": [lat_min, lat_max],
            "source": "router_named_region_union",
        }
    return resolved


def region_alias_matches(lowered_request: str, alias: str) -> bool:
    normalized_request = lowered_request.replace("’", "'").replace("‘", "'")
    normalized_alias = alias.lower()
    normalized_alias = normalized_alias.replace("’", "'").replace("‘", "'")
    if re.search(r"[a-z0-9]", normalized_alias):
        return re.search(rf"\b{re.escape(normalized_alias)}\b", normalized_request) is not None
    return normalized_alias in normalized_request


def coerce_numeric_range(value: Any) -> Optional[List[float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        first = float(value[0])
        second = float(value[1])
    except (TypeError, ValueError):
        return None
    return [min(first, second), max(first, second)]


def named_region_extent_suspicion(
    *,
    lon_range: List[float],
    lat_range: List[float],
    named_regions: List[str],
    context_extents: List[tuple[str, tuple[List[float], List[float]]]],
) -> Optional[str]:
    if bounds_are_reasonable_for_known_region(
        lon_range=lon_range,
        lat_range=lat_range,
        named_regions=named_regions,
    ):
        return None

    for label, (context_lon, context_lat) in context_extents:
        if bounds_nearly_equal(
            lon_range,
            lat_range,
            context_lon,
            context_lat,
        ):
            return label

    for region_name in named_regions:
        if bounds_too_broad_for_known_region(
            lon_range=lon_range,
            lat_range=lat_range,
            region_name=region_name,
        ):
            return f"bounds that are too broad for {region_name}"

    return None


def bounds_are_reasonable_for_known_region(
    *,
    lon_range: List[float],
    lat_range: List[float],
    named_regions: List[str],
) -> bool:
    for region_name in named_regions:
        known = KNOWN_NAMED_REGION_BOUNDS.get(region_name)
        if not known:
            continue
        known_lon = known["lon_range"]
        known_lat = known["lat_range"]
        known_area = bounds_area(known_lon, known_lat)
        candidate_area = bounds_area(lon_range, lat_range)
        if known_area <= 0 or candidate_area <= 0:
            continue
        if candidate_area > known_area * 4:
            continue
        if not ranges_overlap(lon_range, known_lon):
            continue
        if not ranges_overlap(lat_range, known_lat):
            continue
        return True
    return False


def bounds_too_broad_for_known_region(
    *,
    lon_range: List[float],
    lat_range: List[float],
    region_name: str,
) -> bool:
    known = KNOWN_NAMED_REGION_BOUNDS.get(region_name)
    if not known:
        return False
    known_lon = known["lon_range"]
    known_lat = known["lat_range"]
    if not ranges_overlap(lon_range, known_lon):
        return False
    if not ranges_overlap(lat_range, known_lat):
        return False
    known_area = bounds_area(known_lon, known_lat)
    candidate_area = bounds_area(lon_range, lat_range)
    return known_area > 0 and candidate_area > known_area * 4


def bounds_nearly_equal(
    lon_a: List[float],
    lat_a: List[float],
    lon_b: List[float],
    lat_b: List[float],
    tolerance: float = 0.35,
) -> bool:
    return (
        ranges_nearly_equal(lon_a, lon_b, tolerance=tolerance)
        and ranges_nearly_equal(lat_a, lat_b, tolerance=tolerance)
    )


def ranges_nearly_equal(
    first: List[float],
    second: List[float],
    *,
    tolerance: float,
) -> bool:
    return abs(first[0] - second[0]) <= tolerance and abs(first[1] - second[1]) <= tolerance


def ranges_overlap(first: List[float], second: List[float], tolerance: float = 0.2) -> bool:
    return max(first[0], second[0]) <= min(first[1], second[1]) + tolerance


def bounds_area(lon_range: List[float], lat_range: List[float]) -> float:
    return max(0.0, lon_range[1] - lon_range[0]) * max(0.0, lat_range[1] - lat_range[0])


def format_region_list(named_regions: List[str]) -> str:
    return ", ".join(sorted(dict.fromkeys(named_regions)))


def format_spatial_bounds(lon_range: List[float], lat_range: List[float]) -> str:
    return (
        f"lon_range=[{lon_range[0]:.3g}, {lon_range[1]:.3g}], "
        f"lat_range=[{lat_range[0]:.3g}, {lat_range[1]:.3g}]"
    )
