"""Data-scope resolution for OceanMind dataset analysis queries."""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from packages.harness.specs import VerticalSpec
from packages.runtime import get_active_dataset_public_config


VARIABLE_ALIASES: Dict[str, Tuple[str, ...]] = {
    "oxygen": ("oxygen", "o2", "dissolved oxygen", "do", "缺氧", "低氧", "溶解氧", "氧"),
    "temp": ("temp", "temperature", "sst", "水温", "温度"),
    "salt": ("salt", "salinity", "盐度"),
    "u": ("eastward velocity", "u velocity", "u流速", "东向流", "u"),
    "v": ("northward velocity", "v velocity", "v流速", "北向流", "v"),
    "chlorophyll": ("chlorophyll", "chl", "chla", "叶绿素"),
}

NAMED_REGION_BOUNDS: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]] = {
    "east china sea": ((118.0, 126.0), (24.0, 33.0)),
    "east china sea shelf": ((118.0, 126.0), (24.0, 33.0)),
    "yellow sea": ((119.0, 126.0), (32.0, 39.5)),
    "bohai sea": ((117.0, 122.2), (37.0, 41.3)),
    "south china sea": ((108.0, 121.5), (5.0, 23.5)),
    "pearl river estuary": ((112.0, 116.5), (20.0, 24.0)),
}


@dataclass(frozen=True)
class DataScope:
    user_request: str
    extracted_params: Mapping[str, Any]
    additional_context: Mapping[str, Any]
    dataset: str
    lon_range: Tuple[float, float]
    lat_range: Tuple[float, float]
    time_range: Optional[Tuple[str, str]]
    variables: Tuple[str, ...]
    vertical: VerticalSpec
    explicit_region: bool
    explicit_time_range: bool
    dataset_info: Mapping[str, Any]

    def to_requirements_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "lon_range": list(self.lon_range),
            "lat_range": list(self.lat_range),
            "time_range": list(self.time_range) if self.time_range is not None else None,
            "variables": list(self.variables),
            "vertical": vertical_to_dict(self.vertical),
            "explicit_region": self.explicit_region,
            "explicit_time_range": self.explicit_time_range,
        }


class DataScopeResolver:
    """Resolve dataset, region, time, variables, and vertical semantics."""

    def resolve(
        self,
        *,
        user_request: str,
        extracted_params: Optional[Mapping[str, Any]] = None,
        additional_context: Optional[Mapping[str, Any]] = None,
    ) -> DataScope:
        extracted = dict(extracted_params or {})
        context = dict(additional_context or {})
        dataset_info = get_active_dataset_public_config()
        dataset_id = str(dataset_info.get("id") or "current")
        planner_scope = context.get("planner_resolved_scope")
        if isinstance(planner_scope, Mapping):
            return _scope_from_planner_resolution(
                user_request=user_request,
                extracted_params=extracted,
                additional_context=context,
                dataset_info=dataset_info,
                dataset_id=dataset_id,
                planner_scope=planner_scope,
            )
        lon_range, lat_range = resolve_region(user_request, extracted, context, dataset_info)
        time_range = resolve_time_range(user_request, extracted, context, dataset_info)
        variables = infer_variables(user_request, extracted)
        return DataScope(
            user_request=user_request,
            extracted_params=extracted,
            additional_context=context,
            dataset=dataset_id,
            lon_range=lon_range,
            lat_range=lat_range,
            time_range=time_range,
            variables=variables,
            vertical=vertical_spec_from_text(user_request),
            explicit_region=has_explicit_region(user_request, extracted, context),
            explicit_time_range=has_explicit_time_range(user_request, extracted, context),
            dataset_info=dataset_info,
        )


def _scope_from_planner_resolution(
    *,
    user_request: str,
    extracted_params: Mapping[str, Any],
    additional_context: Mapping[str, Any],
    dataset_info: Mapping[str, Any],
    dataset_id: str,
    planner_scope: Mapping[str, Any],
) -> DataScope:
    strict = bool(additional_context.get("planner_resolved_scope_strict"))
    required_fields = {
        str(item)
        for item in additional_context.get("planner_scope_required_fields", [])
        if str(item).strip()
    }
    resolved_region = region_from_mapping(planner_scope)
    if resolved_region is None:
        if strict and {"lon_range", "lat_range", "region"}.intersection(required_fields):
            raise ValueError("Planner contract missing resolved_scope lon_range/lat_range.")
        resolved_region = resolve_region(user_request, extracted_params, additional_context, dataset_info)
    lon_range, lat_range = resolved_region

    time_range = coerce_str_range(planner_scope.get("time_range"))
    if time_range is None:
        if strict and "time_range" in required_fields:
            raise ValueError("Planner contract missing resolved_scope time_range.")
        time_range = resolve_time_range(user_request, extracted_params, additional_context, dataset_info)

    variables = _variables_from_planner_scope(planner_scope)
    if not variables:
        if strict and {"variable", "variables"}.intersection(required_fields):
            raise ValueError("Planner contract missing resolved_scope variable/variables.")
        variables = infer_variables(user_request, extracted_params)

    vertical = vertical_spec_from_mapping(planner_scope)
    if vertical.mode == "unspecified" and not strict:
        vertical = vertical_spec_from_text(user_request)

    return DataScope(
        user_request=user_request,
        extracted_params=extracted_params,
        additional_context=additional_context,
        dataset=str(planner_scope.get("dataset") or dataset_id),
        lon_range=lon_range,
        lat_range=lat_range,
        time_range=time_range,
        variables=variables,
        vertical=vertical,
        explicit_region=region_from_mapping(planner_scope) is not None,
        explicit_time_range=coerce_str_range(planner_scope.get("time_range")) is not None,
        dataset_info=dataset_info,
    )


def _variables_from_planner_scope(planner_scope: Mapping[str, Any]) -> Tuple[str, ...]:
    variables = planner_scope.get("variables")
    if isinstance(variables, (list, tuple)):
        normalized = tuple(str(item).strip() for item in variables if str(item).strip())
        if normalized:
            return normalized
    variable = planner_scope.get("variable")
    if isinstance(variable, str) and variable.strip():
        return (variable.strip(),)
    return ()


def vertical_spec_from_mapping(mapping: Mapping[str, Any]) -> VerticalSpec:
    mode = mapping.get("vertical_mode") or mapping.get("mode")
    if not isinstance(mode, str) or not mode.strip():
        return VerticalSpec(mode="unspecified")
    depth_value = mapping.get("depth_value")
    try:
        depth_value = float(depth_value) if depth_value is not None else None
    except (TypeError, ValueError):
        depth_value = None
    depth_range = coerce_depth_range(mapping.get("depth_range"))
    aggregation = mapping.get("depth_aggregation") or mapping.get("aggregation")
    return VerticalSpec(
        mode=mode.strip(),
        depth_value=depth_value,
        depth_range=depth_range,
        aggregation=str(aggregation).strip() if aggregation is not None else None,
        source_text="planner_resolved_scope",
    )


def infer_variables(text: str, extracted_params: Mapping[str, Any]) -> Tuple[str, ...]:
    lowered = text.lower()
    found: List[str] = []
    for variable, aliases in VARIABLE_ALIASES.items():
        if any(contains_alias(lowered, alias) for alias in aliases):
            found.append(variable)
    explicit = extracted_params.get("variable")
    if isinstance(explicit, str) and explicit and explicit not in found:
        found.append(explicit)
    explicit_many = extracted_params.get("variables")
    if isinstance(explicit_many, (list, tuple)):
        for item in explicit_many:
            if isinstance(item, str) and item and item not in found:
                found.append(item)
    return tuple(found or ("temp",))


def infer_primary_variable(text: str, extracted_params: Mapping[str, Any]) -> str:
    return infer_variables(text, extracted_params)[0]


def vertical_spec_from_text(text: str, *, default_bottom: bool = False) -> VerticalSpec:
    lowered = text.lower()
    thickness = extract_bottom_band_thickness(lowered)
    if thickness is not None:
        return VerticalSpec(
            mode="bottom_band",
            relative_to="bottom",
            band_thickness_m=thickness,
            aggregation="mean",
            retain_depth=False,
            source_text=text,
        )
    fixed_depth = extract_fixed_depth(lowered)
    if fixed_depth is not None:
        return VerticalSpec(mode="fixed_depth", depth_value=fixed_depth, aggregation="mean", source_text=text)
    depth_range = extract_depth_range(lowered)
    if depth_range is not None:
        return VerticalSpec(mode="depth_range", depth_range=depth_range, aggregation="mean", source_text=text)
    if re.search(r"\b(bottom|near[- ]bottom|seafloor|benthic)\b|底层|近底|海底", lowered) or default_bottom:
        return VerticalSpec(mode="bottom", aggregation="mean", source_text=text)
    if re.search(r"\bsst\b|\b(surface|sea[- ]surface)\b|表层|海表", lowered):
        return VerticalSpec(mode="surface", depth_range=(0.0, 0.0), aggregation="mean", source_text=text)
    return VerticalSpec(mode="unspecified", source_text=text)


def infer_lag_variables(text: str) -> Tuple[Optional[str], Optional[str]]:
    lowered = text.lower()
    matches: List[Tuple[int, str]] = []
    for variable, aliases in VARIABLE_ALIASES.items():
        positions = [alias_position(lowered, alias) for alias in aliases]
        positions = [position for position in positions if position is not None]
        if positions:
            matches.append((min(positions), variable))
    matches.sort()
    ordered: List[str] = []
    for _, variable in matches:
        if variable not in ordered:
            ordered.append(variable)
    if len(ordered) >= 2:
        return ordered[0], ordered[1]
    return (ordered[0], None) if ordered else (None, None)


def variable_vertical_spec(text: str, variable: str) -> VerticalSpec:
    depth_range = infer_variable_depth_range(text, variable)
    if depth_range is not None:
        if depth_range[0] == 0.0 and depth_range[1] == 0.0:
            return VerticalSpec(mode="surface", depth_range=depth_range, aggregation="mean", source_text=text)
        if abs(depth_range[0] - depth_range[1]) < 1e-9:
            return VerticalSpec(
                mode="fixed_depth",
                depth_value=depth_range[0],
                depth_range=depth_range,
                aggregation="mean",
                source_text=text,
            )
        return VerticalSpec(mode="depth_range", depth_range=depth_range, aggregation="mean", source_text=text)
    return VerticalSpec(mode="unspecified", source_text=text)


def infer_variable_depth_range(text: str, variable: Optional[str]) -> Optional[Tuple[float, float]]:
    if not variable:
        return None
    lowered = text.lower()
    aliases = VARIABLE_ALIASES.get(variable, (variable,))
    alias_pattern = "|".join(re.escape(alias.lower()) for alias in sorted(aliases, key=len, reverse=True))
    if variable == "temp" and re.search(r"\bsst\b|sea[- ]surface temperature|surface sst", lowered):
        return (0.0, 0.0)
    upper = re.search(rf"\bupper[-–— ]?(\d+(?:\.\d+)?)\s*(?:m|meter|meters|米).{{0,40}}(?:{alias_pattern})", lowered)
    if upper is None:
        upper = re.search(rf"(?:{alias_pattern}).{{0,40}}\bupper[-–— ]?(\d+(?:\.\d+)?)\s*(?:m|meter|meters|米)", lowered)
    if upper:
        return (0.0, float(upper.group(1)))
    fixed = re.search(rf"\bat\s*(\d+(?:\.\d+)?)\s*(?:m|meter|meters|米).{{0,40}}(?:{alias_pattern})", lowered)
    if fixed is None:
        fixed = re.search(rf"(?:{alias_pattern}).{{0,40}}\bat\s*(\d+(?:\.\d+)?)\s*(?:m|meter|meters|米)", lowered)
    if fixed:
        value = float(fixed.group(1))
        return (value, value)
    if re.search(rf"\b(surface|sea surface)\b.{{0,20}}(?:{alias_pattern})", lowered) or re.search(
        rf"(?:{alias_pattern}).{{0,20}}\b(surface|sea surface)\b",
        lowered,
    ):
        return (0.0, 0.0)
    return None


def resolve_region(
    text: str,
    extracted_params: Mapping[str, Any],
    additional_context: Mapping[str, Any],
    dataset_info: Mapping[str, Any],
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    text_region = extract_region_from_text(text)
    if text_region is not None:
        return text_region

    extracted_region = region_from_mapping(extracted_params)
    if extracted_region is not None:
        return extracted_region

    for candidate in (additional_context.get("workspace_context", {}), additional_context):
        region = region_from_mapping(candidate)
        if region is not None:
            return region

    spatial = dataset_info.get("spatial_extent") if isinstance(dataset_info, Mapping) else {}
    lon = coerce_range(spatial.get("lon") if isinstance(spatial, Mapping) else None) or (0.0, 360.0)
    lat = coerce_range(spatial.get("lat") if isinstance(spatial, Mapping) else None) or (-90.0, 90.0)
    return lon, lat


def region_from_mapping(candidate: Any) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    if not isinstance(candidate, Mapping):
        return None
    lon_range = coerce_range(candidate.get("lon_range") or candidate.get("longitude_range"))
    lat_range = coerce_range(candidate.get("lat_range") or candidate.get("latitude_range"))
    region = candidate.get("region")
    if isinstance(region, Mapping):
        lon_range = lon_range or coerce_range(region.get("lon_range") or region.get("longitude_range"))
        lat_range = lat_range or coerce_range(region.get("lat_range") or region.get("latitude_range"))
    region_bounds = candidate.get("region_bounds") or candidate.get("current_region_bounds")
    if isinstance(region_bounds, Mapping):
        lon_range = lon_range or coerce_range(region_bounds.get("lon") or region_bounds.get("lon_range"))
        lat_range = lat_range or coerce_range(region_bounds.get("lat") or region_bounds.get("lat_range"))
        if lon_range is None and {"lon_min", "lon_max"}.issubset(region_bounds):
            lon_range = (float(region_bounds["lon_min"]), float(region_bounds["lon_max"]))
        if lat_range is None and {"lat_min", "lat_max"}.issubset(region_bounds):
            lat_range = (float(region_bounds["lat_min"]), float(region_bounds["lat_max"]))
        if lon_range is None and {"lonMin", "lonMax"}.issubset(region_bounds):
            lon_range = (float(region_bounds["lonMin"]), float(region_bounds["lonMax"]))
        if lat_range is None and {"latMin", "latMax"}.issubset(region_bounds):
            lat_range = (float(region_bounds["latMin"]), float(region_bounds["latMax"]))
    if lon_range is not None and lat_range is not None:
        return lon_range, lat_range
    return None


def resolve_time_range(
    text: str,
    extracted_params: Mapping[str, Any],
    additional_context: Mapping[str, Any],
    dataset_info: Mapping[str, Any],
) -> Optional[Tuple[str, str]]:
    parsed_time_range = extract_time_range_from_text(text)
    if parsed_time_range is not None:
        return parsed_time_range
    years = [int(match) for match in re.findall(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)", text)]
    if len(years) >= 2:
        return f"{min(years)}-01-01", f"{max(years)}-12-31"
    if len(years) == 1:
        return f"{years[0]}-01-01", f"{years[0]}-12-31"
    time_range = coerce_str_range(extracted_params.get("time_range")) if isinstance(extracted_params, Mapping) else None
    if time_range is not None:
        return time_range
    for candidate in (additional_context.get("workspace_context", {}), additional_context):
        if isinstance(candidate, Mapping):
            time_range = coerce_str_range(candidate.get("time_range"))
            if time_range is not None:
                return time_range
    temporal = dataset_info.get("temporal_extent") if isinstance(dataset_info, Mapping) else {}
    if isinstance(temporal, Mapping) and temporal.get("start") and temporal.get("end"):
        return str(temporal["start"]), str(temporal["end"])
    return None


def has_explicit_region(text: str, extracted_params: Mapping[str, Any], additional_context: Mapping[str, Any]) -> bool:
    return any(
        context_has_region(candidate)
        for candidate in (extracted_params, additional_context.get("workspace_context", {}), additional_context)
    ) or extract_region_from_text(text) is not None


def has_explicit_time_range(text: str, extracted_params: Mapping[str, Any], additional_context: Mapping[str, Any]) -> bool:
    if any(
        context_has_time_range(candidate)
        for candidate in (extracted_params, additional_context.get("workspace_context", {}), additional_context)
    ):
        return True
    if extract_time_range_from_text(text) is not None:
        return True
    return bool(re.search(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)", text))


def extract_region_from_text(text: str) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    lon_patterns = (
        r"(?P<a>-?\d+(?:\.\d+)?)\s*(?:°\s*)?[eE]\s*(?:-|–|—|to|到|至)\s*(?P<b>-?\d+(?:\.\d+)?)\s*(?:°\s*)?[eE]",
        r"(?P<a>-?\d+(?:\.\d+)?)\s*(?:-|–|—|to|到|至)\s*(?P<b>-?\d+(?:\.\d+)?)\s*(?:°\s*)?[eE]",
    )
    lat_patterns = (
        r"(?P<a>-?\d+(?:\.\d+)?)\s*(?:°\s*)?[nN]\s*(?:-|–|—|to|到|至)\s*(?P<b>-?\d+(?:\.\d+)?)\s*(?:°\s*)?[nN]",
        r"(?P<a>-?\d+(?:\.\d+)?)\s*(?:-|–|—|to|到|至)\s*(?P<b>-?\d+(?:\.\d+)?)\s*(?:°\s*)?[nN]",
    )
    lon_range = first_range_match(text, lon_patterns)
    lat_range = first_range_match(text, lat_patterns)
    if lon_range is not None and lat_range is not None:
        return lon_range, lat_range
    named_region = named_region_from_text(text)
    if named_region is not None:
        return named_region
    return None


def named_region_from_text(text: str) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    lowered = text.lower()
    for name, bounds in NAMED_REGION_BOUNDS.items():
        if re.search(rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])", lowered):
            return bounds
    return None


def extract_time_range_from_text(text: str) -> Optional[Tuple[str, str]]:
    year_range = re.search(
        r"(19\d{2}|20\d{2})\s*(?:年)?\s*(?:-|–|—|to|through|until|到|至)\s*(19\d{2}|20\d{2})\s*(?:年)?",
        text,
        flags=re.IGNORECASE,
    )
    if year_range:
        start_year = int(year_range.group(1))
        end_year = int(year_range.group(2))
        return f"{min(start_year, end_year)}-01-01", f"{max(start_year, end_year)}-12-31"

    iso_range = re.search(
        r"(19\d{2}|20\d{2})[-/](\d{1,2})[-/](\d{1,2}).{0,20}?(19\d{2}|20\d{2})[-/](\d{1,2})[-/](\d{1,2})",
        text,
    )
    if iso_range:
        start = f"{int(iso_range.group(1)):04d}-{int(iso_range.group(2)):02d}-{int(iso_range.group(3)):02d}"
        end = f"{int(iso_range.group(4)):04d}-{int(iso_range.group(5)):02d}-{int(iso_range.group(6)):02d}"
        return start, end
    return extract_month_range(text)


MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


def extract_month_range(text: str) -> Optional[Tuple[str, str]]:
    lowered = text.lower()
    month_names = "|".join(sorted(MONTHS, key=len, reverse=True))
    same_year = re.search(
        rf"\b({month_names})\b\s*(?:to|-|–|—|through|until|到|至)\s*\b({month_names})\b\s*(19\d{{2}}|20\d{{2}})",
        lowered,
    )
    if same_year:
        year = int(same_year.group(3))
        return month_bounds(year, MONTHS[same_year.group(1)], year, MONTHS[same_year.group(2)])
    cross_year = re.search(
        rf"\b({month_names})\b\s*(19\d{{2}}|20\d{{2}})\s*(?:to|-|–|—|through|until|到|至)\s*\b({month_names})\b\s*(19\d{{2}}|20\d{{2}})",
        lowered,
    )
    if cross_year:
        return month_bounds(
            int(cross_year.group(2)),
            MONTHS[cross_year.group(1)],
            int(cross_year.group(4)),
            MONTHS[cross_year.group(3)],
        )
    return None


def extract_bottom_band_thickness(text: str) -> Optional[float]:
    patterns = [
        r"(?:bottom|seafloor|near[- ]bottom).{0,20}?(\d+(?:\.\d+)?)\s*m",
        r"底(?:部|层|以上|以内|附近).{0,12}?(\d+(?:\.\d+)?)\s*(?:m|米)",
        r"(\d+(?:\.\d+)?)\s*(?:m|米).{0,12}?(?:bottom|seafloor|底)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def extract_fixed_depth(text: str) -> Optional[float]:
    match = re.search(r"(?:at|固定|深度)\s*(\d+(?:\.\d+)?)\s*(?:m|米)", text, flags=re.IGNORECASE)
    if match:
        return float(match.group(1))
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:m|米)\s*(?:depth|深处|处)", text, flags=re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def extract_depth_range(text: str) -> Optional[Tuple[float, float]]:
    upper = re.search(r"\bupper[-–— ]?(\d+(?:\.\d+)?)\s*(?:m|meter|meters|米)", text, flags=re.IGNORECASE)
    if upper:
        return 0.0, float(upper.group(1))
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:-|–|—|to|到|至)\s*(\d+(?:\.\d+)?)\s*(?:m|米)", text, flags=re.IGNORECASE)
    if match:
        return float(match.group(1)), float(match.group(2))
    match = re.search(r"(?:below|deeper than|深于|以下|大于)\s*(\d+(?:\.\d+)?)\s*(?:m|米)", text, flags=re.IGNORECASE)
    if match:
        return float(match.group(1)), 10000.0
    return None


def extract_threshold(text: str, *, default: Optional[float]) -> Optional[float]:
    match = re.search(r"(?:threshold|阈值|低于|小于|below)\s*(\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if match:
        return float(match.group(1))
    return default


def extract_max_lag(text: str, *, default: int) -> int:
    patterns = (
        r"max(?:imum)?\s+lag\s*(?:of|=|:)?\s*(\d+)",
        r"(\d+)\s*(?:step|month|day|year|步|月|天|年)s?\s+(?:lag|lags|滞后)",
        r"滞后\s*(\d+)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            try:
                return max(0, int(float(match.group(1))))
            except (TypeError, ValueError):
                continue
    return int(default)


def coerce_range(value: Any) -> Optional[Tuple[float, float]]:
    if isinstance(value, Mapping) and {"min", "max"}.issubset(value):
        return float(value["min"]), float(value["max"])
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return float(value[0]), float(value[1])
    return None


def coerce_str_range(value: Any) -> Optional[Tuple[str, str]]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return str(value[0]), str(value[1])
    return None


def coerce_depth_range(value: Any) -> Optional[Tuple[float, float]]:
    numeric = coerce_range(value)
    if numeric is None:
        return None
    return tuple(sorted((abs(float(numeric[0])), abs(float(numeric[1])))))


def context_has_region(candidate: Any) -> bool:
    if not isinstance(candidate, Mapping):
        return False
    lon_range = coerce_range(candidate.get("lon_range") or candidate.get("longitude_range"))
    lat_range = coerce_range(candidate.get("lat_range") or candidate.get("latitude_range"))
    if lon_range is not None and lat_range is not None:
        return True
    region = candidate.get("region")
    if isinstance(region, Mapping):
        lon_range = coerce_range(region.get("lon_range") or region.get("longitude_range"))
        lat_range = coerce_range(region.get("lat_range") or region.get("latitude_range"))
        if lon_range is not None and lat_range is not None:
            return True
    region_bounds = candidate.get("region_bounds") or candidate.get("current_region_bounds")
    if isinstance(region_bounds, Mapping):
        if coerce_range(region_bounds.get("lon") or region_bounds.get("lon_range")) and coerce_range(
            region_bounds.get("lat") or region_bounds.get("lat_range")
        ):
            return True
        return (
            {"lon_min", "lon_max", "lat_min", "lat_max"}.issubset(region_bounds)
            or {"lonMin", "lonMax", "latMin", "latMax"}.issubset(region_bounds)
        )
    return False


def context_has_time_range(candidate: Any) -> bool:
    return isinstance(candidate, Mapping) and coerce_str_range(candidate.get("time_range")) is not None


def contains_alias(text: str, alias: str) -> bool:
    alias_lower = alias.lower()
    if re.fullmatch(r"[a-z0-9_]+", alias_lower):
        return bool(re.search(rf"(?<![a-z0-9_]){re.escape(alias_lower)}(?![a-z0-9_])", text))
    return alias_lower in text


def alias_position(text: str, alias: str) -> Optional[int]:
    alias_lower = alias.lower()
    if re.fullmatch(r"[a-z0-9_]+", alias_lower):
        match = re.search(rf"(?<![a-z0-9_]){re.escape(alias_lower)}(?![a-z0-9_])", text)
    else:
        match = re.search(re.escape(alias_lower), text)
    return match.start() if match else None


def first_range_match(text: str, patterns: Iterable[str]) -> Optional[Tuple[float, float]]:
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            first = float(match.group("a"))
            second = float(match.group("b"))
            return min(first, second), max(first, second)
    return None


def month_bounds(start_year: int, start_month: int, end_year: int, end_month: int) -> Tuple[str, str]:
    last_day = calendar.monthrange(end_year, end_month)[1]
    return f"{start_year:04d}-{start_month:02d}-01", f"{end_year:04d}-{end_month:02d}-{last_day:02d}"


def vertical_to_dict(vertical: VerticalSpec) -> Dict[str, Any]:
    return {
        "mode": vertical.mode,
        "depth_value": vertical.depth_value,
        "depth_range": list(vertical.depth_range) if vertical.depth_range is not None else None,
        "relative_to": vertical.relative_to,
        "band_thickness_m": vertical.band_thickness_m,
        "aggregation": vertical.aggregation,
        "retain_depth": vertical.retain_depth,
    }
