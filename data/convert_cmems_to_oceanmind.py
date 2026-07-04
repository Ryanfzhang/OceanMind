#!/usr/bin/env python3
"""Convert raw CMEMS NetCDF files into OceanMaster per-variable Zarr stores.

Default usage converts the current CMEMS raw directory into stores that match
the active Zarr-only loader layout:

    python data/convert_cmems_to_oceanmind.py --dry-run
    python data/convert_cmems_to_oceanmind.py --overwrite

The output layout is one store per available canonical OceanMaster variable:

    CMEMS_temp.zarr
    CMEMS_salt.zarr
    CMEMS_u.zarr
    CMEMS_v.zarr
    CMEMS_chlorophyll.zarr
    CMEMS_oxygen.zarr

Physics-only surface downloads are supported. If chlorophyll/oxygen files are
not present, the converter writes the available physics variables on their
native grid instead of requiring a biogeochemical reference grid.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.zarr_utils import (
    chunks_for_dataarray as _chunks_for_dataarray,
    chunks_for_dataset as _chunks_for_dataset,
    configure_dask as _configure_dask,
    format_bytes as _format_bytes,
    format_chunks as _format_chunks,
    normalize_chunks as _normalize_chunks_with_defaults,
    open_source_dataset as _open_normalized_dataset,
    prepare_output_path as _prepare_output_path,
    require_runtime_dependencies as _require_runtime_dependencies,
    write_zarr_store as _write_zarr_store,
)


DEFAULT_INPUT_PATH = Path("/import/home4/share/CMEMS_2")
DEFAULT_OUTPUT_PATH = Path("/import/home4/share/CMEMS_oceanmind_2")
OUTPUT_PREFIX = "CMEMS"
TARGET_DIMS = ("time", "depth", "lat", "lon")
DEFAULT_CHUNKS = {
    "time": 30,
    "depth": 16,
    "lat": 110,
    "lon": 86,
}


@dataclass(frozen=True)
class VariableSpec:
    output_name: str
    source_name: str
    source_group: str


VARIABLE_SPECS = (
    VariableSpec("temp", "thetao", "thetao"),
    VariableSpec("salt", "so", "so"),
    VariableSpec("u", "uo", "velocity"),
    VariableSpec("v", "vo", "velocity"),
    VariableSpec("chlorophyll", "chl", "chl"),
    VariableSpec("oxygen", "o2", "o2"),
)


def convert_cmems_to_oceanmind(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    output_prefix: str = OUTPUT_PREFIX,
    variables: Sequence[str] | None = None,
    chunks: Mapping[str, int] | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
    regrid_to_bio_grid: bool = True,
    consolidated: bool = True,
    compressor: str = "zstd",
    compression_level: int = 3,
    scheduler: str = "threads",
    workers: int = 8,
) -> list[Path]:
    """Convert CMEMS raw files into per-variable Zarr stores.

    By default, all variables discoverable in the input directory are
    converted. When chlorophyll or oxygen files are present, physics variables
    are interpolated to that coarser CMEMS biogeochemical grid, preserving the
    previous OceanMind conversion behavior. Physics-only downloads stay on
    their native grid.
    """
    input_dir = Path(input_path).expanduser().resolve()
    output_dir = Path(output_path).expanduser().resolve()
    normalized_chunks = _normalize_chunks(chunks)

    netcdf_files = _iter_netcdf_files(input_dir)
    source_files = _resolve_source_files(netcdf_files)
    selected_specs = _select_variable_specs(variables, source_files)
    _validate_required_sources(source_files, selected_specs)

    planned_outputs = [
        output_dir / f"{output_prefix}_{spec.output_name}.zarr"
        for spec in selected_specs
    ]
    _print_plan(
        input_dir=input_dir,
        output_dir=output_dir,
        source_files=source_files,
        variable_specs=selected_specs,
        planned_outputs=planned_outputs,
        output_prefix=output_prefix,
        chunks=normalized_chunks,
        regrid_to_bio_grid=regrid_to_bio_grid,
        dry_run=dry_run,
    )
    if dry_run:
        return planned_outputs

    _require_runtime_dependencies()
    _configure_dask(scheduler=scheduler, workers=workers)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_coords: dict[str, xr.DataArray] | None = None
    if regrid_to_bio_grid:
        reference_file = source_files["chl"] or source_files["o2"]
        if reference_file is not None:
            target_coords = _load_reference_grid(reference_file, chunks=normalized_chunks)
        else:
            print("No chlorophyll/oxygen reference grid found; converting available variables on native grid.")

    written_stores: list[Path] = []
    for spec in selected_specs:
        source_file = source_files[spec.source_group]
        if source_file is None:
            raise FileNotFoundError(f"No source file found for {spec.output_name}.")

        output_store = output_dir / f"{output_prefix}_{spec.output_name}.zarr"
        _convert_variable_to_zarr(
            spec=spec,
            source_file=source_file,
            output_store=output_store,
            output_prefix=output_prefix,
            target_coords=target_coords,
            chunks=normalized_chunks,
            overwrite=overwrite,
            consolidated=consolidated,
            compressor_name=compressor,
            compression_level=compression_level,
        )
        written_stores.append(output_store)

    return written_stores


def _iter_netcdf_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".nc":
            raise ValueError(f"Input file is not a NetCDF file: {input_path}")
        return [input_path]
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    files = sorted(input_path.rglob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No .nc files found under {input_path}")
    return files


def _resolve_source_files(files: Iterable[Path]) -> dict[str, Path | None]:
    file_list = list(files)
    return {
        "thetao": _find_file(file_list, ("thetao",)),
        "so": _find_file(file_list, ("so",)),
        "velocity": _find_velocity_file(file_list),
        "chl": _find_file(file_list, ("chl",)),
        "o2": _find_file(file_list, ("o2",)),
    }


def _select_variable_specs(
    requested_variables: Sequence[str] | None,
    source_files: Mapping[str, Path | None],
) -> tuple[VariableSpec, ...]:
    alias_map: dict[str, VariableSpec] = {}
    for spec in VARIABLE_SPECS:
        for alias in (spec.output_name, spec.source_name, spec.source_group):
            alias_map[alias.lower()] = spec

    if requested_variables is not None:
        selected: list[VariableSpec] = []
        unknown: list[str] = []
        for raw_name in requested_variables:
            name = str(raw_name).strip().lower()
            if not name:
                continue
            spec = alias_map.get(name)
            if spec is None:
                unknown.append(str(raw_name))
                continue
            if spec not in selected:
                selected.append(spec)
        if unknown:
            known = ", ".join(sorted(alias_map))
            raise ValueError(f"Unknown CMEMS/OceanMaster variable(s): {', '.join(unknown)}. Known names: {known}.")
        if not selected:
            raise ValueError("No variables were selected for conversion.")
        return tuple(selected)

    discovered = [
        spec
        for spec in VARIABLE_SPECS
        if source_files.get(spec.source_group) is not None
    ]
    if not discovered:
        raise FileNotFoundError(
            "No supported CMEMS variables were found. Expected one or more filename keywords: "
            "thetao, so, uo-vo, chl, o2."
        )
    return tuple(discovered)


def _validate_required_sources(
    source_files: Mapping[str, Path | None],
    variable_specs: Sequence[VariableSpec],
) -> None:
    missing = [
        spec.source_group
        for spec in variable_specs
        if source_files.get(spec.source_group) is None
    ]
    if missing:
        formatted = ", ".join(dict.fromkeys(missing))
        raise FileNotFoundError(
            f"Missing required CMEMS source file group(s): {formatted}. "
            "Expected filename keywords: thetao, so, uo-vo, chl, o2."
        )


def _find_file(files: list[Path], keywords: tuple[str, ...]) -> Path | None:
    for keyword in keywords:
        matches = [path for path in files if _filename_has_token(path.name, keyword)]
        if matches:
            return sorted(matches, key=lambda path: (len(path.name), path.name))[0]
    return None


def _find_velocity_file(files: list[Path]) -> Path | None:
    uo_vo_matches = [path for path in files if "uo-vo" in path.name.lower()]
    if uo_vo_matches:
        return sorted(uo_vo_matches, key=lambda path: (len(path.name), path.name))[0]

    matches = [
        path
        for path in files
        if _filename_has_token(path.name, "uo") and _filename_has_token(path.name, "vo")
    ]
    if matches:
        return sorted(matches, key=lambda path: (len(path.name), path.name))[0]
    return None


def _filename_has_token(filename: str, token: str) -> bool:
    pattern = rf"(^|[_\-.]){re.escape(token.lower())}([_\-.]|$)"
    return bool(re.search(pattern, filename.lower()))


def _load_reference_grid(reference_file: Path, *, chunks: Mapping[str, int]) -> dict[str, xr.DataArray]:
    dataset = _open_normalized_dataset(reference_file, chunks=chunks)
    try:
        variable_name = "chl" if "chl" in dataset.data_vars else next(iter(dataset.data_vars))
        reference = _sort_known_coords(dataset[variable_name])
        missing_dims = [dim for dim in TARGET_DIMS if dim not in reference.coords]
        if missing_dims:
            raise ValueError(
                f"Bio reference file {reference_file} is missing coordinate(s): "
                f"{', '.join(missing_dims)}"
            )
        return {dim: reference.coords[dim].load() for dim in TARGET_DIMS}
    finally:
        dataset.close()


def _convert_variable_to_zarr(
    *,
    spec: VariableSpec,
    source_file: Path,
    output_store: Path,
    output_prefix: str,
    target_coords: dict[str, xr.DataArray] | None,
    chunks: Mapping[str, int],
    overwrite: bool,
    consolidated: bool,
    compressor_name: str,
    compression_level: int,
) -> None:
    print(f"Converting {spec.source_name} from {source_file.name} -> {output_store.name}")
    _prepare_output_path(output_store, overwrite=overwrite)

    dataset = _open_normalized_dataset(source_file, chunks=chunks)
    try:
        if spec.source_name not in dataset.data_vars:
            available = ", ".join(dataset.data_vars)
            raise ValueError(
                f"Variable '{spec.source_name}' not found in {source_file}. "
                f"Available variables: {available}"
            )

        source = dataset[spec.source_name]
        source_attrs = dict(source.attrs)
        aligned = _sort_known_coords(source)
        if target_coords is not None:
            aligned = _align_to_target_grid(aligned, target_coords)
        else:
            _require_target_dims(aligned)

        aligned = aligned.transpose(*[dim for dim in TARGET_DIMS if dim in aligned.dims])
        aligned = aligned.astype("float32", keep_attrs=True)
        aligned = aligned.chunk(_chunks_for_dataarray(aligned, chunks))
        aligned.name = spec.output_name
        aligned.attrs = source_attrs
        aligned.attrs.update(
            {
                "source_variable": spec.source_name,
                "source_file": source_file.name,
                "oceanmaster_variable": spec.output_name,
                "grid_strategy": "interpolated_to_bio_grid" if target_coords is not None else "native_grid",
            }
        )

        output_dataset = aligned.to_dataset(name=spec.output_name)
        output_dataset.attrs.update(
            {
                "dataset": output_prefix,
                "converted_by": "data/convert_cmems_to_oceanmind.py",
                "storage_layout": "per_variable_zarr",
                "grid_strategy": "physics_interpolated_to_bio_grid"
                if target_coords is not None
                else "native_grid",
            }
        )
        _write_zarr_store(
            output_dataset,
            output_path=output_store,
            chunks=chunks,
            compressor_name=compressor_name,
            compression_level=compression_level,
            consolidated=consolidated,
        )
    finally:
        dataset.close()


def _sort_known_coords(data: xr.DataArray) -> xr.DataArray:
    result = data
    for coord_name in TARGET_DIMS:
        if coord_name not in result.coords:
            continue
        values = np.asarray(result.coords[coord_name].values)
        if values.size > 1 and values[0] > values[-1]:
            result = result.sortby(coord_name)
    return result


def _align_to_target_grid(
    data: xr.DataArray,
    target_coords: dict[str, xr.DataArray],
) -> xr.DataArray:
    result = data
    for dim in TARGET_DIMS:
        if dim not in result.dims:
            raise ValueError(f"Variable '{data.name}' is missing required dimension '{dim}'.")
        target = target_coords[dim]
        if _coordinates_match(result.coords[dim], target):
            result = result.assign_coords({dim: target})
            continue
        result = _interp_or_select(result, dim, target)
    return result


def _require_target_dims(data: xr.DataArray) -> None:
    missing_dims = [dim for dim in TARGET_DIMS if dim not in data.dims]
    if missing_dims:
        raise ValueError(f"Variable '{data.name}' is missing required dimension(s): {', '.join(missing_dims)}")


def _coordinates_match(source: xr.DataArray, target: xr.DataArray) -> bool:
    if source.size != target.size:
        return False
    if np.issubdtype(source.dtype, np.datetime64) or np.issubdtype(target.dtype, np.datetime64):
        return bool(np.array_equal(source.values, target.values))
    return bool(np.allclose(source.values, target.values, equal_nan=True))


def _interp_or_select(data: xr.DataArray, dim: str, target: xr.DataArray) -> xr.DataArray:
    source_values = data.coords[dim].values
    target_values = target.values

    if np.issubdtype(data.coords[dim].dtype, np.datetime64):
        source_index = set(np.asarray(source_values, dtype="datetime64[ns]").astype("int64").tolist())
        target_index = np.asarray(target_values, dtype="datetime64[ns]").astype("int64").tolist()
        if all(value in source_index for value in target_index):
            return data.sel({dim: target})
        return data.interp({dim: target})

    return data.interp({dim: target})


def _normalize_chunks(chunks: Mapping[str, int] | None) -> dict[str, int]:
    return _normalize_chunks_with_defaults(chunks, DEFAULT_CHUNKS)


def _parse_variable_filter(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    variables = tuple(item.strip() for item in value.split(",") if item.strip())
    return variables or None


def _print_plan(
    *,
    input_dir: Path,
    output_dir: Path,
    source_files: Mapping[str, Path | None],
    variable_specs: Sequence[VariableSpec],
    planned_outputs: Sequence[Path],
    output_prefix: str,
    chunks: Mapping[str, int],
    regrid_to_bio_grid: bool,
    dry_run: bool,
) -> None:
    print("CMEMS raw NetCDF to OceanMaster Zarr conversion plan")
    print(f"  input_dir: {input_dir}")
    print(f"  output_dir: {output_dir}")
    print(f"  output_prefix: {output_prefix}")
    print(f"  layout: per-variable zarr")
    print(f"  chunks: {_format_chunks(chunks)}")
    print(f"  regrid_to_bio_grid: {regrid_to_bio_grid}")
    print(f"  selected_variables: {', '.join(spec.output_name for spec in variable_specs)}")
    print(f"  dry_run: {dry_run}")
    for spec, output_store in zip(variable_specs, planned_outputs):
        source_file = source_files.get(spec.source_group)
        size = _format_bytes(source_file.stat().st_size) if source_file and source_file.exists() else "missing"
        source_name = source_file.name if source_file else "missing"
        print(f"  - {spec.output_name}: {source_name} -> {output_store.name}, source_size={size}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert raw CMEMS NetCDF files to OceanMaster per-variable Zarr stores."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Raw CMEMS .nc directory or file. Default: {DEFAULT_INPUT_PATH}",
    )
    parser.add_argument(
        "output_path",
        nargs="?",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Converted Zarr output directory. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument("--output-prefix", default=OUTPUT_PREFIX, help="Output Zarr store prefix. Default: CMEMS.")
    parser.add_argument(
        "--variables",
        default=None,
        help=(
            "Comma-separated variables to convert. Accepts OceanMaster names "
            "(temp,salt,u,v,chlorophyll,oxygen) or CMEMS names (thetao,so,uo,vo,chl,o2). "
            "Default: convert all variables discovered in input_path."
        ),
    )
    parser.add_argument("--chunk-time", type=int, default=DEFAULT_CHUNKS["time"], help="Dask/Zarr time chunk size.")
    parser.add_argument(
        "--chunk-depth",
        type=int,
        default=DEFAULT_CHUNKS["depth"],
        help="Dask/Zarr depth chunk size.",
    )
    parser.add_argument("--chunk-lat", type=int, default=DEFAULT_CHUNKS["lat"], help="Dask/Zarr latitude chunk size.")
    parser.add_argument("--chunk-lon", type=int, default=DEFAULT_CHUNKS["lon"], help="Dask/Zarr longitude chunk size.")
    parser.add_argument(
        "--compressor",
        choices=("zstd", "lz4", "zlib", "none"),
        default="zstd",
        help="Blosc compressor for data variables. Default: zstd.",
    )
    parser.add_argument("--compression-level", type=int, default=3, help="Compression level. Default: 3.")
    parser.add_argument(
        "--scheduler",
        choices=("threads", "processes", "single-threaded", "synchronous"),
        default="threads",
        help="Dask scheduler for writing. Default: threads.",
    )
    parser.add_argument("--workers", type=int, default=8, help="Dask worker count. Default: 8.")
    parser.add_argument(
        "--keep-native-grid",
        action="store_true",
        help="Do not interpolate physics variables to the BGC reference grid.",
    )
    parser.add_argument(
        "--no-consolidated",
        action="store_true",
        help="Do not write consolidated Zarr metadata.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output stores.")
    parser.add_argument("--dry-run", action="store_true", help="Print conversion plan without writing.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    chunks = {
        "time": args.chunk_time,
        "depth": args.chunk_depth,
        "lat": args.chunk_lat,
        "lon": args.chunk_lon,
    }
    written = convert_cmems_to_oceanmind(
        args.input_path,
        args.output_path,
        output_prefix=args.output_prefix,
        variables=_parse_variable_filter(args.variables),
        chunks=chunks,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        regrid_to_bio_grid=not args.keep_native_grid,
        consolidated=not args.no_consolidated,
        compressor=args.compressor,
        compression_level=args.compression_level,
        scheduler=args.scheduler,
        workers=args.workers,
    )
    action = "Planned" if args.dry_run else "Wrote"
    print(f"Done. {action} {len(written)} Zarr store(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
