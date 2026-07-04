#!/usr/bin/env python3
"""Convert yearly NetCDF ocean files into chunked Zarr stores.

The script is designed for OceanMaster-style yearly files:

    CMOMS_temp_Zlev_2011.nc
    CMEMS_temp_Zlev_2025.nc
    CMOMS_u_Zlev_2011.nc
    ...

It also works for similar ``<prefix>_<variable>_..._<year>.nc`` file names.
Use ``--dry-run`` first on large collections.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.zarr_utils import (
    chunks_for_dataset as _chunks_for_dataset,
    configure_dask as _configure_dask,
    format_bytes as _format_bytes,
    format_chunks as _format_chunks,
    open_source_dataset as _open_source_dataset,
    prepare_output_path as _prepare_output_path,
    require_runtime_dependencies as _require_runtime_dependencies,
    write_zarr_store as _write_zarr_store,
)


DEFAULT_VARIABLES = ("temp", "salt", "u", "v", "chlorophyll", "oxygen")
DEFAULT_CHUNKS = {
    "time": 30,
    "depth": 16,
    "z": 16,
    "lat": 110,
    "lon": 86,
}
YEAR_RE = re.compile(r"(?P<year>\d{4})(?=\.nc$)")


@dataclass(frozen=True)
class SourceFile:
    path: Path
    variable: str
    year: int | None
    prefix: str | None


def main() -> None:
    args = _parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    sources = _discover_sources(
        input_dir=input_dir,
        prefix=args.prefix,
        variables=args.variables,
        years=_parse_years(args.years),
        pattern=args.pattern,
    )
    if args.max_files is not None:
        sources = sources[: args.max_files]
    if not sources:
        raise SystemExit(f"No NetCDF files matched under {input_dir}")

    chunks = _chunk_mapping(args)
    grouped = _group_by_variable(sources)
    output_prefix = args.output_prefix or _infer_output_prefix(sources) or "ocean"
    _print_plan(
        grouped=grouped,
        output_dir=output_dir,
        output_prefix=output_prefix,
        mode=args.mode,
        chunks=chunks,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        return

    _require_runtime_dependencies()
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_dask(scheduler=args.scheduler, workers=args.workers)

    if args.mode == "per-file":
        for source in sources:
            output_path = output_dir / f"{source.path.stem}.zarr"
            _convert_single_file(
                source.path,
                output_path=output_path,
                chunks=chunks,
                compressor_name=args.compressor,
                compression_level=args.compression_level,
                overwrite=args.overwrite,
                normalize_dims=not args.keep_native_dims,
                consolidated=not args.no_consolidated,
            )
        return

    for variable, variable_sources in grouped.items():
        output_path = output_dir / f"{output_prefix}_{variable}.zarr"
        _convert_variable_group(
            variable_sources,
            output_path=output_path,
            chunks=chunks,
            compressor_name=args.compressor,
            compression_level=args.compression_level,
            overwrite=args.overwrite,
            normalize_dims=not args.keep_native_dims,
            consolidated=not args.no_consolidated,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert CMOMS/OceanMaster NetCDF files to chunked Zarr stores."
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing source .nc files.")
    parser.add_argument("output_dir", type=Path, help="Directory where .zarr stores will be written.")
    parser.add_argument(
        "--prefix",
        help=(
            "Optional dataset filename prefix to strip, e.g. CMOMS or CMEMS. "
            "When omitted, the script infers variables from known variable tokens."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        help="Prefix for per-variable Zarr stores. Default: inferred from input filenames.",
    )
    parser.add_argument("--pattern", default="*.nc", help="Glob pattern under input_dir. Default: *.nc.")
    parser.add_argument(
        "--variables",
        nargs="+",
        default=list(DEFAULT_VARIABLES),
        help="Variables to convert. Default: temp salt u v chlorophyll oxygen.",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        help="Years to convert, e.g. 2011:2022 or 2011 2012 2013. Default: all matched years.",
    )
    parser.add_argument(
        "--mode",
        choices=("per-variable", "per-file"),
        default="per-variable",
        help="Write one store per variable or one store per source file. Default: per-variable.",
    )
    parser.add_argument("--chunk-time", type=int, default=30, help="Dask/Zarr time chunk size.")
    parser.add_argument("--chunk-depth", type=int, default=16, help="Dask/Zarr depth/z chunk size.")
    parser.add_argument("--chunk-lat", type=int, default=110, help="Dask/Zarr latitude chunk size.")
    parser.add_argument("--chunk-lon", type=int, default=86, help="Dask/Zarr longitude chunk size.")
    parser.add_argument(
        "--compressor",
        choices=("zstd", "lz4", "zlib", "none"),
        default="zstd",
        help="Blosc compressor for data variables. Default: zstd.",
    )
    parser.add_argument("--compression-level", type=int, default=3, help="Compression level.")
    parser.add_argument(
        "--scheduler",
        choices=("threads", "processes", "single-threaded", "synchronous"),
        default="threads",
        help="Dask scheduler for writing. Default: threads.",
    )
    parser.add_argument("--workers", type=int, default=8, help="Dask worker count for local writes.")
    parser.add_argument(
        "--keep-native-dims",
        action="store_true",
        help="Do not rename z/longitude/latitude dimensions to depth/lon/lat.",
    )
    parser.add_argument(
        "--no-consolidated",
        action="store_true",
        help="Do not write consolidated Zarr metadata.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output stores.")
    parser.add_argument("--dry-run", action="store_true", help="Print conversion plan without writing.")
    parser.add_argument("--max-files", type=int, help="Debug helper: convert at most this many files.")
    return parser.parse_args()


def _parse_years(values: Sequence[str] | None) -> set[int] | None:
    if not values:
        return None

    years: set[int] = set()
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        if ":" in text or "-" in text:
            separator = ":" if ":" in text else "-"
            start_text, end_text = text.split(separator, 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                start, end = end, start
            years.update(range(start, end + 1))
        else:
            years.add(int(text))
    return years


def _discover_sources(
    *,
    input_dir: Path,
    prefix: str | None,
    variables: Sequence[str],
    years: set[int] | None,
    pattern: str,
) -> list[SourceFile]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    variable_set = {variable.lower() for variable in variables}
    sources: list[SourceFile] = []
    for path in sorted(input_dir.glob(pattern)):
        if not path.is_file() or path.suffix.lower() != ".nc":
            continue
        variable, inferred_prefix = _infer_variable_and_prefix(
            path,
            prefix=prefix,
            known_variables=variable_set,
        )
        if variable is None or variable.lower() not in variable_set:
            continue
        year = _infer_year(path)
        if years is not None and year not in years:
            continue
        sources.append(
            SourceFile(
                path=path,
                variable=variable,
                year=year,
                prefix=inferred_prefix,
            )
        )

    return sorted(sources, key=lambda source: (source.variable, source.year or -1, source.path.name))


def _infer_variable_and_prefix(
    path: Path,
    *,
    prefix: str | None,
    known_variables: set[str],
) -> tuple[str | None, str | None]:
    stem = path.stem
    inferred_prefix: str | None = None
    if prefix and stem.startswith(f"{prefix}_"):
        stem = stem[len(prefix) + 1 :]
        inferred_prefix = prefix

    year = _infer_year(path)
    if year is not None:
        stem = re.sub(rf"[_-]?{year}$", "", stem)

    stem = re.sub(r"[_-]?Zlev$", "", stem, flags=re.IGNORECASE)
    parts = [part for part in re.split(r"[_-]+", stem) if part]
    if not parts:
        return None, inferred_prefix

    lowered_parts = [part.lower() for part in parts]
    for index, part in enumerate(lowered_parts):
        if part in known_variables:
            if inferred_prefix is None and index > 0:
                inferred_prefix = "_".join(parts[:index])
            return parts[index], inferred_prefix
    return parts[0], inferred_prefix


def _infer_year(path: Path) -> int | None:
    match = YEAR_RE.search(path.name)
    if match is None:
        return None
    return int(match.group("year"))


def _group_by_variable(sources: Iterable[SourceFile]) -> dict[str, list[SourceFile]]:
    grouped: dict[str, list[SourceFile]] = defaultdict(list)
    for source in sources:
        grouped[source.variable].append(source)
    return {
        variable: sorted(items, key=lambda source: (source.year or -1, source.path.name))
        for variable, items in sorted(grouped.items())
    }


def _infer_output_prefix(sources: Sequence[SourceFile]) -> str | None:
    prefixes = [source.prefix for source in sources if source.prefix]
    if not prefixes:
        return None
    counts: dict[str, int] = defaultdict(int)
    for prefix in prefixes:
        counts[prefix] += 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _chunk_mapping(args: argparse.Namespace) -> dict[str, int]:
    return {
        "time": max(1, int(args.chunk_time)),
        "depth": max(1, int(args.chunk_depth)),
        "z": max(1, int(args.chunk_depth)),
        "lat": max(1, int(args.chunk_lat)),
        "lon": max(1, int(args.chunk_lon)),
    }


def _print_plan(
    *,
    grouped: Mapping[str, Sequence[SourceFile]],
    output_dir: Path,
    output_prefix: str,
    mode: str,
    chunks: Mapping[str, int],
    dry_run: bool,
) -> None:
    files = [source.path for sources in grouped.values() for source in sources]
    total_bytes = sum(path.stat().st_size for path in files)
    print("NetCDF to Zarr conversion plan")
    print(f"  mode: {mode}")
    print(f"  output_dir: {output_dir}")
    print(f"  output_prefix: {output_prefix}")
    print(f"  chunks: {_format_chunks(chunks)}")
    print(f"  source_files: {len(files)}")
    print(f"  source_disk_size: {_format_bytes(total_bytes)}")
    print(f"  dry_run: {dry_run}")
    for variable, sources in grouped.items():
        years = [source.year for source in sources if source.year is not None]
        year_text = f"{min(years)}-{max(years)}" if years else "unknown"
        size = sum(source.path.stat().st_size for source in sources)
        print(f"  - {variable}: {len(sources)} file(s), years={year_text}, size={_format_bytes(size)}")


def _convert_single_file(
    source_path: Path,
    *,
    output_path: Path,
    chunks: Mapping[str, int],
    compressor_name: str,
    compression_level: int,
    overwrite: bool,
    normalize_dims: bool,
    consolidated: bool,
) -> None:
    print(f"Converting {source_path.name} -> {output_path.name}")
    _prepare_output_path(output_path, overwrite=overwrite)
    dataset = _open_source_dataset(source_path, chunks=chunks, normalize_dims=normalize_dims)
    try:
        _write_zarr_store(
            dataset,
            output_path=output_path,
            chunks=chunks,
            compressor_name=compressor_name,
            compression_level=compression_level,
            consolidated=consolidated,
        )
    finally:
        dataset.close()


def _convert_variable_group(
    sources: Sequence[SourceFile],
    *,
    output_path: Path,
    chunks: Mapping[str, int],
    compressor_name: str,
    compression_level: int,
    overwrite: bool,
    normalize_dims: bool,
    consolidated: bool,
) -> None:
    print(f"Converting {len(sources)} file(s) -> {output_path.name}")
    _prepare_output_path(output_path, overwrite=overwrite)
    datasets = [
        _open_source_dataset(source.path, chunks=chunks, normalize_dims=normalize_dims)
        for source in sources
    ]
    try:
        combined = _combine_yearly_datasets(datasets)
        combined = combined.chunk(_chunks_for_dataset(combined, chunks))
        _write_zarr_store(
            combined,
            output_path=output_path,
            chunks=chunks,
            compressor_name=compressor_name,
            compression_level=compression_level,
            consolidated=consolidated,
        )
    finally:
        for dataset in datasets:
            dataset.close()


def _combine_yearly_datasets(datasets: Sequence[xr.Dataset]) -> xr.Dataset:
    if len(datasets) == 1:
        return datasets[0]
    if all("time" in dataset.dims or "time" in dataset.coords for dataset in datasets):
        combined = xr.concat(
            datasets,
            dim="time",
            data_vars="minimal",
            coords="minimal",
            compat="override",
            combine_attrs="override",
        )
        return combined.sortby("time")
    return xr.merge(datasets, compat="override", combine_attrs="override")


if __name__ == "__main__":
    main()
