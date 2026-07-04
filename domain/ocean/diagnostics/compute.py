"""
海洋诊断量计算工具

包含密度、导出场、垂直积分等诊断量计算
"""

import numpy as np
import xarray as xr
from typing import Any, Dict, Tuple, Optional, Literal

from packages.runtime.dataset_config import get_active_dataset_config
from domain.ocean.data_access.partitioned import find_partitioned_values, materialize_partitioned_xarray
from domain.ocean.dask_utils import chunk_summary, dataarray_to_numpy, is_dask_backed, report_phase


def compute_density(
    data: xr.Dataset,
    temp_var: Optional[str] = None,
    salt_var: Optional[str] = None,
) -> xr.DataArray:
    """
    计算海水位势密度（使用TEOS-10）

    使用gsw包计算位势密度，需要温度、盐度和深度数据

    Args:
        data: 包含温度和盐度的数据集
        temp_var: 温度变量名
        salt_var: 盐度变量名

    Returns:
        密度场（kg/m³）

    Example:
        >>> ds = xr.open_dataset('ocean_data.nc')
        >>> density = compute_density(ds)
        >>> print(f"Density range: {density.min().values:.2f} - {density.max().values:.2f} kg/m³")
    """
    if find_partitioned_values((data,)):
        from packages.tool_loader.partitioned_execution import execute_partition_aware

        return execute_partition_aware(
            tool_name="compute_density",
            tool_func=compute_density,
            params={"data": data, "temp_var": temp_var, "salt_var": salt_var},
        )
    data = materialize_partitioned_xarray(data)

    try:
        import gsw
    except ImportError:
        raise ImportError("gsw package required. Install with: pip install gsw")

    # Resolve variable names from config when not explicitly provided
    config = get_active_dataset_config()
    if temp_var is None:
        temp_var = config.resolve_variable('temp')
    if salt_var is None:
        salt_var = config.resolve_variable('salt')

    if temp_var not in data:
        raise ValueError(f"Temperature variable '{temp_var}' not found in dataset")
    if salt_var not in data:
        if salt_var == 'salinity' and 'salt' in data:
            salt_var = 'salt'
        else:
            raise ValueError(f"Salinity variable '{salt_var}' not found in dataset")

    depth_name = _get_depth_name(data)
    if depth_name is None:
        raise ValueError("Depth coordinate required for density calculation")
    if 'lon' not in data.coords or 'lat' not in data.coords:
        raise ValueError("Longitude and latitude coordinates are required for density calculation")

    temp = data[temp_var]
    salt = data[salt_var]
    temp, salt = xr.align(temp, salt, join='inner')

    uses_dask = is_dask_backed((temp, salt))
    report_phase(
        phase="preparing_compute",
        message="Preparing density calculation",
        percent=0.0,
        compute_backend="dask" if uses_dask else "xarray",
        chunks=chunk_summary(xr.Dataset({"temp": temp, "salt": salt})) if uses_dask else None,
    )

    depth_coord = xr.DataArray(data[depth_name], coords={depth_name: data[depth_name]}, dims=(depth_name,))
    lat_coord = xr.DataArray(data.lat, coords={'lat': data.lat}, dims=('lat',))
    lon_coord = xr.DataArray(data.lon, coords={'lon': data.lon}, dims=('lon',))

    if uses_dask:
        report_phase(
            phase="preparing_density_coordinate_grids",
            message="Preparing broadcast-ready coordinate arrays",
            percent=0.05,
            compute_backend="dask",
            chunks=chunk_summary(temp),
        )

    # TEOS-10 calculation. Keep coordinates low-dimensional and let xarray
    # broadcast them against the Zarr-backed fields inside each Dask block.
    p = xr.apply_ufunc(
        gsw.p_from_z,
        depth_coord,
        lat_coord,
        dask="parallelized",
        output_dtypes=[float],
    )

    if uses_dask:
        report_phase(
            phase="density_coordinate_grids_prepared",
            message="Prepared broadcast-ready coordinate arrays",
            percent=0.1,
            compute_backend="dask",
            chunks=chunk_summary(temp),
        )

    SA = xr.apply_ufunc(
        gsw.SA_from_SP,
        salt,
        p,
        lon_coord,
        lat_coord,
        dask="parallelized",
        output_dtypes=[float],
    )
    CT = xr.apply_ufunc(
        gsw.CT_from_t,
        SA,
        temp,
        p,
        dask="parallelized",
        output_dtypes=[float],
    )
    density = xr.apply_ufunc(
        gsw.rho,
        SA,
        CT,
        0.0,
        dask="parallelized",
        output_dtypes=[float],
    )
    density = density.assign_coords(temp.coords)
    density.attrs = {
        'long_name': 'Potential Density',
        'units': 'kg/m³',
        'reference_pressure': '0 dbar'
    }
    density.name = 'density'

    if is_dask_backed(density):
        report_phase(
            phase="lazy_result_prepared",
            message="Prepared lazy density field",
            percent=1.0,
            compute_backend="dask",
            chunks=chunk_summary(density),
        )

    return density


def compute_derived_field(
    data: Optional[xr.Dataset] = None,
    field_type: Optional[Literal['vorticity', 'speed', 'horizontal_gradient', 'vertical_gradient', 'buoyancy_frequency']] = None,
    variable: Optional[str] = None,
    dataset: Optional[xr.Dataset] = None,
    u: Optional[xr.DataArray] = None,
    v: Optional[xr.DataArray] = None,
    density: Optional[xr.DataArray] = None,
    field: Optional[xr.DataArray] = None,
    temp: Optional[xr.DataArray] = None,
    salt: Optional[xr.DataArray] = None,
) -> xr.DataArray:
    """
    计算导出物理场

    支持：涡度、流速、水平梯度、垂直梯度、浮力频率

    Args:
        data: 输入数据集
        field_type: 场类型
            - 'vorticity': 相对涡度 (需要u,v)
            - 'speed': 流速 (需要u,v)
            - 'horizontal_gradient': 水平梯度 (需要variable)
            - 'vertical_gradient': 垂直梯度 (需要variable和depth)
            - 'buoyancy_frequency': 浮力频率 (需要density和depth)
        variable: 变量名（用于梯度计算）

    Returns:
        导出场

    Example:
        >>> ds = xr.open_dataset('current.nc')
        >>> vorticity = compute_derived_field(ds, 'vorticity')
        >>> speed = compute_derived_field(ds, 'speed')
    """
    if find_partitioned_values((data, dataset, u, v, density, field, temp, salt)):
        from packages.tool_loader.partitioned_execution import execute_partition_aware

        return execute_partition_aware(
            tool_name="compute_derived_field",
            tool_func=compute_derived_field,
            params={
                "data": data,
                "field_type": field_type,
                "variable": variable,
                "dataset": dataset,
                "u": u,
                "v": v,
                "density": density,
                "field": field,
                "temp": temp,
                "salt": salt,
            },
        )
    data = materialize_partitioned_xarray(data)
    dataset = materialize_partitioned_xarray(dataset)
    u = materialize_partitioned_xarray(u)
    v = materialize_partitioned_xarray(v)
    density = materialize_partitioned_xarray(density)
    field = materialize_partitioned_xarray(field)
    temp = materialize_partitioned_xarray(temp)
    salt = materialize_partitioned_xarray(salt)

    if field_type is None:
        raise ValueError("field_type is required for compute_derived_field")

    data = _normalize_derived_field_input(
        data=data,
        dataset=dataset,
        u=u,
        v=v,
        density=density,
        field=field,
        temp=temp,
        salt=salt,
        variable=variable,
        field_type=field_type,
    )

    if field_type == 'vorticity':
        if 'u' not in data or 'v' not in data:
            raise ValueError("u and v variables required for vorticity")
        return _compute_vorticity(data)

    elif field_type == 'speed':
        if 'u' not in data or 'v' not in data:
            raise ValueError("u and v variables required for speed")
        return _compute_speed(data)

    elif field_type == 'horizontal_gradient':
        if not variable or variable not in data:
            raise ValueError(f"Variable '{variable}' required for horizontal_gradient")
        return _compute_horizontal_gradient(data, variable)

    elif field_type == 'vertical_gradient':
        if not variable or variable not in data:
            raise ValueError(f"Variable '{variable}' required for vertical_gradient")
        if 'depth' not in data.coords:
            raise ValueError("Depth coordinate required for vertical_gradient")
        return _compute_vertical_gradient(data, variable)

    elif field_type == 'buoyancy_frequency':
        if 'density' not in data:
            raise ValueError("Density variable required. Run compute_density first.")
        if 'depth' not in data.coords:
            raise ValueError("Depth coordinate required for buoyancy_frequency")
        return _compute_buoyancy_frequency(data)

    else:
        raise ValueError(f"Unknown field_type: {field_type}")


def compute_spatial_vorticity_map(
    u: xr.DataArray,
    v: xr.DataArray,
    time_range: Optional[Tuple[str, str]] = None,
    time_aggregation: Literal['mean', 'max', 'min', 'std', 'median'] = 'mean',
    depth_range: Optional[Tuple[float, float]] = None,
    depth_aggregation: Literal['mean', 'max', 'min', 'surface'] = 'mean',
    mask: Optional[xr.DataArray] = None,
) -> Dict:
    """
    Compute a map-ready relative vorticity field after reducing u/v first.

    This fast path avoids building a full 4D vorticity field for requests that
    ultimately need a time/depth aggregated spatial map.
    """
    report_phase(
        phase="prepare_vorticity_inputs",
        message="Preparing relative vorticity inputs",
        percent=0.02,
        chunks={"u": chunk_summary(u), "v": chunk_summary(v)},
    )
    if find_partitioned_values((u, v, mask)):
        from packages.tool_loader.partitioned_execution import execute_partition_aware

        return execute_partition_aware(
            tool_name="compute_spatial_vorticity_map",
            tool_func=compute_spatial_vorticity_map,
            params={
                "u": u,
                "v": v,
                "time_range": time_range,
                "time_aggregation": time_aggregation,
                "depth_range": depth_range,
                "depth_aggregation": depth_aggregation,
                "mask": mask,
            },
        )
    u = materialize_partitioned_xarray(u)
    v = materialize_partitioned_xarray(v)
    mask = materialize_partitioned_xarray(mask)

    report_phase(
        phase="align_vorticity_inputs",
        message="Aligning velocity fields",
        percent=0.08,
    )
    u_field, v_field = xr.align(u, v, join='inner')
    if 'lat' not in u_field.dims or 'lon' not in u_field.dims:
        raise ValueError("compute_spatial_vorticity_map requires lat and lon dimensions")

    if time_range is not None and 'time' in u_field.coords:
        report_phase(
            phase="subset_vorticity_time",
            message="Subsetting velocity fields by time",
            percent=0.14,
        )
        u_field = u_field.sel(time=slice(*time_range))
        v_field = v_field.sel(time=slice(*time_range))

    if mask is not None:
        report_phase(
            phase="apply_vorticity_mask",
            message="Applying vorticity mask",
            percent=0.2,
        )
        u_field = u_field.where(mask)
        v_field = v_field.where(mask)

    had_depth_dim = _get_depth_name(u_field) is not None or _get_depth_name(v_field) is not None
    report_phase(
        phase="aggregate_vorticity_depth",
        message="Aggregating velocity fields over depth",
        percent=0.28,
        chunks={"u": chunk_summary(u_field), "v": chunk_summary(v_field)},
    )
    u_field = _aggregate_vorticity_depth(u_field, depth_range, depth_aggregation)
    v_field = _aggregate_vorticity_depth(v_field, depth_range, depth_aggregation)
    report_phase(
        phase="aggregate_vorticity_time",
        message="Aggregating velocity fields over time",
        percent=0.42,
        chunks={"u": chunk_summary(u_field), "v": chunk_summary(v_field)},
    )
    u_field = _aggregate_vorticity_time(u_field, time_aggregation)
    v_field = _aggregate_vorticity_time(v_field, time_aggregation)

    report_phase(
        phase="build_vorticity_field",
        message="Building relative vorticity field",
        percent=0.55,
        chunks={"u": chunk_summary(u_field), "v": chunk_summary(v_field)},
    )
    u_field, v_field = xr.align(u_field, v_field, join='inner')
    vorticity = _compute_vorticity(
        xr.Dataset({
            'u': u_field.rename('u'),
            'v': v_field.rename('v'),
        })
    )
    if 'lat' not in vorticity.dims or 'lon' not in vorticity.dims:
        raise ValueError("Aggregated vorticity must retain lat and lon dimensions")

    ordered = vorticity.transpose('lat', 'lon')
    values = dataarray_to_numpy(
        ordered,
        label="relative vorticity map",
        dtype=float,
        start=0.62,
        end=0.96,
    )
    units = ordered.attrs.get('units', 's^-1')
    report_phase(
        phase="package_vorticity_map",
        message="Packaging relative vorticity map",
        percent=0.98,
    )

    return {
        'lon': ordered.lon.values.tolist(),
        'lat': ordered.lat.values.tolist(),
        'values': values,
        'metadata': {
            'variable': 'relative_vorticity',
            'units': units,
            'unit': units,
            'time_range': list(time_range) if time_range is not None else None,
            'time_aggregation': time_aggregation if 'time' in u.dims or 'time' in v.dims else None,
            'depth_range': list(depth_range) if depth_range is not None else None,
            'depth_aggregation': depth_aggregation if had_depth_dim else None,
            'statistics': _compute_map_statistics(values),
            'estimation_method': 'computed after time/depth aggregation of u and v',
        },
    }


def _normalize_derived_field_input(
    data: Optional[xr.Dataset],
    dataset: Optional[xr.Dataset],
    u: Optional[xr.DataArray],
    v: Optional[xr.DataArray],
    density: Optional[xr.DataArray],
    field: Optional[xr.DataArray],
    temp: Optional[xr.DataArray],
    salt: Optional[xr.DataArray],
    variable: Optional[str],
    field_type: str,
) -> xr.Dataset:
    candidate = data if data is not None else dataset
    resolved = _unwrap_dataset_candidate(candidate)
    if resolved is not None:
        return resolved

    variables: Dict[str, xr.DataArray] = {}

    if u is not None:
        variables['u'] = _unwrap_dataarray_candidate(u, 'u')
    if v is not None:
        variables['v'] = _unwrap_dataarray_candidate(v, 'v')
    if density is not None:
        variables['density'] = _unwrap_dataarray_candidate(density, 'density')
    if temp is not None:
        variables['temp'] = _unwrap_dataarray_candidate(temp, 'temp')
    if salt is not None:
        variables['salt'] = _unwrap_dataarray_candidate(salt, 'salt')

    if field is not None:
        field_array = _unwrap_dataarray_candidate(field, variable or 'field')
        variables[field_array.name or variable or 'field'] = field_array
    elif variable and candidate is not None:
        extracted = _extract_named_array(candidate, variable)
        if extracted is not None:
            variables[variable] = extracted

    if variables:
        return xr.Dataset(variables)

    raise ValueError(
        "compute_derived_field expects 'data'/'dataset' as an xarray Dataset or "
        "compatible wrapped result, or explicit arrays such as u/v/density/field."
    )


def _unwrap_dataset_candidate(value: object) -> Optional[xr.Dataset]:
    if value is None:
        return None
    if isinstance(value, xr.Dataset):
        return value
    if isinstance(value, dict):
        data = value.get('data')
        if isinstance(data, xr.Dataset):
            return data
    return None


def _unwrap_dataarray_candidate(value: object, fallback_name: str) -> xr.DataArray:
    if isinstance(value, xr.DataArray):
        if value.name:
            return value
        return value.rename(fallback_name)
    if isinstance(value, dict):
        data = value.get('data')
        if isinstance(data, xr.DataArray):
            if data.name:
                return data
            return data.rename(fallback_name)
        if isinstance(data, xr.Dataset):
            if fallback_name in data:
                return data[fallback_name]
            data_vars = list(data.data_vars)
            if len(data_vars) == 1:
                return data[data_vars[0]].rename(fallback_name)
    raise ValueError(f"Expected a DataArray-like input for '{fallback_name}'")


def _extract_named_array(value: object, variable: str) -> Optional[xr.DataArray]:
    if isinstance(value, xr.Dataset) and variable in value:
        return value[variable]
    if isinstance(value, dict):
        data = value.get('data')
        if isinstance(data, xr.Dataset) and variable in data:
            return data[variable]
    return None


def _aggregate_vorticity_depth(
    data: xr.DataArray,
    depth_range: Optional[Tuple[float, float]],
    depth_aggregation: str,
) -> xr.DataArray:
    depth_dim = _get_depth_name(data)
    if depth_dim is None:
        return data

    field = data
    if depth_range is not None:
        from domain.ocean.data_access.load import _build_coord_slice, _normalize_depth_range

        depth_values = np.asarray(field[depth_dim].values, dtype=float)
        normalized_range = _normalize_depth_range(depth_values, depth_range)
        field = field.sel({depth_dim: _build_coord_slice(depth_values, normalized_range)})

    if depth_aggregation == 'surface':
        return field.isel({depth_dim: 0})
    if depth_aggregation in {'mean', 'max', 'min'}:
        return _aggregate_vorticity_dimension(field, depth_aggregation, [depth_dim])
    raise ValueError(f"Unknown depth_aggregation: {depth_aggregation}")


def _aggregate_vorticity_time(data: xr.DataArray, time_aggregation: str) -> xr.DataArray:
    if 'time' not in data.dims:
        return data
    if time_aggregation in {'mean', 'max', 'min', 'std', 'median'}:
        return _aggregate_vorticity_dimension(data, time_aggregation, ['time'])
    raise ValueError(f"Unknown time_aggregation: {time_aggregation}")


def _aggregate_vorticity_dimension(data: xr.DataArray, aggregation: str, dims: list[str]) -> xr.DataArray:
    if aggregation == 'mean':
        return data.mean(dim=dims, skipna=True)
    if aggregation == 'max':
        return data.max(dim=dims, skipna=True)
    if aggregation == 'min':
        return data.min(dim=dims, skipna=True)
    if aggregation == 'std':
        return data.std(dim=dims, skipna=True)
    if aggregation == 'median':
        return data.median(dim=dims, skipna=True)
    raise ValueError(f"Unknown aggregation: {aggregation}")


def _compute_map_statistics(values: np.ndarray) -> Dict:
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return {'mean': None, 'std': None, 'min': None, 'max': None, 'n_valid': 0}
    return {
        'mean': float(np.mean(valid)),
        'std': float(np.std(valid)),
        'min': float(np.min(valid)),
        'max': float(np.max(valid)),
        'n_valid': int(valid.size),
    }


def _compute_vorticity(data: xr.Dataset) -> xr.DataArray:
    """计算相对涡度: dv/dx - du/dy"""
    u = data['u']
    v = data['v']

    # xarray.differentiate on lon/lat coordinates returns derivatives per degree.
    # Convert those degree-based derivatives to meter-based derivatives.
    lon = data.lon
    lat = data.lat
    dx = _calculate_dx(lon, lat)
    dy = _calculate_dy(lat)

    _validate_vorticity_spatial_dims(u)
    _validate_vorticity_spatial_dims(v)
    v = _ensure_spatial_gradient_chunks(v, 'lon')
    u = _ensure_spatial_gradient_chunks(u, 'lat')

    # 计算梯度
    dvdx = v.differentiate('lon') / dx
    dudy = u.differentiate('lat') / dy

    vorticity = dvdx - dudy
    vorticity.name = 'relative_vorticity'
    vorticity.attrs = {
        'long_name': 'Relative Vorticity',
        'units': 's^-1',
        'field_type': 'vorticity',
    }
    if is_dask_backed(vorticity):
        report_phase(
            phase="lazy_result_prepared",
            message="Prepared lazy relative vorticity field",
            percent=1.0,
            compute_backend="dask",
        )

    return vorticity


def _validate_vorticity_spatial_dims(data: xr.DataArray) -> None:
    if 'lat' not in data.dims or 'lon' not in data.dims:
        raise ValueError("relative vorticity requires lat and lon dimensions")
    if data.sizes.get('lat', 0) < 3 or data.sizes.get('lon', 0) < 3:
        raise ValueError("relative vorticity needs at least 3 grid points along both lat and lon")


def _ensure_spatial_gradient_chunks(data: xr.DataArray, dim: str) -> xr.DataArray:
    """Dask gradient needs every differentiated spatial chunk to contain enough points."""
    chunks = getattr(data, "chunks", None)
    if not chunks or dim not in data.dims:
        return data

    axis = data.get_axis_num(dim)
    dim_chunks = chunks[axis]
    if dim_chunks and min(int(value) for value in dim_chunks) <= 2:
        report_phase(
            phase="preparing_gradient_chunks",
            message=f"Rechunking {dim} for relative vorticity gradient",
            percent=0.0,
            compute_backend="dask",
        )
        return data.chunk({dim: -1})
    return data


def _compute_speed(data: xr.Dataset) -> xr.DataArray:
    """计算流速: sqrt(u² + v²)"""
    u = data['u']
    v = data['v']

    speed = np.sqrt(u**2 + v**2)
    speed.name = 'current_speed'
    speed.attrs = {
        'long_name': 'Current Speed',
        'units': 'm/s',
        'field_type': 'speed',
    }

    return speed


def _compute_horizontal_gradient(data: xr.Dataset, variable: str) -> xr.DataArray:
    """计算水平梯度幅度: sqrt((df/dx)² + (df/dy)²)"""
    field = data[variable]

    lon = data.lon
    lat = data.lat
    dx = _calculate_dx(lon, lat)
    dy = _calculate_dy(lat)

    dfdx = field.differentiate('lon') / dx
    dfdy = field.differentiate('lat') / dy

    gradient = np.sqrt(dfdx**2 + dfdy**2)
    gradient.attrs = {
        'long_name': f'Horizontal Gradient of {variable}',
        'units': f"{field.attrs.get('units', '')} / m"
    }

    return gradient


def _compute_vertical_gradient(data: xr.Dataset, variable: str) -> xr.DataArray:
    """计算垂直梯度: df/dz"""
    field = data[variable]

    dfdz = field.differentiate('depth')
    dfdz.attrs = {
        'long_name': f'Vertical Gradient of {variable}',
        'units': f"{field.attrs.get('units', '')} / m"
    }

    return dfdz


def _compute_buoyancy_frequency(data: xr.Dataset) -> xr.DataArray:
    """计算浮力频率: N² = -(g/ρ₀) * dρ/dz"""
    g = 9.81  # m/s²
    rho0 = 1025.0  # kg/m³

    density = data['density']
    drhodz = density.differentiate('depth')

    N2 = -(g / rho0) * drhodz
    N2.attrs = {
        'long_name': 'Buoyancy Frequency Squared',
        'units': 's⁻²'
    }

    return N2


def _calculate_dx(lon: Any, lat: Any) -> xr.DataArray | float:
    """Return meters per native longitude coordinate unit at each latitude."""
    lon_coord = _as_coordinate_array(lon, "lon")
    lat_coord = _as_coordinate_array(lat, "lat")
    unit_scale = _coordinate_unit_to_meter_scale(lon_coord, axis="lon", lat_coord=lat_coord)
    if np.isscalar(unit_scale):
        return float(unit_scale)

    lat_values = np.asarray(lat_coord.values, dtype=float)
    return xr.DataArray(
        unit_scale,
        coords={'lat': lat_coord.values},
        dims=('lat',),
        name='meters_per_lon_coordinate_unit',
    ).where(np.isfinite(lat_values), np.nan)


def _calculate_dy(lat: Any) -> float:
    """Return meters per native latitude coordinate unit."""
    lat_coord = _as_coordinate_array(lat, "lat")
    return float(_coordinate_unit_to_meter_scale(lat_coord, axis="lat"))


def _as_coordinate_array(value: Any, default_name: str) -> xr.DataArray:
    if isinstance(value, xr.DataArray):
        return value
    array = np.asarray(value, dtype=float)
    return xr.DataArray(array, coords={default_name: array}, dims=(default_name,), name=default_name)


def _coordinate_unit_to_meter_scale(
    coord: xr.DataArray,
    *,
    axis: Literal["lon", "lat"],
    lat_coord: Optional[xr.DataArray] = None,
) -> np.ndarray | float:
    units = _normalize_units(coord.attrs.get("units"))
    if _is_meter_unit(units):
        return 1.0
    if _is_kilometer_unit(units):
        return 1000.0

    R = 6371000  # 地球半径（米）
    angular_scale = 1.0 if _is_radian_unit(units) else np.pi / 180.0
    if axis == "lat":
        return R * angular_scale

    if lat_coord is None:
        raise ValueError("lat_coord is required for longitude metric conversion")
    lat_values = np.asarray(lat_coord.values, dtype=float)
    lat_radians = lat_values if _is_radian_unit(_normalize_units(lat_coord.attrs.get("units"))) else np.deg2rad(lat_values)
    meters = R * np.cos(lat_radians) * angular_scale
    return np.where(np.abs(meters) > 1e-12, meters, np.nan)


def _normalize_units(units: Any) -> str:
    return str(units or "").strip().lower().replace("_", " ").replace("-", " ")


def _is_meter_unit(units: str) -> bool:
    compact = units.replace(" ", "")
    return compact in {"m", "meter", "meters", "metre", "metres"}


def _is_kilometer_unit(units: str) -> bool:
    compact = units.replace(" ", "")
    return compact in {"km", "kilometer", "kilometers", "kilometre", "kilometres"}


def _is_radian_unit(units: str) -> bool:
    compact = units.replace(" ", "")
    return compact in {"rad", "radian", "radians"}


def _get_depth_name(data: xr.Dataset) -> Optional[str]:
    """Return the normalized depth coordinate name if present."""
    for name in ('depth', 'z'):
        if name in data.coords or name in data.dims:
            return name
    return None


def compute_vertical_integral(
    data: xr.DataArray,
    depth_range: Tuple[float, float]
) -> xr.DataArray:
    """
    计算垂直积分

    在指定深度范围内对场进行垂直积分（梯形法则）

    Args:
        data: 输入数据（需要depth维度）
        depth_range: 深度范围 [浅, 深]，例如 [0, -200]

    Returns:
        垂直积分后的场（消除depth维度）

    Example:
        >>> N2 = data['N2']
        >>> integral = compute_vertical_integral(N2, depth_range=[0, -200])
        >>> print(f"Integrated N2: {integral.mean().values:.4e}")
    """
    if find_partitioned_values((data,)):
        from packages.tool_loader.partitioned_execution import execute_partition_aware

        return execute_partition_aware(
            tool_name="compute_vertical_integral",
            tool_func=compute_vertical_integral,
            params={"data": data, "depth_range": depth_range},
        )
    data = materialize_partitioned_xarray(data)

    if 'depth' not in data.dims:
        raise ValueError("Data must have depth dimension")

    # 选择深度范围（归一化正负号）
    from domain.ocean.data_access.load import _normalize_depth_range, _build_coord_slice
    depth_values = np.asarray(data.depth.values, dtype=float)
    nr = _normalize_depth_range(depth_values, depth_range)
    data_subset = data.sel(depth=_build_coord_slice(depth_values, nr))

    if len(data_subset.depth) < 2:
        raise ValueError(f"Not enough depth levels in range {nr}")

    # 梯形积分
    integrated = data_subset.integrate('depth')

    # 取绝对值（因为深度是负值）
    integrated = np.abs(integrated)

    integrated.attrs = {
        'long_name': f"Vertical Integral of {data.attrs.get('long_name', data.name)}",
        'units': f"{data.attrs.get('units', '')} * m",
        'depth_range': f"[{nr[0]}, {nr[1]}] m",
        'n_levels': len(data_subset.depth)
    }

    return integrated
