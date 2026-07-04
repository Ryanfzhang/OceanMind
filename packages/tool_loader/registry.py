"""
Tool registry - 工具输入输出契约注册表

集中维护：
1. tool -> output_type
2. tool -> 参数输入约束
3. tool -> 引用使用建议
"""

from typing import Any, Dict, Optional


TOOL_CONTRACTS: Dict[str, Dict[str, Any]] = {
    "load_dataset": {
        "output_type": "data_container_result",
        "inputs": {
            "dataset": {"kind": "literal", "type": "string"},
            "variable": {"kind": "literal", "type": "string"},
            "lon_range": {"kind": "literal", "type": "array"},
            "lat_range": {"kind": "literal", "type": "array"},
            "time_range": {"kind": "literal_or_ref", "type": "array"},
            "season_filter": {"kind": "literal", "type": "string"},
            "depth_range": {"kind": "literal_or_ref", "type": "array"},
            "vertical_mode": {"kind": "literal", "type": "string"},
            "depth_value": {"kind": "literal", "type": "number"},
        },
        "reference_examples": [],
    },
    "get_dataset_info": {
        "output_type": "metadata_result",
        "inputs": {
            "dataset": {"kind": "literal", "type": "string"},
            "include_runtime_probe": {"kind": "literal", "type": "boolean", "default": False},
        },
        "reference_examples": [],
    },
    "list_available_datasets": {
        "output_type": "metadata_result",
        "inputs": {},
        "reference_examples": [],
    },
    "assemble_dataset": {
        "output_type": "data_container_result",
        "inputs": {
            "variables": {"kind": "literal_or_ref", "type": "object"},
        },
        "reference_examples": [],
    },
    "extract_4d_subset": {
        "output_type": "data_container_result",
        "inputs": {
            "data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "lon_range": {"kind": "literal_or_ref", "type": "array"},
            "lat_range": {"kind": "literal_or_ref", "type": "array"},
            "depth_range": {"kind": "literal_or_ref", "type": "array"},
            "time_range": {"kind": "literal_or_ref", "type": "array"},
        },
        "reference_examples": ["$ref:raw_data.data"],
    },
    "filter_data": {
        "output_type": "data_container_result",
        "inputs": {
            "data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "filter_type": {"kind": "literal", "type": "string"},
            "cutoff_period": {"kind": "literal_or_ref", "type": "array"},
            "dimension": {"kind": "literal", "type": "string"},
            "method": {"kind": "literal", "type": "string"},
            "order": {"kind": "literal", "type": "integer"},
        },
        "reference_examples": ["$ref:raw_data.data"],
    },
    "interpolate_data": {
        "output_type": "data_container_result",
        "inputs": {
            "data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "lon_points": {"kind": "literal_or_ref", "type": "array"},
            "lat_points": {"kind": "literal_or_ref", "type": "array"},
            "depth_points": {"kind": "literal_or_ref", "type": "array"},
            "time_points": {"kind": "literal_or_ref", "type": "array"},
        },
        "reference_examples": ["$ref:raw_data.data"],
    },
    "apply_mask": {
        "output_type": "data_container_result",
        "inputs": {
            "data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "mask": {"kind": "ref_field", "expected": "data_container_result.data"},
            "fill_value": {"kind": "literal", "type": "number"},
        },
        "reference_examples": ["$ref:raw_data.data", "$ref:region_mask.data"],
    },
    "select_vertical": {
        "output_type": "data_container_result",
        "inputs": {
            "data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "mode": {"kind": "literal", "type": "string"},
            "depth_value": {"kind": "literal_or_ref", "type": "number"},
            "depth_range": {"kind": "literal_or_ref", "type": "array"},
            "relative_to": {"kind": "literal", "type": "string"},
            "band_thickness_m": {"kind": "literal_or_ref", "type": "number"},
            "aggregation": {"kind": "literal", "type": "string"},
            "retain_depth": {"kind": "literal", "type": "boolean"},
        },
        "reference_examples": ["$ref:raw_field.data"],
    },
    "build_threshold_mask": {
        "output_type": "data_container_result",
        "inputs": {
            "data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "threshold": {"kind": "literal_or_ref", "type": "number"},
            "comparison": {"kind": "literal", "type": "string"},
            "mask_name": {"kind": "literal", "type": "string"},
        },
        "reference_examples": ["$ref:field.data"],
    },
    "build_condition_mask": {
        "output_type": "data_container_result",
        "inputs": {
            "fields": {"kind": "literal_or_ref", "type": "object"},
            "expression": {"kind": "literal", "type": "string"},
            "mask_name": {"kind": "literal", "type": "string"},
        },
        "reference_examples": ["$ref:oxygen_field.data", "$ref:temperature_field.data"],
    },
    "compute_masked_area_fraction_timeseries": {
        "output_type": "timeseries_result",
        "inputs": {
            "event_mask": {"kind": "ref_field", "expected": "data_container_result.data"},
            "analysis_mask": {"kind": "ref_field", "expected": "data_container_result.data"},
            "min_valid_fraction": {"kind": "literal", "type": "number"},
        },
        "reference_examples": ["$ref:event_mask.data", "$ref:analysis_mask.data"],
    },
    "compute_masked_mean_timeseries": {
        "output_type": "timeseries_result",
        "inputs": {
            "data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "analysis_mask": {"kind": "ref_field", "expected": "data_container_result.data"},
            "event_mask": {"kind": "ref_field", "expected": "data_container_result.data"},
        },
        "reference_examples": ["$ref:field.data", "$ref:analysis_mask.data"],
    },
    "compute_speed_from_uv": {
        "output_type": "data_container_result",
        "inputs": {
            "u": {"kind": "ref_field", "expected": "data_container_result.data"},
            "v": {"kind": "ref_field", "expected": "data_container_result.data"},
        },
        "reference_examples": ["$ref:u_field.data", "$ref:v_field.data"],
    },
    "build_polygon_mask": {
        "output_type": "data_container_result",
        "inputs": {
            "data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "polygon_points": {"kind": "literal_or_ref", "type": "array"},
            "invert": {"kind": "literal", "type": "boolean"},
        },
        "reference_examples": ["$ref:raw_data.data"],
    },
    "build_isobath_mask": {
        "output_type": "data_container_result",
        "inputs": {
            "data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "isobath_depth": {"kind": "literal", "type": "number"},
            "comparison": {"kind": "literal", "type": "string"},
            "bathymetry": {"kind": "ref_field", "expected": "data_container_result.data"},
            "invert": {"kind": "literal", "type": "boolean"},
        },
        "reference_examples": ["$ref:raw_data.data"],
    },
    "combine_masks": {
        "output_type": "data_container_result",
        "inputs": {
            "masks": {"kind": "literal_or_ref", "type": "array"},
            "operation": {"kind": "literal", "type": "string"},
            "invert": {"kind": "literal", "type": "boolean"},
        },
        "reference_examples": ["$ref:polygon_mask.data", "$ref:isobath_mask.data"],
    },
    "compute_density": {
        "output_type": "data_container_result",
        "inputs": {
            "data": {"kind": "ref_field", "expected": "data_container_result.data"},
        },
        "reference_examples": ["$ref:dataset.data"],
    },
    "compute_derived_field": {
        "output_type": "data_container_result",
        "inputs": {
            "data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "dataset": {"kind": "ref_field", "expected": "data_container_result.data"},
            "u": {"kind": "ref_field", "expected": "data_container_result.data"},
            "v": {"kind": "ref_field", "expected": "data_container_result.data"},
            "density": {"kind": "ref_field", "expected": "data_container_result.data"},
            "field": {"kind": "ref_field", "expected": "data_container_result.data"},
            "temp": {"kind": "ref_field", "expected": "data_container_result.data"},
            "salt": {"kind": "ref_field", "expected": "data_container_result.data"},
            "variable": {"kind": "literal", "type": "string"},
            "field_type": {"kind": "literal", "type": "string", "expected": ["vorticity", "speed", "horizontal_gradient", "vertical_gradient", "buoyancy_frequency"]},
        },
        "reference_examples": ["$ref:dataset.data", "$ref:density_field.data", "$ref:u_field.data", "$ref:v_field.data"],
    },
    "compute_spatial_vorticity_map": {
        "output_type": "spatial_field_result",
        "inputs": {
            "u": {"kind": "ref_field", "expected": "data_container_result.data"},
            "v": {"kind": "ref_field", "expected": "data_container_result.data"},
            "time_range": {"kind": "literal_or_ref", "type": "array"},
            "time_aggregation": {"kind": "literal", "type": "string"},
            "depth_range": {"kind": "literal_or_ref", "type": "array"},
            "depth_aggregation": {"kind": "literal", "type": "string"},
            "mask": {"kind": "ref_field", "expected": "data_container_result.data"},
        },
        "reference_examples": ["$ref:u_field.data", "$ref:v_field.data"],
    },
    "compute_vertical_integral": {
        "output_type": "data_container_result",
        "inputs": {
            "data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "depth_range": {"kind": "literal_or_ref", "type": "array"},
        },
        "reference_examples": ["$ref:field.data"],
    },
    "compute_richardson_number": {
        "output_type": "data_container_result",
        "inputs": {
            "n2_data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "u_data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "v_data": {"kind": "ref_field", "expected": "data_container_result.data"},
        },
        "reference_examples": ["$ref:n2_field.data", "$ref:u_field.data", "$ref:v_field.data"],
    },
    "compute_kinetic_energy": {
        "output_type": "data_container_result",
        "inputs": {
            "u_data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "v_data": {"kind": "ref_field", "expected": "data_container_result.data"},
        },
        "reference_examples": ["$ref:u_result.data", "$ref:v_result.data"],
    },
    "compute_eddy_kinetic_energy": {
        "output_type": "data_container_result",
        "inputs": {
            "u_data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "v_data": {"kind": "ref_field", "expected": "data_container_result.data"},
        },
        "reference_examples": ["$ref:u_result.data", "$ref:v_result.data"],
    },
    "compute_vertical_shear": {
        "output_type": "data_container_result",
        "inputs": {
            "u_data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "v_data": {"kind": "ref_field", "expected": "data_container_result.data"},
        },
        "reference_examples": ["$ref:u_result.data", "$ref:v_result.data"],
    },
    "compute_strain_rate": {
        "output_type": "data_container_result",
        "inputs": {
            "u_data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "v_data": {"kind": "ref_field", "expected": "data_container_result.data"},
        },
        "reference_examples": ["$ref:u_result.data", "$ref:v_result.data"],
    },
    "compute_rossby_number": {
        "output_type": "data_container_result",
        "inputs": {
            "u_data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "v_data": {"kind": "ref_field", "expected": "data_container_result.data"},
        },
        "reference_examples": ["$ref:u_result.data", "$ref:v_result.data"],
    },
    "compute_divergence": {
        "output_type": "data_container_result",
        "inputs": {
            "u_data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "v_data": {"kind": "ref_field", "expected": "data_container_result.data"},
        },
        "reference_examples": ["$ref:u_result.data", "$ref:v_result.data"],
    },
    "extract_regional_mean": {
        "output_type": "timeseries_result",
        "inputs": {
            "data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "lon_range": {"kind": "literal_or_ref", "type": "array"},
            "lat_range": {"kind": "literal_or_ref", "type": "array"},
            "depth_range": {"kind": "literal_or_ref", "type": "array"},
            "depth_aggregation": {"kind": "literal", "type": "string"},
        },
        "reference_examples": ["$ref:dataset.data"],
    },
    "compute_area_weighted_mean": {
        "output_type": "timeseries_result",
        "inputs": {
            "data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "lon_range": {"kind": "literal_or_ref", "type": "array"},
            "lat_range": {"kind": "literal_or_ref", "type": "array"},
            "depth_range": {"kind": "literal_or_ref", "type": "array"},
            "depth_aggregation": {"kind": "literal", "type": "string"},
        },
        "reference_examples": ["$ref:dataset.data"],
    },
    "compute_volume_weighted_mean": {
        "output_type": "timeseries_result",
        "inputs": {
            "data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "lon_range": {"kind": "literal_or_ref", "type": "array"},
            "lat_range": {"kind": "literal_or_ref", "type": "array"},
            "depth_range": {"kind": "literal_or_ref", "type": "array"},
        },
        "reference_examples": ["$ref:dataset.data"],
    },
    "compute_area_integral": {
        "output_type": "timeseries_result",
        "inputs": {
            "data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "lon_range": {"kind": "literal_or_ref", "type": "array"},
            "lat_range": {"kind": "literal_or_ref", "type": "array"},
            "depth_range": {"kind": "literal_or_ref", "type": "array"},
            "depth_aggregation": {"kind": "literal", "type": "string"},
        },
        "reference_examples": ["$ref:dataset.data"],
    },
    "compute_volume_integral": {
        "output_type": "timeseries_result",
        "inputs": {
            "data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "lon_range": {"kind": "literal_or_ref", "type": "array"},
            "lat_range": {"kind": "literal_or_ref", "type": "array"},
            "depth_range": {"kind": "literal_or_ref", "type": "array"},
        },
        "reference_examples": ["$ref:dataset.data"],
    },
    "extract_point_timeseries": {
        "output_type": "timeseries_result",
        "inputs": {
            "data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "lon": {"kind": "literal_or_ref", "type": "number"},
            "lat": {"kind": "literal_or_ref", "type": "number"},
            "method": {"kind": "literal", "type": "string"},
            "depth_range": {"kind": "literal_or_ref", "type": "array"},
            "depth_aggregation": {"kind": "literal", "type": "string"},
        },
        "reference_examples": ["$ref:dataset.data"],
    },
    "compute_mixed_layer_mean": {
        "output_type": "data_container_result",
        "inputs": {
            "data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "mixed_layer_depth": {"kind": "ref_field", "expected": "data_container_result.data"},
        },
        "reference_examples": ["$ref:raw_data.data", "$ref:mld_field.data"],
    },
    "compute_layer_mean": {
        "output_type": "data_container_result",
        "inputs": {
            "data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "upper_bound_value": {"kind": "literal_or_ref", "type": "number"},
            "lower_bound_value": {"kind": "literal_or_ref", "type": "number"},
            "upper_bound_field": {"kind": "ref_field", "expected": "data_container_result.data"},
            "lower_bound_field": {"kind": "ref_field", "expected": "data_container_result.data"},
        },
        "reference_examples": [
            "$ref:raw_data.data",
            "$ref:mld_field.data",
            "$ref:thermocline_depth_field.data",
        ],
    },
    "extract_timeseries": {
        "output_type": "timeseries_result",
        "inputs": {
            "data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "lon": {"kind": "literal", "type": "number"},
            "lat": {"kind": "literal", "type": "number"},
            "lon_range": {"kind": "literal_or_ref", "type": "array"},
            "lat_range": {"kind": "literal_or_ref", "type": "array"},
            "spatial_aggregation": {"kind": "literal", "type": "string"},
            "depth_range": {"kind": "literal_or_ref", "type": "array"},
            "depth_aggregation": {"kind": "literal", "type": "string"},
            "method": {"kind": "literal", "type": "string"},
            "mask": {"kind": "ref_field", "expected": "data_container_result.data"},
        },
        "reference_examples": ["$ref:dataset.data"],
    },
    "compute_climatology": {
        "output_type": "climatology_result",
        "inputs": {
            "timeseries": {"kind": "ref_result", "expected": "timeseries_result"},
            "period": {"kind": "literal", "type": "string"},
        },
        "reference_examples": ["$ref:timeseries_result"],
    },
    "compute_anomaly": {
        "output_type": "timeseries_result",
        "inputs": {
            "timeseries": {"kind": "ref_result", "expected": "timeseries_result"},
            "climatology": {"kind": "ref_result", "expected": "climatology_result"},
        },
        "reference_examples": ["$ref:timeseries_result", "$ref:climatology_result"],
    },
    "compute_trend": {
        "output_type": "trend_result",
        "inputs": {
            "timeseries": {"kind": "ref_result", "expected": "timeseries_result"},
            "method": {"kind": "literal", "type": "string"},
            "confidence_level": {"kind": "literal", "type": "number"},
        },
        "reference_examples": ["$ref:anomaly_result"],
    },
    "compute_field_climatology": {
        "output_type": "data_container_result",
        "inputs": {
            "data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "period": {"kind": "literal", "type": "string"},
        },
        "reference_examples": ["$ref:raw_field.data"],
    },
    "compute_field_anomaly": {
        "output_type": "data_container_result",
        "inputs": {
            "data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "climatology": {"kind": "ref_field", "expected": "data_container_result.data"},
            "period": {"kind": "literal", "type": "string"},
        },
        "reference_examples": ["$ref:raw_field.data", "$ref:field_climatology.data"],
    },
    "compute_field_trend": {
        "output_type": "field_trend_result",
        "inputs": {
            "data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "method": {"kind": "literal", "type": "string"},
            "confidence_level": {"kind": "literal", "type": "number"},
        },
        "reference_examples": ["$ref:raw_field.data"],
    },
    "compute_local_tendency": {
        "output_type": "data_container_result",
        "inputs": {
            "data": {"kind": "ref_field", "expected": "data_container_result.data"},
        },
        "reference_examples": ["$ref:raw_field.data"],
    },
    "compute_horizontal_advection": {
        "output_type": "data_container_result",
        "inputs": {
            "data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "u_data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "v_data": {"kind": "ref_field", "expected": "data_container_result.data"},
        },
        "reference_examples": ["$ref:raw_field.data", "$ref:u_field.data", "$ref:v_field.data"],
    },
    "compute_vertical_advection": {
        "output_type": "data_container_result",
        "inputs": {
            "data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "w_data": {"kind": "ref_field", "expected": "data_container_result.data"},
        },
        "reference_examples": ["$ref:raw_field.data", "$ref:w_field.data"],
    },
    "resample_timeseries": {
        "output_type": "timeseries_result",
        "inputs": {
            "timeseries": {"kind": "ref_result", "expected": "timeseries_result"},
            "freq": {"kind": "literal", "type": "str",
                     "enum": ["MS", "W", "D", "YS"],
                     "default": "MS",
                     "description": "Target resampling frequency: MS=monthly, W=weekly, D=daily, YS=yearly"},
            "method": {"kind": "literal", "type": "str",
                       "enum": ["mean", "sum", "min", "max"],
                       "default": "mean",
                       "description": "Aggregation method applied when resampling"},
        },
        "reference_examples": ["$ref:timeseries_result"],
    },
    "remove_seasonal_cycle": {
        "output_type": "timeseries_result",
        "inputs": {
            "timeseries": {"kind": "ref_result", "expected": "timeseries_result"},
        },
        "reference_examples": ["$ref:timeseries_result"],
    },
    "compute_lag_correlation": {
        "output_type": "lag_correlation_result",
        "inputs": {
            "timeseries1": {"kind": "ref_result", "expected": "timeseries_result"},
            "timeseries2": {"kind": "ref_result", "expected": "timeseries_result"},
            "max_lag": {"kind": "literal", "type": "integer"},
            "confidence_level": {"kind": "literal", "type": "number"},
        },
        "reference_examples": ["$ref:ts1", "$ref:ts2"],
    },
    "extract_vertical_profile": {
        "output_type": "profile_result",
        "inputs": {
            "data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "lon": {"kind": "literal", "type": "number"},
            "lat": {"kind": "literal", "type": "number"},
            "method": {"kind": "literal", "type": "string"},
        },
        "reference_examples": ["$ref:dataset.data"],
    },
    "identify_mixed_layer_depth": {
        "output_type": "data_container_result",
        "inputs": {
            "density": {"kind": "ref_field", "expected": "data_container_result.data"},
        },
        "reference_examples": ["$ref:density.data"],
    },
    "identify_thermocline_depth": {
        "output_type": "data_container_result",
        "inputs": {
            "temp": {"kind": "ref_field", "expected": "data_container_result.data"},
        },
        "reference_examples": ["$ref:temp.data"],
    },
    "identify_pycnocline_depth": {
        "output_type": "data_container_result",
        "inputs": {
            "density": {"kind": "ref_field", "expected": "data_container_result.data"},
        },
        "reference_examples": ["$ref:density.data"],
    },
    "analyze_vertical_structure": {
        "output_type": "data_container_result",
        "inputs": {
            "data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "structure_type": {
                "kind": "literal",
                "type": "string",
                "enum": ["mld", "thermocline", "pycnocline"],
            },
            "variable": {"kind": "literal", "type": "string"},
            "threshold": {"kind": "literal", "type": "number"},
        },
        "reference_examples": ["$ref:dataset.data"],
    },
    "perform_eof_analysis": {
        "output_type": "eof_result",
        "inputs": {
            "data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "n_modes": {"kind": "literal", "type": "integer"},
            "preprocessing": {"kind": "literal", "type": "string"},
            "weight_by_latitude": {"kind": "literal", "type": "boolean"},
        },
        "reference_examples": ["$ref:dataset.data"],
    },
    "reconstruct_from_eof": {
        "output_type": "data_container_result",
        "inputs": {
            "eof_result": {"kind": "ref_result", "expected": "eof_result"},
            "mode_indices": {"kind": "literal_or_ref", "type": "array"},
        },
        "reference_examples": ["$ref:eof_result"],
    },
    "compute_spatial_field": {
        "output_type": "spatial_field_result",
        "inputs": {
            "data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "time_range": {"kind": "literal_or_ref", "type": "array"},
            "time_aggregation": {"kind": "literal", "type": "string"},
            "depth_range": {"kind": "literal_or_ref", "type": "array"},
            "depth_aggregation": {"kind": "literal", "type": "string"},
            "mask": {"kind": "ref_field", "expected": "data_container_result.data"},
        },
        "reference_examples": ["$ref:diagnostic.data"],
    },
    "compute_hovmoller": {
        "output_type": "hovmoller_result",
        "inputs": {
            "data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "diagram_type": {"kind": "literal", "type": "string"},
            "fixed_lat": {"kind": "literal", "type": "number"},
            "fixed_lon": {"kind": "literal", "type": "number"},
            "fixed_lat_range": {"kind": "literal_or_ref", "type": "array"},
            "fixed_lon_range": {"kind": "literal_or_ref", "type": "array"},
            "aggregate_dim": {"kind": "literal", "type": "string"},
            "spatial_weighting": {"kind": "literal", "type": "string"},
            "depth": {"kind": "literal", "type": "number"},
            "depth_range": {"kind": "literal_or_ref", "type": "array"},
        },
        "reference_examples": ["$ref:dataset.data"],
    },
    "extract_transect_section": {
        "output_type": "section_result",
        "inputs": {
            "data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "transect_points": {"kind": "literal_or_ref", "type": "array"},
            "n_samples": {"kind": "literal", "type": "integer"},
            "method": {"kind": "literal", "type": "string"},
        },
        "reference_examples": ["$ref:raw_field.data"],
    },
    "compute_section_hovmoller": {
        "output_type": "hovmoller_result",
        "inputs": {
            "section": {"kind": "ref_result", "expected": "section_result"},
            "diagram_type": {"kind": "literal", "type": "string"},
            "fixed_depth": {"kind": "literal", "type": "number"},
            "depth_range": {"kind": "literal_or_ref", "type": "array"},
            "fixed_distance_km": {"kind": "literal", "type": "number"},
            "distance_range_km": {"kind": "literal_or_ref", "type": "array"},
            "aggregate_method": {"kind": "literal", "type": "string"},
        },
        "reference_examples": ["$ref:section_result"],
    },
    "compute_volume_transport": {
        "output_type": "timeseries_result",
        "inputs": {
            "u": {"kind": "ref_field", "expected": "data_container_result.data"},
            "v": {"kind": "ref_field", "expected": "data_container_result.data"},
            "transect_points": {"kind": "literal_or_ref", "type": "array"},
            "depth_range": {"kind": "literal_or_ref", "type": "array"},
            "n_samples": {"kind": "literal", "type": "integer"},
            "method": {"kind": "literal", "type": "string"},
        },
        "reference_examples": ["$ref:u_field.data", "$ref:v_field.data"],
    },
    "compute_heat_transport": {
        "output_type": "timeseries_result",
        "inputs": {
            "u": {"kind": "ref_field", "expected": "data_container_result.data"},
            "v": {"kind": "ref_field", "expected": "data_container_result.data"},
            "temp": {"kind": "ref_field", "expected": "data_container_result.data"},
            "transect_points": {"kind": "literal_or_ref", "type": "array"},
            "depth_range": {"kind": "literal_or_ref", "type": "array"},
            "rho0": {"kind": "literal", "type": "number"},
            "cp": {"kind": "literal", "type": "number"},
            "n_samples": {"kind": "literal", "type": "integer"},
            "method": {"kind": "literal", "type": "string"},
        },
        "reference_examples": ["$ref:u_field.data", "$ref:v_field.data", "$ref:temp_field.data"],
    },
    "compute_salt_transport": {
        "output_type": "timeseries_result",
        "inputs": {
            "u": {"kind": "ref_field", "expected": "data_container_result.data"},
            "v": {"kind": "ref_field", "expected": "data_container_result.data"},
            "salt": {"kind": "ref_field", "expected": "data_container_result.data"},
            "transect_points": {"kind": "literal_or_ref", "type": "array"},
            "depth_range": {"kind": "literal_or_ref", "type": "array"},
            "n_samples": {"kind": "literal", "type": "integer"},
            "method": {"kind": "literal", "type": "string"},
        },
        "reference_examples": ["$ref:u_field.data", "$ref:v_field.data", "$ref:salt_field.data"],
    },
    "compute_freshwater_transport": {
        "output_type": "timeseries_result",
        "inputs": {
            "u": {"kind": "ref_field", "expected": "data_container_result.data"},
            "v": {"kind": "ref_field", "expected": "data_container_result.data"},
            "salt": {"kind": "ref_field", "expected": "data_container_result.data"},
            "transect_points": {"kind": "literal_or_ref", "type": "array"},
            "depth_range": {"kind": "literal_or_ref", "type": "array"},
            "s_ref": {"kind": "literal", "type": "number"},
            "n_samples": {"kind": "literal", "type": "integer"},
            "method": {"kind": "literal", "type": "string"},
        },
        "reference_examples": ["$ref:u_field.data", "$ref:v_field.data", "$ref:salt_field.data"],
    },
    "compute_transport_by_layer": {
        "output_type": "layer_transport_result",
        "inputs": {
            "u": {"kind": "ref_field", "expected": "data_container_result.data"},
            "v": {"kind": "ref_field", "expected": "data_container_result.data"},
            "transect_points": {"kind": "literal_or_ref", "type": "array"},
            "layer_bounds": {"kind": "literal_or_ref", "type": "array"},
            "transport_type": {"kind": "literal", "type": "string"},
            "temp": {"kind": "ref_field", "expected": "data_container_result.data"},
            "salt": {"kind": "ref_field", "expected": "data_container_result.data"},
            "rho0": {"kind": "literal", "type": "number"},
            "cp": {"kind": "literal", "type": "number"},
            "s_ref": {"kind": "literal", "type": "number"},
            "n_samples": {"kind": "literal", "type": "integer"},
            "method": {"kind": "literal", "type": "string"},
        },
        "reference_examples": ["$ref:u_field.data", "$ref:v_field.data"],
    },
    "compute_transport_streamfunction_map": {
        "output_type": "spatial_field_result",
        "inputs": {
            "u": {"kind": "ref_field", "expected": "data_container_result.data"},
            "v": {"kind": "ref_field", "expected": "data_container_result.data"},
            "depth_range": {"kind": "literal_or_ref", "type": "array"},
            "time_aggregation": {"kind": "literal", "type": "string"},
            "regional_gauge": {"kind": "literal", "type": "string"},
        },
        "reference_examples": ["$ref:u_field.data", "$ref:v_field.data"],
    },
    "compute_transect_normal_flux_hovmoller": {
        "output_type": "hovmoller_result",
        "inputs": {
            "u": {"kind": "ref_field", "expected": "data_container_result.data"},
            "v": {"kind": "ref_field", "expected": "data_container_result.data"},
            "transect_points": {"kind": "literal_or_ref", "type": "array"},
            "depth_range": {"kind": "literal_or_ref", "type": "array"},
            "n_samples": {"kind": "literal", "type": "integer"},
            "method": {"kind": "literal", "type": "string"},
        },
        "reference_examples": ["$ref:u_field.data", "$ref:v_field.data"],
    },
    "compute_histogram": {
        "output_type": "histogram_result",
        "inputs": {
            "data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "n_bins": {"kind": "literal", "type": "integer"},
            "bin_range": {"kind": "literal_or_ref", "type": "array"},
            "normalize": {"kind": "literal", "type": "boolean"},
            "mask": {"kind": "ref_field", "expected": "data_container_result.data"},
        },
        "reference_examples": ["$ref:field.data"],
    },
    "compute_2d_histogram": {
        "output_type": "histogram_2d_result",
        "inputs": {
            "data_x": {"kind": "ref_field", "expected": "data_container_result.data"},
            "data_y": {"kind": "ref_field", "expected": "data_container_result.data"},
            "n_bins": {"kind": "literal", "type": "integer"},
            "range_x": {"kind": "literal_or_ref", "type": "array"},
            "range_y": {"kind": "literal_or_ref", "type": "array"},
            "normalize": {"kind": "literal", "type": "boolean"},
            "mask": {"kind": "ref_field", "expected": "data_container_result.data"},
        },
        "reference_examples": ["$ref:x_field.data", "$ref:y_field.data"],
    },
    "compute_ts_diagram": {
        "output_type": "ts_diagram_result",
        "inputs": {
            "temp": {"kind": "ref_field", "expected": "data_container_result.data"},
            "salt": {"kind": "ref_field", "expected": "data_container_result.data"},
            "color_field": {"kind": "ref_field", "expected": "data_container_result.data"},
            "max_points": {"kind": "literal", "type": "integer"},
            "sampling": {"kind": "literal", "type": "string"},
        },
        "reference_examples": ["$ref:temp_field.data", "$ref:salt_field.data"],
    },
    "compute_watermass_event_association": {
        "output_type": "watermass_event_association_result",
        "inputs": {
            "event_field": {"kind": "ref_field", "expected": "data_container_result.data"},
            "temp": {"kind": "ref_field", "expected": "data_container_result.data"},
            "salt": {"kind": "ref_field", "expected": "data_container_result.data"},
            "density": {"kind": "ref_field", "expected": "data_container_result.data"},
            "event_detection": {"kind": "ref_result", "expected": "event_detection_result"},
            "lon_range": {"kind": "literal_or_ref", "type": "array"},
            "lat_range": {"kind": "literal_or_ref", "type": "array"},
            "subregion_grid": {"kind": "literal", "type": "array"},
            "hotspot_quantile": {"kind": "literal", "type": "number"},
            "max_ts_points": {"kind": "literal", "type": "integer"},
            "sampling": {"kind": "literal", "type": "string"},
            "watermass_config_path": {"kind": "literal", "type": "string"},
        },
        "reference_examples": [
            "$ref:event_field.data",
            "$ref:temp_field.data",
            "$ref:salt_field.data",
            "$ref:density_field.data",
            "$ref:event_detection",
        ],
    },
    "build_watermass_tile_map": {
        "output_type": "spatial_field_result",
        "inputs": {
            "association_result": {"kind": "ref_result", "expected": "watermass_event_association_result"},
            "map_kind": {"kind": "literal", "type": "string"},
        },
        "reference_examples": ["$ref:watermass_association"],
    },
    "build_watermass_ts_diagram": {
        "output_type": "ts_diagram_result",
        "inputs": {
            "association_result": {"kind": "ref_result", "expected": "watermass_event_association_result"},
        },
        "reference_examples": ["$ref:watermass_association"],
    },
    "extract_isopycnal_surface": {
        "output_type": "data_container_result",
        "inputs": {
            "density": {"kind": "ref_field", "expected": "data_container_result.data"},
            "target_sigma0": {"kind": "literal", "type": "number"},
        },
        "reference_examples": ["$ref:density_field.data"],
    },
    "compute_isopycnal_layer_mean": {
        "output_type": "data_container_result",
        "inputs": {
            "data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "density": {"kind": "ref_field", "expected": "data_container_result.data"},
            "sigma0_upper": {"kind": "literal", "type": "number"},
            "sigma0_lower": {"kind": "literal", "type": "number"},
        },
        "reference_examples": ["$ref:raw_field.data", "$ref:density_field.data"],
    },
    "compute_regression_map": {
        "output_type": "regression_map_result",
        "inputs": {
            "field": {"kind": "ref_field", "expected": "data_container_result.data"},
            "index_timeseries": {"kind": "ref_result", "expected": "timeseries_result"},
            "lag": {"kind": "literal", "type": "integer"},
            "remove_seasonal_cycle": {"kind": "literal", "type": "boolean"},
            "significance_level": {"kind": "literal", "type": "number"},
        },
        "reference_examples": ["$ref:field_result.data", "$ref:index_timeseries"],
    },
    "compute_composite_field": {
        "output_type": "composite_result",
        "inputs": {
            "field": {"kind": "ref_field", "expected": "data_container_result.data"},
            "index_timeseries": {"kind": "ref_result", "expected": "timeseries_result"},
            "quantile": {
                "kind": "literal",
                "type": "number",
                "minimum": 0.0,
                "maximum": 0.5,
                "exclusive_minimum": True,
                "exclusive_maximum": True,
                "default": 0.2,
                "normalize_percentile_to_fraction": True,
                "mirror_above_half": True,
            },
            "lag": {"kind": "literal", "type": "integer"},
            "anomaly": {"kind": "literal", "type": "boolean"},
        },
        "reference_examples": ["$ref:field_result.data", "$ref:index_timeseries"],
    },
    "compute_spectrum": {
        "output_type": "spectrum_result",
        "inputs": {
            "timeseries": {"kind": "ref_result", "expected": "timeseries_result"},
            "method": {"kind": "literal", "type": "string"},
            "detrend": {"kind": "literal", "type": "string"},
            "window": {"kind": "literal", "type": "string"},
        },
        "reference_examples": ["$ref:index_timeseries"],
    },
    "detect_heatwaves": {
        "output_type": "event_detection_result",
        "inputs": {
            "temp": {"kind": "ref_field", "expected": "data_container_result.data"},
            "percentile_threshold": {"kind": "literal", "type": "number"},
            "min_duration_days": {"kind": "literal", "type": "integer"},
            "min_area_km2": {"kind": "literal", "type": "number"},
            "analysis_mask": {"kind": "ref_field", "expected": "data_container_result.data"},
            "vertical_mode": {"kind": "literal", "type": "string"},
            "depth_value": {"kind": "literal", "type": "number"},
            "depth_range": {"kind": "literal_or_ref", "type": "array"},
            "depth_aggregation": {"kind": "literal", "type": "string"},
        },
        "reference_examples": ["$ref:temperature_field.data"],
    },
    "detect_hypoxia": {
        "output_type": "event_detection_result",
        "inputs": {
            "oxygen": {"kind": "ref_field", "expected": "data_container_result.data"},
            "oxygen_threshold": {"kind": "literal", "type": "number"},
            "severe_threshold": {"kind": "literal", "type": "number"},
            "min_area_km2": {"kind": "literal", "type": "number"},
            "min_duration_days": {"kind": "literal", "type": "integer"},
            "analysis_mask": {"kind": "ref_field", "expected": "data_container_result.data"},
            "vertical_mode": {"kind": "literal", "type": "string"},
            "depth_value": {"kind": "literal", "type": "number"},
            "depth_range": {"kind": "literal_or_ref", "type": "array"},
            "depth_aggregation": {"kind": "literal", "type": "string"},
        },
        "reference_examples": ["$ref:oxygen_field.data"],
    },
    "detect_algal_blooms": {
        "output_type": "event_detection_result",
        "inputs": {
            "chlorophyll": {"kind": "ref_field", "expected": "data_container_result.data"},
            "threshold": {"kind": "literal", "type": "number"},
            "percentile_threshold": {"kind": "literal", "type": "number"},
            "min_duration_days": {"kind": "literal", "type": "integer"},
            "min_area_km2": {"kind": "literal", "type": "number"},
            "bloom_type": {"kind": "literal", "type": "string"},
            "analysis_mask": {"kind": "ref_field", "expected": "data_container_result.data"},
            "vertical_mode": {"kind": "literal", "type": "string"},
            "depth_value": {"kind": "literal", "type": "number"},
            "depth_range": {"kind": "literal_or_ref", "type": "array"},
            "depth_aggregation": {"kind": "literal", "type": "string"},
        },
        "reference_examples": ["$ref:chlorophyll_field.data"],
    },
    "detect_eddies": {
        "output_type": "event_detection_result",
        "inputs": {
            "u": {"kind": "ref_field", "expected": "data_container_result.data"},
            "v": {"kind": "ref_field", "expected": "data_container_result.data"},
            "ow_threshold": {"kind": "literal", "type": "number"},
            "min_radius_km": {"kind": "literal", "type": "number"},
            "max_radius_km": {"kind": "literal", "type": "number"},
            "min_pixels": {"kind": "literal", "type": "integer"},
        },
        "reference_examples": ["$ref:u_field.data", "$ref:v_field.data"],
    },
    "detect_fronts": {
        "output_type": "event_detection_result",
        "inputs": {
            "data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "variable": {"kind": "literal", "type": "string"},
            "gradient_threshold": {"kind": "literal", "type": "number"},
            "min_length_km": {"kind": "literal", "type": "number"},
            "min_pixels": {"kind": "literal", "type": "integer"},
            "smoothing_sigma": {"kind": "literal", "type": "number"},
        },
        "reference_examples": ["$ref:temp_field.data"],
    },
    "detect_upwelling": {
        "output_type": "event_detection_result",
        "inputs": {
            "temp": {"kind": "ref_field", "expected": "data_container_result.data"},
            "percentile_threshold": {"kind": "literal", "type": "number"},
            "min_duration_days": {"kind": "literal", "type": "integer"},
            "min_area_km2": {"kind": "literal", "type": "number"},
            "analysis_mask": {"kind": "ref_field", "expected": "data_container_result.data"},
            "vertical_mode": {"kind": "literal", "type": "string"},
            "depth_value": {"kind": "literal", "type": "number"},
            "depth_range": {"kind": "literal_or_ref", "type": "array"},
            "depth_aggregation": {"kind": "literal", "type": "string"},
        },
        "reference_examples": ["$ref:temperature_field.data"],
    },
    "detect_jets": {
        "output_type": "event_detection_result",
        "inputs": {
            "u": {"kind": "ref_field", "expected": "data_container_result.data"},
            "v": {"kind": "ref_field", "expected": "data_container_result.data"},
            "speed_threshold": {"kind": "literal", "type": "number"},
            "percentile_threshold": {"kind": "literal", "type": "number"},
            "min_length_km": {"kind": "literal", "type": "number"},
            "min_aspect_ratio": {"kind": "literal", "type": "number"},
            "min_pixels": {"kind": "literal", "type": "integer"},
        },
        "reference_examples": ["$ref:u_field.data", "$ref:v_field.data"],
    },
    "detect_meanders": {
        "output_type": "event_detection_result",
        "inputs": {
            "u": {"kind": "ref_field", "expected": "data_container_result.data"},
            "v": {"kind": "ref_field", "expected": "data_container_result.data"},
            "curvature_threshold": {"kind": "literal", "type": "number"},
            "percentile_threshold": {"kind": "literal", "type": "number"},
            "min_length_km": {"kind": "literal", "type": "number"},
            "min_pixels": {"kind": "literal", "type": "integer"},
        },
        "reference_examples": ["$ref:u_field.data", "$ref:v_field.data"],
    },
    "detect_eutrophication": {
        "output_type": "event_detection_result",
        "inputs": {
            "chlorophyll": {"kind": "ref_field", "expected": "data_container_result.data"},
            "oxygen": {"kind": "ref_field", "expected": "data_container_result.data"},
            "chlorophyll_percentile": {"kind": "literal", "type": "number"},
            "oxygen_threshold": {"kind": "literal", "type": "number"},
            "min_duration_days": {"kind": "literal", "type": "integer"},
            "min_area_km2": {"kind": "literal", "type": "number"},
            "analysis_mask": {"kind": "ref_field", "expected": "data_container_result.data"},
            "vertical_mode": {"kind": "literal", "type": "string"},
            "depth_value": {"kind": "literal", "type": "number"},
            "depth_range": {"kind": "literal_or_ref", "type": "array"},
            "depth_aggregation": {"kind": "literal", "type": "string"},
        },
        "reference_examples": ["$ref:chlorophyll_field.data", "$ref:oxygen_field.data"],
    },
    "track_eddies": {
        "output_type": "event_detection_result",
        "inputs": {
            "u": {"kind": "ref_field", "expected": "data_container_result.data"},
            "v": {"kind": "ref_field", "expected": "data_container_result.data"},
        },
        "reference_examples": ["$ref:u_field.data", "$ref:v_field.data"],
    },
    "compute_event_statistics": {
        "output_type": "event_statistics_result",
        "inputs": {
            "events": {"kind": "ref_field", "expected": "event_detection_result.events"},
            "group_by": {"kind": "literal", "type": "string"},
        },
        "reference_examples": ["$ref:detection.events"],
    },
    "compute_event_spatial_distribution": {
        "output_type": "event_spatial_distribution_result",
        "inputs": {
            "events": {"kind": "ref_field", "expected": "event_detection_result.events"},
        },
        "reference_examples": ["$ref:detection.events"],
    },
    "compare_event_periods": {
        "output_type": "event_comparison_result",
        "inputs": {
            "events1": {"kind": "ref_field", "expected": "event_detection_result.events"},
            "events2": {"kind": "ref_field", "expected": "event_detection_result.events"},
            "period1_label": {"kind": "literal", "type": "str", "default": "Period 1"},
            "period2_label": {"kind": "literal", "type": "str", "default": "Period 2"},
        },
        "reference_examples": ["$ref:events_p1.events", "$ref:events_p2.events"],
    },
    "compute_event_frequency_map": {
        "output_type": "spatial_field_result",
        "inputs": {
            "event_detection": {"kind": "ref_result", "expected": "event_detection_result"},
            "events": {"kind": "ref_field", "expected": "event_detection_result.events"},
            "data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "lon_range": {"kind": "literal_or_ref", "type": "array"},
            "lat_range": {"kind": "literal_or_ref", "type": "array"},
            "resolution_deg": {"kind": "literal", "type": "number"},
            "normalize": {"kind": "literal", "type": "boolean"},
        },
        "reference_examples": ["$ref:detection.events"],
    },
    "compute_event_summary_map": {
        "output_type": "spatial_field_result",
        "inputs": {
            "event_detection": {"kind": "ref_result", "expected": "event_detection_result"},
            "data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "summary_mode": {"kind": "literal", "type": "string"},
        },
        "reference_examples": ["$ref:detection", "$ref:event_field.data"],
    },
    "compute_event_timeseries_count": {
        "output_type": "timeseries_result",
        "inputs": {
            "events": {"kind": "ref_field", "expected": "event_detection_result.events"},
            "weight_by": {"kind": "literal", "type": "string"},
        },
        "reference_examples": ["$ref:detection.events"],
    },
    "compute_stratification_index": {
        "output_type": "data_container_result",
        "inputs": {
            "data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "temp": {"kind": "ref_field", "expected": "data_container_result.data"},
            "salt": {"kind": "ref_field", "expected": "data_container_result.data"},
            "density": {"kind": "ref_field", "expected": "data_container_result.data"},
            "depth_range": {"kind": "literal_or_ref", "type": "array"},
            "surface_depth": {"kind": "literal", "type": "number"},
            "bottom_depth": {"kind": "literal", "type": "number"},
            "method": {"kind": "literal", "type": "string"},
        },
        "reference_examples": ["$ref:temp_field.data", "$ref:salt_field.data", "$ref:density_field.data"],
    },
    "compute_brunt_vaisala_frequency": {
        "output_type": "data_container_result",
        "inputs": {
            "data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "temp": {"kind": "ref_field", "expected": "data_container_result.data"},
            "salt": {"kind": "ref_field", "expected": "data_container_result.data"},
            "density": {"kind": "ref_field", "expected": "data_container_result.data"},
        },
        "reference_examples": ["$ref:temp_field.data", "$ref:salt_field.data", "$ref:density_field.data"],
    },
    "compute_density_gradient_profile": {
        "output_type": "profile_result",
        "inputs": {
            "data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "temp": {"kind": "ref_field", "expected": "data_container_result.data"},
            "salt": {"kind": "ref_field", "expected": "data_container_result.data"},
            "density": {"kind": "ref_field", "expected": "data_container_result.data"},
            "lon": {"kind": "literal", "type": "number"},
            "lat": {"kind": "literal", "type": "number"},
            "method": {"kind": "literal", "type": "string"},
        },
        "reference_examples": ["$ref:temp_field.data", "$ref:salt_field.data", "$ref:density_field.data"],
    },
    "compute_mld_thermocline_offset": {
        "output_type": "data_container_result",
        "inputs": {
            "mixed_layer_depth": {"kind": "ref_field", "expected": "data_container_result.data"},
            "thermocline_depth": {"kind": "ref_field", "expected": "data_container_result.data"},
        },
        "reference_examples": ["$ref:mld_field.data", "$ref:thermocline_depth.data"],
    },
    "compute_vertical_stability_timeseries": {
        "output_type": "timeseries_result",
        "inputs": {
            "data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "temp": {"kind": "ref_field", "expected": "data_container_result.data"},
            "salt": {"kind": "ref_field", "expected": "data_container_result.data"},
            "density": {"kind": "ref_field", "expected": "data_container_result.data"},
            "lon_range": {"kind": "literal_or_ref", "type": "array"},
            "lat_range": {"kind": "literal_or_ref", "type": "array"},
            "depth_range": {"kind": "literal_or_ref", "type": "array"},
            "weighting": {"kind": "literal", "type": "string"},
        },
        "reference_examples": ["$ref:temp_field.data", "$ref:salt_field.data", "$ref:density_field.data"],
    },
    "compute_tracer_horizontal_advection_timeseries": {
        "output_type": "timeseries_result",
        "inputs": {
            "data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "u_data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "v_data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "lon_range": {"kind": "literal_or_ref", "type": "array"},
            "lat_range": {"kind": "literal_or_ref", "type": "array"},
            "depth_range": {"kind": "literal_or_ref", "type": "array"},
            "depth_aggregation": {"kind": "literal", "type": "string"},
            "weighting": {"kind": "literal", "type": "string"},
        },
        "reference_examples": ["$ref:tracer_field.data", "$ref:u_field.data", "$ref:v_field.data"],
    },
    "compute_tracer_advection_map": {
        "output_type": "spatial_field_result",
        "inputs": {
            "data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "u_data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "v_data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "time_range": {"kind": "literal_or_ref", "type": "array"},
            "depth_range": {"kind": "literal_or_ref", "type": "array"},
            "depth_aggregation": {"kind": "literal", "type": "string"},
            "time_aggregation": {"kind": "literal", "type": "string"},
        },
        "reference_examples": ["$ref:tracer_field.data", "$ref:u_field.data", "$ref:v_field.data"],
    },
    "compute_partial_tracer_budget": {
        "output_type": "data_container_result",
        "inputs": {
            "data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "u_data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "v_data": {"kind": "ref_field", "expected": "data_container_result.data"},
        },
        "reference_examples": ["$ref:tracer_field.data", "$ref:u_field.data", "$ref:v_field.data"],
    },
    "compute_budget_residual": {
        "output_type": "timeseries_result",
        "inputs": {
            "local_tendency": {"kind": "ref_result", "expected": "timeseries_result"},
            "horizontal_advection": {"kind": "ref_result", "expected": "timeseries_result"},
            "vertical_advection": {"kind": "ref_result", "expected": "timeseries_result"},
        },
        "reference_examples": ["$ref:local_tendency_ts", "$ref:horizontal_advection_ts"],
    },
    "compare_budget_term_magnitudes": {
        "output_type": "mechanism_score_result",
        "inputs": {
            "local_tendency": {"kind": "ref_result", "expected": "timeseries_result"},
            "horizontal_advection": {"kind": "ref_result", "expected": "timeseries_result"},
            "vertical_advection": {"kind": "ref_result", "expected": "timeseries_result"},
            "residual": {"kind": "ref_result", "expected": "timeseries_result"},
        },
        "reference_examples": ["$ref:local_tendency_ts", "$ref:horizontal_advection_ts", "$ref:residual_ts"],
    },
    "compute_front_proximity_index": {
        "output_type": "data_container_result",
        "inputs": {
            "data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "percentile": {"kind": "literal", "type": "number"},
        },
        "reference_examples": ["$ref:tracer_field.data"],
    },
    "compute_eddy_influence_mask": {
        "output_type": "data_container_result",
        "inputs": {
            "u_data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "v_data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "percentile": {"kind": "literal", "type": "number"},
        },
        "reference_examples": ["$ref:u_field.data", "$ref:v_field.data"],
    },
    "compute_tracer_gradient_alignment": {
        "output_type": "data_container_result",
        "inputs": {
            "data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "u_data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "v_data": {"kind": "ref_field", "expected": "data_container_result.data"},
        },
        "reference_examples": ["$ref:tracer_field.data", "$ref:u_field.data", "$ref:v_field.data"],
    },
    "compute_mesoscale_background_separation": {
        "output_type": "data_container_result",
        "inputs": {
            "data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "cutoff_period": {"kind": "literal", "type": "number"},
            "component": {"kind": "literal", "type": "string"},
        },
        "reference_examples": ["$ref:tracer_field.data"],
    },
    "compute_flow_structure_context": {
        "output_type": "data_container_result",
        "inputs": {
            "u_data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "v_data": {"kind": "ref_field", "expected": "data_container_result.data"},
        },
        "reference_examples": ["$ref:u_field.data", "$ref:v_field.data"],
    },
    "compute_event_precursor_composite": {
        "output_type": "spatial_field_result",
        "inputs": {
            "field": {"kind": "ref_field", "expected": "data_container_result.data"},
            "events": {"kind": "ref_field", "expected": "event_detection_result.events"},
            "lead_steps": {"kind": "literal", "type": "integer"},
            "depth_range": {"kind": "literal_or_ref", "type": "array"},
            "depth_aggregation": {"kind": "literal", "type": "string"},
        },
        "reference_examples": ["$ref:field.data", "$ref:detection.events"],
    },
    "compute_event_lead_lag_regression": {
        "output_type": "lag_correlation_result",
        "inputs": {
            "field": {"kind": "ref_field", "expected": "data_container_result.data"},
            "events": {"kind": "ref_field", "expected": "event_detection_result.events"},
            "lon_range": {"kind": "literal_or_ref", "type": "array"},
            "lat_range": {"kind": "literal_or_ref", "type": "array"},
            "depth_range": {"kind": "literal_or_ref", "type": "array"},
            "depth_aggregation": {"kind": "literal", "type": "string"},
            "max_lag": {"kind": "literal", "type": "integer"},
        },
        "reference_examples": ["$ref:field.data", "$ref:detection.events"],
    },
    "compute_oxygen_chla_coupling_metrics": {
        "output_type": "mechanism_score_result",
        "inputs": {
            "oxygen_timeseries": {"kind": "ref_result", "expected": "timeseries_result"},
            "chla_timeseries": {"kind": "ref_result", "expected": "timeseries_result"},
        },
        "reference_examples": ["$ref:oxygen_ts", "$ref:chla_ts"],
    },
    "compute_stratification_response_index": {
        "output_type": "mechanism_score_result",
        "inputs": {
            "stratification": {"kind": "ref_result", "expected": "timeseries_result"},
            "response": {"kind": "ref_result", "expected": "timeseries_result"},
        },
        "reference_examples": ["$ref:stratification_ts", "$ref:response_ts"],
    },
    "compute_event_condition_contrast": {
        "output_type": "mechanism_score_result",
        "inputs": {
            "field": {"kind": "ref_field", "expected": "data_container_result.data"},
            "events": {"kind": "ref_field", "expected": "event_detection_result.events"},
            "lon_range": {"kind": "literal_or_ref", "type": "array"},
            "lat_range": {"kind": "literal_or_ref", "type": "array"},
            "depth_range": {"kind": "literal_or_ref", "type": "array"},
            "depth_aggregation": {"kind": "literal", "type": "string"},
            "partition_mode": {"kind": "literal", "type": "string"},
            "subregion_grid": {"kind": "literal_or_ref", "type": "array"},
            "subregion_weighting": {"kind": "literal", "type": "string"},
        },
        "reference_examples": ["$ref:field.data", "$ref:detection.events"],
    },
    "replace_field_with_climatology": {
        "output_type": "data_container_result",
        "inputs": {
            "data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "period": {"kind": "literal", "type": "string"},
        },
        "reference_examples": ["$ref:field.data"],
    },
    "remove_field_anomaly_component": {
        "output_type": "data_container_result",
        "inputs": {
            "data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "period": {"kind": "literal", "type": "string"},
        },
        "reference_examples": ["$ref:field.data"],
    },
    "filter_mesoscale_component": {
        "output_type": "data_container_result",
        "inputs": {
            "data": {"kind": "ref_field", "expected": "data_container_result.data"},
            "cutoff_period": {"kind": "literal", "type": "number"},
            "component": {"kind": "literal", "type": "string"},
        },
        "reference_examples": ["$ref:field.data"],
    },
    "run_proxy_counterfactual_experiment": {
        "output_type": "timeseries_result",
        "inputs": {
            "baseline": {"kind": "ref_field", "expected": "data_container_result.data"},
            "counterfactual": {"kind": "ref_field", "expected": "data_container_result.data"},
            "lon_range": {"kind": "literal_or_ref", "type": "array"},
            "lat_range": {"kind": "literal_or_ref", "type": "array"},
            "depth_range": {"kind": "literal_or_ref", "type": "array"},
            "depth_aggregation": {"kind": "literal", "type": "string"},
        },
        "reference_examples": ["$ref:baseline_field.data", "$ref:counterfactual_field.data"],
    },
    "compare_counterfactual_outcome": {
        "output_type": "evidence_report_result",
        "inputs": {
            "baseline": {"kind": "ref_result", "expected": "timeseries_result"},
            "counterfactual": {"kind": "ref_result", "expected": "timeseries_result"},
            "mechanism_name": {"kind": "literal", "type": "string"},
        },
        "reference_examples": ["$ref:baseline_ts", "$ref:counterfactual_ts"],
    },
    "rank_mechanism_support": {
        "output_type": "mechanism_score_result",
        "inputs": {
            "evidence_items": {"kind": "literal_or_ref", "type": "array"},
        },
        "reference_examples": ["$ref:mechanism_result"],
    },
    "grade_evidence_strength": {
        "output_type": "evidence_report_result",
        "inputs": {
            "evidence_items": {"kind": "literal_or_ref", "type": "array"},
        },
        "reference_examples": ["$ref:mechanism_result", "$ref:evidence_result"],
    },
    "assemble_mechanism_evidence_report": {
        "output_type": "evidence_report_result",
        "inputs": {
            "mechanism_scores": {"kind": "literal_or_ref", "type": "array"},
            "context_note": {"kind": "literal", "type": "string"},
        },
        "reference_examples": ["$ref:mechanism_result"],
    },
    "assemble_environment_health_report": {
        "output_type": "environment_assessment_result",
        "inputs": {
            "branches": {"kind": "literal_or_ref", "type": "array"},
            "context_note": {"kind": "literal", "type": "string"},
        },
        "reference_examples": ["$ref:trend_result", "$ref:event_statistics_result"],
    },
    "assemble_policy_recommendation_report": {
        "output_type": "policy_recommendation_result",
        "inputs": {
            "evidence_items": {"kind": "literal_or_ref", "type": "array"},
            "region_scope": {"kind": "literal", "type": "string"},
            "policy_context": {"kind": "literal", "type": "string"},
            "management_objective": {"kind": "literal", "type": "string"},
            "context_note": {"kind": "literal", "type": "string"},
        },
        "reference_examples": ["$ref:trend_result", "$ref:event_detection", "$ref:environment_health_assessment"],
    },
    "check_claim_support_level": {
        "output_type": "evidence_report_result",
        "inputs": {
            "claim": {"kind": "literal", "type": "string"},
            "evidence": {"kind": "literal_or_ref", "type": "object"},
            "requested_strength": {"kind": "literal", "type": "string"},
        },
        "reference_examples": ["$ref:mechanism_result", "$ref:evidence_result"],
    },
}

OUTPUT_REFERENCE_FIELDS: Dict[str, str] = {
    "data_container_result": "data",
    "field_trend_result": "data",
    "event_detection_result": "events",
}


def get_tool_contract(tool_name: str) -> Optional[Dict[str, Any]]:
    """按短名或完整名获取工具契约。"""
    if tool_name in TOOL_CONTRACTS:
        return TOOL_CONTRACTS[tool_name]

    short_name = tool_name.split(".")[-1]
    return TOOL_CONTRACTS.get(short_name)


def get_tool_output_type(tool_name: str) -> Optional[str]:
    """获取工具的显式输出类型。"""
    contract = get_tool_contract(tool_name)
    if not contract:
        return None
    return contract.get("output_type")


def get_param_contract(tool_name: str, param_name: str) -> Optional[Dict[str, Any]]:
    """获取工具某个参数的输入契约。"""
    contract = get_tool_contract(tool_name)
    if not contract:
        return None
    return contract.get("inputs", {}).get(param_name)


def get_primary_reference_field(output_type: Optional[str]) -> Optional[str]:
    """返回某类结果最常用的字段引用。"""
    if not output_type:
        return None
    return OUTPUT_REFERENCE_FIELDS.get(output_type)


def build_result_reference_examples(output_type: Optional[str]) -> Dict[str, str]:
    """
    为某个输出类型生成标准引用模板。

    这里使用 `<result_id>` 作为 planner 提示中的占位符。
    """
    examples = {
        "result": "$ref:<result_id>",
    }

    primary_field = get_primary_reference_field(output_type)
    if primary_field:
        examples["primary_field"] = f"$ref:<result_id>.{primary_field}"

    return examples


def build_param_reference_template(tool_name: str, param_name: str) -> Optional[str]:
    """按输入契约生成参数推荐引用模板。"""
    param_contract = get_param_contract(tool_name, param_name)
    if not param_contract:
        return None

    kind = param_contract.get("kind")
    expected = param_contract.get("expected")

    if kind == "ref_result":
        return "$ref:<result_id>"

    if kind == "ref_field" and isinstance(expected, str) and "." in expected:
        field_path = expected.split(".", 1)[1]
        return f"$ref:<result_id>.{field_path}"

    return None


def get_tool_planner_contract(tool_name: str) -> Dict[str, Any]:
    """
    生成给 planner 使用的轻量契约。

    输出重点不是 Python 类型，而是如何连接前后步骤。
    """
    contract = get_tool_contract(tool_name) or {}
    output_type = contract.get("output_type")
    inputs = contract.get("inputs", {})

    return {
        "output_type": output_type,
        "result_reference_examples": build_result_reference_examples(output_type),
        "params": {
            param_name: {
                **param_contract,
                "reference_template": build_param_reference_template(tool_name, param_name),
            }
            for param_name, param_contract in inputs.items()
        },
    }
