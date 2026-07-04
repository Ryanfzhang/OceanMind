"""
时间序列分析工具（补充）

添加去季节循环和滞后相关分析
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


def remove_seasonal_cycle(timeseries: Dict) -> Dict:
    """
    去除时间序列的季节循环

    对月尺度及更粗分辨率数据，按 month-of-year 估计季节循环；
    对日尺度和更高频率数据，在存在足够重复 calendar-day 样本时按
    calendar-day 估计季节循环，否则回退到 month-of-year。

    Args:
        timeseries: 时间序列字典（来自extract_regional_mean等）

    Returns:
        去季节化后的时间序列

    Example:
        >>> deseasoned = remove_seasonal_cycle(timeseries)
        >>> print(deseasoned['metadata']['seasonal_cycle'])
    """
    times = pd.to_datetime(timeseries['times'])
    values = np.array(timeseries['values'], dtype=float)

    if len(times) < 12:
        raise ValueError("Time series too short for seasonal cycle removal (need >= 12 points)")

    seasonal_method, seasonal_keys = _infer_seasonal_cycle_keys(times)
    unique_keys = sorted(pd.Index(seasonal_keys).unique().tolist())
    seasonal_lookup = {}
    seasonal_cycle = []
    for key in unique_keys:
        mask = seasonal_keys == key
        seasonal_value = float(np.nanmean(values[mask])) if np.any(mask) else np.nan
        seasonal_lookup[key] = seasonal_value
        seasonal_cycle.append(seasonal_value)

    deseasoned = values - np.array([seasonal_lookup[key] for key in seasonal_keys], dtype=float)

    # 计算统计信息
    statistics = {
        'mean': float(np.nanmean(deseasoned)),
        'std': float(np.nanstd(deseasoned)),
        'min': float(np.nanmin(deseasoned)),
        'max': float(np.nanmax(deseasoned)),
        'n_valid': int(np.sum(~np.isnan(deseasoned))),
        'n_total': len(deseasoned)
    }

    return {
        'times': timeseries['times'],
        'values': deseasoned.tolist(),
        'metadata': {
            **timeseries.get('metadata', {}),
            'is_deseasoned': True,
            'seasonal_cycle': seasonal_cycle,
            'seasonal_cycle_labels': [str(key) for key in unique_keys],
            'seasonal_cycle_method': seasonal_method,
            'statistics': statistics
        }
    }


def compute_lag_correlation(
    timeseries1: Dict,
    timeseries2: Dict,
    max_lag: int = 12,
    confidence_level: float = 0.95,
) -> Dict:
    """
    计算两个时间序列的滞后相关性

    Args:
        timeseries1: 第一个时间序列
        timeseries2: 第二个时间序列
        max_lag: 最大滞后步数
        confidence_level: 置信水平

    Returns:
        滞后相关分析结果

    Example:
        >>> corr = compute_lag_correlation(ts1, ts2, max_lag=12)
        >>> print(f"Optimal lag: {corr['optimal_lag']}")
    """
    from scipy import stats

    values1 = np.array(timeseries1['values'], dtype=float)
    values2 = np.array(timeseries2['values'], dtype=float)
    times1 = _coerce_datetime_index(timeseries1.get('times', []))
    times2 = _coerce_datetime_index(timeseries2.get('times', []))

    # 确保长度相同
    min_len = min(len(values1), len(values2))
    values1 = values1[:min_len]
    values2 = values2[:min_len]
    if times1 is not None:
        times1 = times1[:min_len]
    if times2 is not None:
        times2 = times2[:min_len]

    # 去除NaN
    mask = ~(np.isnan(values1) | np.isnan(values2))
    values1 = values1[mask]
    values2 = values2[mask]
    if times1 is not None:
        times1 = times1[mask]
    if times2 is not None:
        times2 = times2[mask]

    if len(values1) < max_lag + 10:
        raise ValueError(f"Time series too short for lag analysis (need >= {max_lag + 10} points)")

    # 计算不同滞后的相关系数
    lags = _lag_candidates(max_lag)
    correlations = []
    p_values = []

    for lag in lags:
        if lag < 0:
            # 负滞后：ts1滞后于ts2
            v1 = values1[:lag]
            v2 = values2[-lag:]
        elif lag > 0:
            # 正滞后：ts1领先于ts2
            v1 = values1[lag:]
            v2 = values2[:-lag]
        else:
            # 零滞后
            v1 = values1
            v2 = values2

        if len(v1) > 0:
            corr, pval = stats.pearsonr(v1, v2)
            correlations.append(float(corr))
            p_values.append(float(pval))
        else:
            correlations.append(np.nan)
            p_values.append(np.nan)

    # 找到最大相关系数
    correlations_array = np.array(correlations)
    if not np.any(np.isfinite(correlations_array)):
        raise ValueError("Lag correlation undefined because all lagged correlations are NaN")
    max_idx = _select_optimal_lag_index(lags, correlations_array)
    optimal_lag = lags[max_idx]
    max_correlation = correlations[max_idx]

    # 计算置信界限
    n = len(values1)
    confidence_bound = stats.norm.ppf((1 + confidence_level) / 2) / np.sqrt(n)

    metadata1 = timeseries1.get('metadata', {}) if isinstance(timeseries1.get('metadata'), dict) else {}
    metadata2 = timeseries2.get('metadata', {}) if isinstance(timeseries2.get('metadata'), dict) else {}
    analysis_mode = _infer_lag_analysis_mode(metadata1, metadata2)
    step_days = _estimate_time_step_days(times1)

    result = {
        'lags': lags,
        'correlations': correlations,
        'p_values': p_values,
        'optimal_lag': optimal_lag,
        'max_correlation': float(max_correlation),
        'confidence_bound': float(confidence_bound),
        'confidence_level': confidence_level,
        'ts1_label': metadata1.get('variable', 'ts1'),
        'ts2_label': metadata2.get('variable', 'ts2'),
        'n_points': n,
        'analysis_mode': analysis_mode,
        'ts1_is_deseasoned': bool(metadata1.get('is_deseasoned')),
        'ts2_is_deseasoned': bool(metadata2.get('is_deseasoned')),
        'ts1_seasonal_cycle_method': metadata1.get('seasonal_cycle_method'),
        'ts2_seasonal_cycle_method': metadata2.get('seasonal_cycle_method'),
    }
    if step_days is not None:
        result['median_step_days'] = float(step_days)
        result['optimal_lag_days'] = float(optimal_lag * step_days)

    return result


def _infer_seasonal_cycle_keys(times: pd.DatetimeIndex) -> Tuple[str, np.ndarray]:
    median_step_days = _estimate_time_step_days(times)
    if median_step_days is not None and median_step_days <= 10.0:
        calendar_day_keys = times.strftime("%m-%d").to_numpy()
        key_counts = pd.Series(calendar_day_keys).value_counts()
        if not key_counts.empty and float(key_counts.median()) >= 2.0:
            return "calendar_day", calendar_day_keys
    return "month_of_year", times.month.to_numpy()


def _estimate_time_step_days(times: Optional[pd.DatetimeIndex]) -> Optional[float]:
    if times is None or len(times) < 2:
        return None
    deltas = np.diff(times.values).astype("timedelta64[s]").astype(float) / 86400.0
    deltas = deltas[np.isfinite(deltas) & (deltas > 0)]
    if deltas.size == 0:
        return None
    return float(np.nanmedian(deltas))


def _coerce_datetime_index(values) -> Optional[pd.DatetimeIndex]:
    if values is None:
        return None
    try:
        times = pd.to_datetime(values)
    except (TypeError, ValueError):
        return None
    if len(times) == 0:
        return None
    return pd.DatetimeIndex(times)


def _infer_lag_analysis_mode(metadata1: Dict, metadata2: Dict) -> str:
    ts1_deseasoned = bool(metadata1.get('is_deseasoned'))
    ts2_deseasoned = bool(metadata2.get('is_deseasoned'))
    if ts1_deseasoned and ts2_deseasoned:
        return "deseasoned"
    if not ts1_deseasoned and not ts2_deseasoned:
        return "raw"
    return "mixed"


def _lag_candidates(max_lag: int) -> List[int]:
    max_lag = int(max_lag)
    if max_lag < 0:
        raise ValueError("max_lag must be non-negative")
    return list(range(-max_lag, max_lag + 1))


def _select_optimal_lag_index(
    lags: List[int],
    correlations: np.ndarray,
) -> int:
    if not any(np.isfinite(value) for value in correlations):
        raise ValueError("Lag correlation undefined because all lagged correlations are NaN")
    return int(np.nanargmax(np.abs(correlations)))
