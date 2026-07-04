"""
Eddy tracking across multiple time steps.
"""

from typing import Dict, List, Optional

import numpy as np
import xarray as xr

from .detect import detect_eddies


def track_eddies(
    u: xr.DataArray,
    v: xr.DataArray,
    ow_threshold: float = -2e-12,
    min_radius_km: float = 30.0,
    max_radius_km: float = 300.0,
    min_pixels: int = 10,
    max_displacement_km: float = 200.0,
    radius_change_ratio: float = 0.75,
    min_lifespan: int = 2
) -> Dict:
    """
    Track eddies through time using nearest-neighbor matching with simple
    physical consistency constraints.
    """
    u, v = xr.align(u, v, join='inner')
    if 'time' not in u.dims:
        raise ValueError("track_eddies requires velocity fields with a time dimension")

    if 'depth' in u.dims:
        u = u.isel(depth=0)
    if 'depth' in v.dims:
        v = v.isel(depth=0)

    active_tracks: List[Dict] = []
    completed_tracks: List[Dict] = []
    next_track_id = 1

    for time_index in range(u.sizes['time']):
        detection = detect_eddies(
            u.isel(time=time_index),
            v.isel(time=time_index),
            ow_threshold=ow_threshold,
            min_radius_km=min_radius_km,
            max_radius_km=max_radius_km,
            min_pixels=min_pixels,
        )
        step_eddies = []
        for event in detection['events']:
            entry = dict(event)
            entry['time_index'] = time_index
            step_eddies.append(entry)

        matched_track_ids = set()
        for eddy in step_eddies:
            track = _find_best_track(
                eddy,
                active_tracks,
                matched_track_ids,
                max_displacement_km,
                radius_change_ratio,
            )
            if track is None:
                track = {
                    'track_id': f'eddy_track_{next_track_id}',
                    'type': eddy.get('type'),
                    'events': [],
                }
                next_track_id += 1
                active_tracks.append(track)

            track['events'].append(eddy)
            matched_track_ids.add(track['track_id'])

        still_active = []
        for track in active_tracks:
            last_time = track['events'][-1]['time_index']
            if last_time == time_index:
                still_active.append(track)
            else:
                completed_tracks.append(track)
        active_tracks = still_active

    completed_tracks.extend(active_tracks)
    tracks = [_summarize_track(track) for track in completed_tracks if len(track['events']) >= min_lifespan]

    statistics = {
        'total_count': len(tracks),
        'mean_lifespan': float(np.mean([track['lifespan_steps'] for track in tracks])) if tracks else 0.0,
        'max_lifespan': int(np.max([track['lifespan_steps'] for track in tracks])) if tracks else 0,
        'detection_params': {
            'ow_threshold': ow_threshold,
            'min_radius_km': min_radius_km,
            'max_radius_km': max_radius_km,
            'max_displacement_km': max_displacement_km,
            'radius_change_ratio': radius_change_ratio,
            'min_lifespan': min_lifespan,
        },
    }

    return {
        'event_type': 'eddy_track',
        'events': tracks,
        'statistics': statistics,
        'coordinates': {
            'lon': u.lon.values.tolist(),
            'lat': u.lat.values.tolist(),
        },
    }


def _find_best_track(
    eddy: Dict,
    tracks: List[Dict],
    matched_track_ids: set,
    max_displacement_km: float,
    radius_change_ratio: float,
) -> Optional[Dict]:
    best_track = None
    best_distance = None
    for track in tracks:
        if track['track_id'] in matched_track_ids:
            continue
        last = track['events'][-1]
        if track.get('type') != eddy.get('type'):
            continue
        distance = _haversine_km(
            last['center']['lon'],
            last['center']['lat'],
            eddy['center']['lon'],
            eddy['center']['lat'],
        )
        if distance > max_displacement_km:
            continue

        prev_radius = float(last.get('radius_km', 0.0))
        new_radius = float(eddy.get('radius_km', 0.0))
        if prev_radius > 0:
            ratio = abs(new_radius - prev_radius) / prev_radius
            if ratio > radius_change_ratio:
                continue

        if best_distance is None or distance < best_distance:
            best_track = track
            best_distance = distance
    return best_track


def _summarize_track(track: Dict) -> Dict:
    events = track['events']
    return {
        'track_id': track['track_id'],
        'type': track.get('type'),
        'lifespan_steps': int(len(events)),
        'start_time_index': int(events[0]['time_index']),
        'end_time_index': int(events[-1]['time_index']),
        'mean_radius_km': float(np.mean([event.get('radius_km', 0.0) for event in events])),
        'mean_area_km2': float(np.mean([event.get('area_km2', 0.0) for event in events])),
        'path': [event['center'] for event in events],
        'events': events,
    }


def _haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(np.deg2rad, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return float(2.0 * R * np.arcsin(np.sqrt(a)))
