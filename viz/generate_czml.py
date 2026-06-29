"""Generate a CZML document for CesiumJS from the orbital risk analysis.

Builds a time-dynamic 3D scene: the target object, its actionable-risk
neighbours, their orbital paths, and conjunction lines that light up at each
time of closest approach (TCA).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from skyfield.api import load, wgs84

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts" / "orbital"))
sys.path.insert(0, str(_ROOT / "scripts" / "risk"))

from propagate import load_satellites  # noqa: E402
from conjunctions import screen_conjunctions  # noqa: E402
from classifier import classify_conjunctions  # noqa: E402

_VIZ_DIR = _ROOT / "viz"

_COLOR_TARGET = [181, 70, 47, 255]      # red
_COLOR_NEIGHBOR = [200, 205, 210, 255]   # gray
_COLOR_LINE = [224, 176, 90, 255]       # gold


def _iso(t) -> str:
    """Skyfield Time -> ISO 8601 UTC string."""
    return t.utc_strftime("%Y-%m-%dT%H:%M:%SZ")


def _position_property(sat, times, epoch_iso) -> dict:
    """CZML position via cartographicDegrees (lon, lat, height) + epoch."""
    geocentric = sat.at(times)
    sp = wgs84.subpoint(geocentric)
    lons, lats, heights = sp.longitude.degrees, sp.latitude.degrees, sp.elevation.m
    seconds = (times.tt - times.tt[0]) * 86400.0
    samples = []
    for k in range(len(times)):
        samples += [float(seconds[k]), float(lons[k]), float(lats[k]), float(heights[k])]
    return {
        "epoch": epoch_iso,
        "cartographicDegrees": samples,
        "interpolationAlgorithm": "LAGRANGE",
        "interpolationDegree": 5,
    }


def build_czml(target_catnr: int, hours: float = 24.0,
               step_minutes: float = 2.0, max_neighbors: int = 6) -> list:
    sats = load_satellites()
    by_catnr = {int(s.model.satnum): s for s in sats}
    if target_catnr not in by_catnr:
        raise ValueError(f"Target {target_catnr} not in catalog.")

    # Actionable neighbours from the risk pipeline
    conj = screen_conjunctions(target_catnr, hours=72.0, step_minutes=5.0, threshold_km=100.0)
    classified = classify_conjunctions(conj)
    actionable = (
        classified.filter(pl.col("risk_flag") == "ACIONÁVEL")
        .sort("actionable_proba", descending=True)
        .head(max_neighbors)
    )
    neighbor_catnrs = [c for c in actionable["neighbor_catnr"].to_list() if c in by_catnr]

    # Visualization time grid
    ts = load.timescale()
    t0 = ts.now()
    n_steps = int((hours * 60) / step_minutes) + 1
    minutes = np.arange(n_steps) * step_minutes
    times = ts.tt_jd(t0.tt + minutes / (24 * 60))
    epoch_iso, end_iso = _iso(times[0]), _iso(times[-1])

    target_sat = by_catnr[target_catnr]
    target_xyz = target_sat.at(times).position.km.T  # (T, 3), for TCA distance

    packets = [{
        "id": "document",
        "name": f"Orbital risk - object {target_catnr}",
        "version": "1.0",
        "clock": {
            "interval": f"{epoch_iso}/{end_iso}",
            "currentTime": epoch_iso,
            "multiplier": 600,
            "range": "LOOP_STOP",
            "step": "SYSTEM_CLOCK_MULTIPLIER",
        },
    }]

    # Target
    packets.append({
        "id": f"sat-{target_catnr}",
        "name": f"{target_catnr} (TARGET)",
        "availability": f"{epoch_iso}/{end_iso}",
        "position": _position_property(target_sat, times, epoch_iso),
        "point": {"color": {"rgba": _COLOR_TARGET}, "pixelSize": 14,
                  "outlineColor": {"rgba": [0, 0, 0, 255]}, "outlineWidth": 1},
        "path": {"material": {"solidColor": {"color": {"rgba": _COLOR_TARGET}}},
                 "width": 2, "leadTime": 0, "trailTime": 5400, "resolution": 120},
        "label": {"text": str(target_catnr), "font": "12pt sans-serif",
                  "fillColor": {"rgba": _COLOR_TARGET},
                  "pixelOffset": {"cartesian2": [14, 0]}, "scale": 0.8,
                  "showBackground": True},
    })

    # Neighbours + conjunction lines at TCA
    for cat in neighbor_catnrs:
        nsat = by_catnr[cat]
        packets.append({
            "id": f"sat-{cat}",
            "name": f"{cat} (debris)",
            "availability": f"{epoch_iso}/{end_iso}",
            "position": _position_property(nsat, times, epoch_iso),
            "point": {"color": {"rgba": _COLOR_NEIGHBOR}, "pixelSize": 10},
            "path": {"material": {"solidColor": {"color": {"rgba": _COLOR_NEIGHBOR}}},
            "width": 2, "leadTime": 0, "trailTime": 3600, "resolution": 120},
        })

        nxyz = nsat.at(times).position.km.T
        dist = np.linalg.norm(target_xyz - nxyz, axis=1)
        tca_k = int(np.argmin(dist))
        if dist[tca_k] > 100.0:
            continue  # no genuine close approach in-window; no flash
        tca = times[tca_k]
        lo = ts.tt_jd(tca.tt - 30 / (24 * 60))   # +/- 3 min around TCA
        hi = ts.tt_jd(tca.tt + 30 / (24 * 60))
        packets.append({
            "id": f"line-{cat}",
            "availability": f"{_iso(lo)}/{_iso(hi)}",
            "polyline": {
                "positions": {"references": [
                    f"sat-{target_catnr}#position", f"sat-{cat}#position"]},
                "material": {"solidColor": {"color": {"rgba": _COLOR_LINE}}},
                "width": 6,
            },
        })

    return packets


if __name__ == "__main__":
    czml = build_czml(target_catnr=22675, hours=72.0, step_minutes=2.0, max_neighbors=6)
    _VIZ_DIR.mkdir(parents=True, exist_ok=True)
    out = _VIZ_DIR / "orbits.czml"
    with open(out, "w") as f:
        json.dump(czml, f)

    n_sats = sum(1 for p in czml if str(p.get("id", "")).startswith("sat-"))
    n_lines = sum(1 for p in czml if str(p.get("id", "")).startswith("line-"))
    print(f"CZML packets: {len(czml)}  satellites: {n_sats}  conjunction lines: {n_lines}")
    print(f"Saved: {out}")