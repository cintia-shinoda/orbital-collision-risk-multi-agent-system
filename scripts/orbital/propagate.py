"""Propagação orbital: OMM -> posições XYZ numa grade de tempo.

Carrega os registros OMM do catálogo, constrói satélites SGP4 via Skyfield
e propaga suas posições geocêntricas (GCRS, km) numa janela de tempo.
"""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl
from sgp4 import omm
from sgp4.api import Satrec
from skyfield.api import EarthSatellite, load

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_satellites(filename: str = "catalog.parquet") -> list[EarthSatellite]:
    """Carrega o catálogo Parquet e constrói objetos EarthSatellite do Skyfield."""
    df = pl.read_parquet(_DATA_DIR / filename)
    ts = load.timescale()

    satellites: list[EarthSatellite] = []
    for record in df.iter_rows(named=True):
        # Reconstrói o Satrec a partir do dicionário OMM
        sat = Satrec()
        omm.initialize(sat, record)
        # Envelopa no EarthSatellite do Skyfield (usa o nome legível)
        satellites.append(EarthSatellite.from_satrec(sat, ts))
    return satellites


def propagate(
    satellites: list[EarthSatellite],
    hours: float = 24.0,
    step_minutes: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Propaga os satélites numa grade de tempo.

    Args:
        satellites: lista de EarthSatellite.
        hours: duração da janela a partir de agora, em horas.
        step_minutes: passo da grade temporal, em minutos.

    Returns:
        times: array Skyfield Time (T instantes).
        positions: array (N_sats, T, 3) com posições GCRS em km.
    """
    ts = load.timescale()
    now = ts.now()

    # Grade temporal: now, now+step, ... até now+hours
    n_steps = int((hours * 60) / step_minutes) + 1
    minutes = np.arange(n_steps) * step_minutes
    times = ts.tt_jd(now.tt + minutes / (24 * 60))

    # Para cada satélite, posição GCRS (km) em cada instante
    positions = np.empty((len(satellites), n_steps, 3))
    for i, sat in enumerate(satellites):
        geocentric = sat.at(times)
        positions[i] = geocentric.position.km.T  # (T, 3)

    return times, positions


if __name__ == "__main__":
    sats = load_satellites()
    print(f"Satélites carregados: {len(sats)}")

    times, positions = propagate(sats, hours=6.0, step_minutes=5.0)
    print(f"Formato das posições: {positions.shape}  (N_sats, T, 3)")

    # Sanidade: posição do primeiro satélite no primeiro instante
    print(f"Exemplo XYZ (km): {positions[0, 0]}")
    # Distância ao centro da Terra (deve ser ~6800-7200 km em LEO)
    r = np.linalg.norm(positions[0, 0])
    print(f"Raio orbital do exemplo: {r:.1f} km")
