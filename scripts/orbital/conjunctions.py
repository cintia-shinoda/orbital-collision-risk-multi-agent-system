"""Triagem de conjunções (one-vs-all) via filtro espacial KDTree.

Dado um objeto-alvo, identifica candidatos a conjunção: objetos que se
aproximam abaixo de um limiar em qualquer instante da janela propagada.

Pipeline de dois estágios (padrão operacional):
  1. Filtro grosso: KDTree localiza vizinhos dentro do limiar a cada passo.
  2. Refino: para cada candidato, distância mínima exata + velocidade
     relativa real (do SGP4) no instante de maior aproximação.

NOTA: numa grade de 5 min, objetos LEO percorrem ~2200 km entre passos.
Esta é uma triagem GROSSA (candidatos), não o TCA preciso, que exigiria
refino temporal fino — documentado como próximo passo.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from scipy.spatial import cKDTree

from propagate import load_satellites, propagate

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def screen_conjunctions(
    target_catnr: int,
    hours: float = 24.0,
    step_minutes: float = 5.0,
    threshold_km: float = 50.0,
) -> pl.DataFrame:
    """Tria candidatos a conjunção com o objeto-alvo (one-vs-all).

    Returns:
        DataFrame: neighbor_catnr, neighbor_name, min_distance_km,
        t_ca_min (minutos desde o início), rel_speed_km_s.
    """
    sats = load_satellites()
    times, positions = propagate(sats, hours=hours, step_minutes=step_minutes)

    catnrs = np.array([s.model.satnum for s in sats])
    matches = np.where(catnrs == target_catnr)[0]
    if matches.size == 0:
        raise ValueError(f"Alvo {target_catnr} não está no catálogo.")
    target_idx = int(matches[0])

    n_steps = positions.shape[1]

    # --- Estágio 1: filtro grosso com KDTree ---
    candidate_idxs: set[int] = set()
    for t in range(n_steps):
        tree = cKDTree(positions[:, t, :])
        neighbors = tree.query_ball_point(positions[target_idx, t, :], r=threshold_km)
        candidate_idxs.update(neighbors)
    candidate_idxs.discard(target_idx)

    # --- Estágio 2: refino exato por candidato ---
    target_track = positions[target_idx]  # (T, 3)
    rows = []
    for j in candidate_idxs:
        dist = np.linalg.norm(target_track - positions[j], axis=1)  # (T,)
        t_ca = int(np.argmin(dist))

        # Velocidade relativa REAL do SGP4 no instante de maior aproximação
        v_target = sats[target_idx].at(times[t_ca]).velocity.km_per_s
        v_neighbor = sats[j].at(times[t_ca]).velocity.km_per_s
        rel_speed = float(np.linalg.norm(v_target - v_neighbor))

        rows.append(
            {
                "neighbor_catnr": int(catnrs[j]),
                "min_distance_km": round(float(dist[t_ca]), 3),
                "t_ca_min": round(t_ca * step_minutes, 1),
                "rel_speed_km_s": round(rel_speed, 3),
            }
        )

    schema = {
        "neighbor_catnr": pl.Int64,
        "min_distance_km": pl.Float64,
        "t_ca_min": pl.Float64,
        "rel_speed_km_s": pl.Float64,
    }
    if not rows:
        return pl.DataFrame(schema=schema)

    result = pl.DataFrame(rows).sort("min_distance_km")

    # Junta o nome legível a partir do catálogo
    catalog = pl.read_parquet(_DATA_DIR / "catalog.parquet").select(
        pl.col("NORAD_CAT_ID").alias("neighbor_catnr"),
        pl.col("OBJECT_NAME").alias("neighbor_name"),
    )
    return result.join(catalog, on="neighbor_catnr", how="left")


def screen_all_pairs(
    hours: float = 24.0,
    step_minutes: float = 5.0,
    threshold_km: float = 50.0,
) -> pl.DataFrame:
    """Tria TODOS os pares de conjunção dentro do catálogo (all-on-all).

    Constrói a rede de risco do enxame: cada aresta é um par de objetos que
    se aproxima abaixo do limiar em algum instante da janela.

    Returns:
        DataFrame: catnr_a, catnr_b, min_distance_km, rel_speed_km_s.
        (catnr_a < catnr_b sempre, para evitar arestas duplicadas.)
    """
    sats = load_satellites()
    times, positions = propagate(sats, hours=hours, step_minutes=step_minutes)
    catnrs = np.array([s.model.satnum for s in sats])
    n_steps = positions.shape[1]

    # --- Estágio 1: KDTree query_pairs a cada instante (filtro grosso) ---
    candidate_pairs: set[tuple[int, int]] = set()
    for t in range(n_steps):
        tree = cKDTree(positions[:, t, :])
        # query_pairs devolve todos os pares (i, j) dentro do limiar
        candidate_pairs.update(tree.query_pairs(r=threshold_km))

    # --- Estágio 2: refino exato por par candidato ---
    rows = []
    for i, j in candidate_pairs:
        dist = np.linalg.norm(positions[i] - positions[j], axis=1)  # (T,)
        t_ca = int(np.argmin(dist))

        v_i = sats[i].at(times[t_ca]).velocity.km_per_s
        v_j = sats[j].at(times[t_ca]).velocity.km_per_s
        rel_speed = float(np.linalg.norm(v_i - v_j))

        # Ordena o par por catnr para aresta canônica (sem duplicar)
        a, b = sorted((int(catnrs[i]), int(catnrs[j])))
        rows.append(
            {
                "catnr_a": a,
                "catnr_b": b,
                "min_distance_km": round(float(dist[t_ca]), 3),
                "rel_speed_km_s": round(rel_speed, 3),
            }
        )

    schema = {
        "catnr_a": pl.Int64,
        "catnr_b": pl.Int64,
        "min_distance_km": pl.Float64,
        "rel_speed_km_s": pl.Float64,
    }
    if not rows:
        return pl.DataFrame(schema=schema)

    return pl.DataFrame(rows).unique(subset=["catnr_a", "catnr_b"]).sort("min_distance_km")


if __name__ == "__main__":
    edges = screen_all_pairs(hours=72.0, step_minutes=5.0, threshold_km=100.0)
    edges.write_parquet(_DATA_DIR / "conjunction_edges.parquet")

    print(f"Arestas (pares de conjunção): {edges.height}")
    print(edges.head(10))

    # Quantos objetos distintos participam de ao menos uma conjunção?
    nodes = set(edges["catnr_a"].to_list()) | set(edges["catnr_b"].to_list())
    print(f"Nós com ao menos uma aresta: {len(nodes)}")