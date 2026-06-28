"""Construção do grafo de conjunções e centralidade dos nós.

Lê as arestas all-on-all e constrói a rede de risco do enxame. Calcula
centralidade para medir a criticidade de cada objeto na propagação de risco
(lógica Hub/Ponte/Periférico reusada do TCC).
"""
from __future__ import annotations

from pathlib import Path

import networkx as nx
import polars as pl

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def build_graph(edges_file: str = "conjunction_edges.parquet") -> nx.Graph:
    """Constrói o grafo de conjunções.

    Nó = objeto (NORAD ID). Aresta = conjunção, com peso de risco =
    velocidade relativa / distância mínima (mais perto + mais rápido =>
    encontro mais perigoso).
    """
    edges = pl.read_parquet(_DATA_DIR / edges_file)
    graph = nx.Graph()
    for row in edges.iter_rows(named=True):
        risk = row["rel_speed_km_s"] / (row["min_distance_km"] + 1e-6)
        graph.add_edge(
            row["catnr_a"],
            row["catnr_b"],
            min_distance_km=row["min_distance_km"],
            rel_speed_km_s=row["rel_speed_km_s"],
            risk=risk,
        )
    return graph


def compute_centrality(graph: nx.Graph) -> pl.DataFrame:
    """Calcula métricas de centralidade por nó.

    Returns:
        DataFrame: catnr, degree, weighted_degree, betweenness.
    """
    degree = dict(graph.degree())
    weighted_degree = dict(graph.degree(weight="risk"))
    # Intermediação: distância como custo (perto = laço forte)
    betweenness = nx.betweenness_centrality(graph, weight="min_distance_km")

    rows = [
        {
            "catnr": int(n),
            "degree": int(degree[n]),
            "weighted_degree": round(float(weighted_degree[n]), 4),
            "betweenness": round(float(betweenness[n]), 5),
        }
        for n in graph.nodes()
    ]
    return pl.DataFrame(rows).sort("weighted_degree", descending=True)


if __name__ == "__main__":
    g = build_graph()
    print(f"Nós: {g.number_of_nodes()}  Arestas: {g.number_of_edges()}")

    components = list(nx.connected_components(g))
    print(f"Componentes conexos: {len(components)}")
    print(f"Maior componente: {len(max(components, key=len))} nós")

    centrality = compute_centrality(g)
    centrality.write_parquet(_DATA_DIR / "node_centrality.parquet")

    catalog = pl.read_parquet(_DATA_DIR / "catalog.parquet").select(
        pl.col("NORAD_CAT_ID").alias("catnr"),
        pl.col("OBJECT_NAME").alias("name"),
    )
    print("\nTop 5 por grau ponderado de risco:")
    print(centrality.join(catalog, on="catnr", how="left").head(5))