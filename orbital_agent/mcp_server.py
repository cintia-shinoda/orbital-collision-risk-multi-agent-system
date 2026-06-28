"""Servidor MCP do agente orbital (FastMCP).

Expõe as funções da camada orbital e de risco como ferramentas MCP,
consumíveis pelo agente ADK. Importa a lógica de scripts/orbital e scripts/risk.
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
from mcp.server.fastmcp import FastMCP

# scripts/ não é um pacote; coloca os módulos de lógica no path
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts" / "orbital"))
sys.path.insert(0, str(_ROOT / "scripts" / "risk"))

from conjunctions import screen_conjunctions  # noqa: E402
from graph import build_graph, compute_centrality  # noqa: E402
from classifier import classify_conjunctions  # noqa: E402

mcp = FastMCP("orbital-tools")


@mcp.tool()
def analyze_conjunctions(target_catnr: int, top_n: int = 10) -> dict:
    """Tria conjunções do objeto-alvo e classifica o risco de cada uma.

    Args:
        target_catnr: número de catálogo NORAD do objeto-alvo.
        top_n: quantas conjunções de maior risco retornar.

    Returns:
        dict com contagens e a lista das conjunções de maior risco.
    """
    conjunctions = screen_conjunctions(
        target_catnr=target_catnr, hours=72.0, step_minutes=5.0, threshold_km=100.0
    )
    if conjunctions.height == 0:
        return {
            "target_catnr": target_catnr,
            "n_conjunctions": 0,
            "n_actionable": 0,
            "conjunctions": [],
        }

    classified = classify_conjunctions(conjunctions).sort(
        "actionable_proba", descending=True
    )
    n_actionable = int((classified["risk_flag"] == "ACIONÁVEL").sum())

    top = classified.head(top_n).select(
        "neighbor_catnr", "neighbor_name", "min_distance_km",
        "rel_speed_km_s", "t_ca_min", "risk_flag", "actionable_proba",
    )
    return {
        "target_catnr": target_catnr,
        "n_conjunctions": classified.height,
        "n_actionable": n_actionable,
        "conjunctions": top.to_dicts(),
    }


@mcp.tool()
def network_role(target_catnr: int) -> dict:
    """Mede a criticidade do objeto na rede de conjunções do enxame.

    Args:
        target_catnr: número de catálogo NORAD do objeto-alvo.

    Returns:
        dict com grau, grau ponderado, intermediação e ranking percentil
        (hub vs periférico) na rede de risco.
    """
    graph = build_graph()
    centrality = compute_centrality(graph)

    if target_catnr not in centrality["catnr"].to_list():
        return {
            "target_catnr": target_catnr,
            "in_network": False,
            "note": "Objeto sem conjunções na rede do enxame.",
        }

    # Ranking percentil por grau ponderado de risco (1.0 = hub máximo)
    ranked = centrality.with_columns(
        (pl.col("weighted_degree").rank() / pl.len()).alias("pctl")
    )
    row = ranked.filter(pl.col("catnr") == target_catnr).to_dicts()[0]
    return {
        "target_catnr": target_catnr,
        "in_network": True,
        "degree": row["degree"],
        "weighted_degree": row["weighted_degree"],
        "betweenness": row["betweenness"],
        "centrality_percentile": round(row["pctl"], 3),
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")