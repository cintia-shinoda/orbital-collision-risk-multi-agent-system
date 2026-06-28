"""Ingestão de dados orbitais GP do CelesTrak (formato JSON/OMM).

Busca elementos GP (General Perturbations) via API HTTP do CelesTrak e
salva o catálogo em Parquet. Usa JSON/OMM — não o TLE legado, que está
esgotando os números de catálogo de 5 dígitos.
"""
from __future__ import annotations

from pathlib import Path

import httpx
import polars as pl

# Endpoint base da API GP do CelesTrak
_GP_URL = "https://celestrak.org/NORAD/elements/gp.php"

# Diretório de saída (na raiz do projeto: scripts/orbital -> raiz)
_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def fetch_gp(group: str | None = None, catnr: int | None = None) -> pl.DataFrame:
    """Busca elementos GP do CelesTrak em formato JSON (OMM).

    Forneça exatamente UM dos parâmetros:
        group: grupo de satélites (ex.: "active", "starlink", "last-30-days").
        catnr: número de catálogo NORAD de um único objeto.

    Returns:
        DataFrame Polars com um registro OMM por objeto.
    """
    # Exige exatamente um dos dois parâmetros
    if (group is None) == (catnr is None):
        raise ValueError("Forneça exatamente um: group OU catnr.")

    params = {"FORMAT": "json"}
    if group is not None:
        params["GROUP"] = group
    else:
        params["CATNR"] = str(catnr)

    # follow_redirects trata o 301 de .com/.net para .org
    response = httpx.get(_GP_URL, params=params, timeout=30.0, follow_redirects=True)
    response.raise_for_status()

    records = response.json()
    if not records:
        raise ValueError("Nenhum objeto retornado. Verifique group/catnr.")

    return pl.DataFrame(records)


def save_catalog(df: pl.DataFrame, filename: str = "catalog.parquet") -> Path:
    """Salva o catálogo em Parquet no diretório data/."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = _DATA_DIR / filename
    df.write_parquet(path)
    return path


if __name__ == "__main__":
    # Teste manual: detritos do Cosmos 2251 — primeira colisão acidental de
    # satélites (2009). Ótimo campo de risco para a demo do agente.
    catalog = fetch_gp(group="cosmos-2251-debris")
    path = save_catalog(catalog)

    print(f"Objetos obtidos: {catalog.height}")
    print(f"Colunas: {catalog.columns[:6]} ...")
    print(catalog.select(["OBJECT_NAME", "NORAD_CAT_ID", "EPOCH"]).head())
    print(f"Salvo em: {path}")