"""Servidor MCP mínimo (spike): expõe uma tool trivial via FastMCP."""
from mcp.server.fastmcp import FastMCP

# Cria o servidor MCP
mcp = FastMCP("orbital-tools")


@mcp.tool()
def get_orbital_period(altitude_km: float) -> dict:
    """Calcula o período orbital de uma órbita circular a partir da altitude.

    Args:
        altitude_km: altitude acima da superfície da Terra, em km.

    Returns:
        dict com o período orbital em minutos.
    """
    import math

    # Constantes (Terra)
    mu = 398600.4418  # parâmetro gravitacional padrão, km^3/s^2
    earth_radius_km = 6378.137

    # Raio da órbita = raio da Terra + altitude
    r = earth_radius_km + altitude_km
    # Terceira lei de Kepler: T = 2*pi*sqrt(r^3 / mu)
    period_s = 2 * math.pi * math.sqrt(r**3 / mu)

    return {"altitude_km": altitude_km, "period_min": round(period_s / 60, 2)}


if __name__ == "__main__":
    # Roda o servidor via stdio (o agente vai iniciá-lo como subprocesso)
    mcp.run(transport="stdio")