"""Sistema multi-agente de análise de risco orbital (ADK SequentialAgent).

Três sub-agentes especializados executam em ordem fixa, compartilhando o
estado da sessão:
  1. data_agent     -> tria conjunções e classifica risco (tool MCP)
  2. analysis_agent -> avalia criticidade na rede do enxame (tool MCP)
  3. briefing_agent -> sintetiza o briefing operacional em português
"""
import os

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

_MODEL = "gemini-2.5-flash"
_SERVER_PATH = os.path.join(os.path.dirname(__file__), "mcp_server.py")

# Toolset MCP compartilhado (sobe o servidor como subprocesso via stdio)
_orbital_tools = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="python",
            args=[_SERVER_PATH],
        ),
    ),
)

# --- Sub-agente 1: dados ---
data_agent = LlmAgent(
    model=_MODEL,
    name="data_agent",
    description="Tria conjunções do objeto-alvo e classifica o risco de cada uma.",
    instruction=(
        "Você é o agente de dados orbitais. Dado um número de catálogo NORAD, "
        "use a ferramenta analyze_conjunctions para obter as conjunções e suas "
        "classificações de risco. Relate de forma objetiva: total de conjunções, "
        "quantas são acionáveis, e a lista das principais com distância, "
        "velocidade relativa e flag de risco. Não invente dados."
    ),
    tools=[_orbital_tools],
    output_key="dados_conjuncoes",
)

# --- Sub-agente 2: análise de rede ---
analysis_agent = LlmAgent(
    model=_MODEL,
    name="analysis_agent",
    description="Avalia a criticidade do objeto na rede de conjunções do enxame.",
    instruction=(
        "Você é o agente de análise de rede. Use a ferramenta network_role para "
        "medir a centralidade do objeto-alvo na rede de risco do enxame. "
        "Interprete: percentil alto (>0.7) indica um HUB — objeto cuja "
        "fragmentação propagaria a cascata de detritos; percentil baixo indica "
        "objeto periférico. Combine com os dados de conjunção a seguir:\n\n"
        "{dados_conjuncoes}\n\n"
        "Produza um veredito de risco em duas dimensões: risco de colisão "
        "(das conjunções acionáveis) e criticidade estrutural (da centralidade)."
    ),
    tools=[_orbital_tools],
    output_key="analise_risco",
)

# --- Sub-agente 3: briefing ---
briefing_agent = LlmAgent(
    model=_MODEL,
    name="briefing_agent",
    description="Sintetiza o briefing operacional final.",
    instruction=(
        "Você é o agente de briefing. Com base na análise a seguir, escreva um "
        "briefing operacional conciso em português para um operador de satélite. "
        "Estrutura: (1) resumo do alvo, (2) conjunções acionáveis com prazo até a "
        "aproximação, (3) criticidade do objeto na rede, (4) recomendação. "
        "Tom técnico e direto.\n\nAnálise:\n{analise_risco}"
    ),
    output_key="briefing_final",
)

# --- Orquestração sequencial ---
root_agent = SequentialAgent(
    name="orbital_risk_pipeline",
    sub_agents=[data_agent, analysis_agent, briefing_agent],
)