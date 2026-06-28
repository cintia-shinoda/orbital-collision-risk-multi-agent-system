from google.adk.agents.llm_agent import Agent

root_agent = Agent(
    model='gemini-2.5-flash',
    name='root_agent',
    description='A helpful assistant for user questions.',
    instruction='Answer user questions to the best of your knowledge',
)


"""Agente raiz: consome a tool MCP via McpToolset."""
import os

from google.adk.agents.llm_agent import Agent
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

# Caminho absoluto para o servidor MCP que está ao lado deste arquivo
_SERVER_PATH = os.path.join(os.path.dirname(__file__), "mcp_server.py")

root_agent = Agent(
    model="gemini-2.5-flash",
    name="root_agent",
    description="Assistente de análise orbital.",
    instruction=(
        "Você ajuda com cálculos orbitais. "
        "Quando o usuário perguntar sobre período orbital, use a ferramenta disponível."
    ),
    tools=[
        McpToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command="python",
                    args=[_SERVER_PATH],
                ),
            ),
        ),
    ],
)