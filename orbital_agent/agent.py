"""Multi-agent orbital collision-risk pipeline (ADK SequentialAgent).

Three specialized sub-agents run in a fixed order, sharing session state:
  1. data_agent     -> screens conjunctions and classifies risk (MCP tool)
  2. analysis_agent -> assesses criticality in the swarm network (MCP tool)
  3. briefing_agent -> writes the operational briefing in English
"""
import os

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

_MODEL = "gemini-3.5-flash"
_SERVER_PATH = os.path.join(os.path.dirname(__file__), "mcp_server.py")

# Shared MCP toolset (spawns the server as a stdio subprocess)
_orbital_tools = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="python",
            args=[_SERVER_PATH],
        ),
    ),
)

# --- Sub-agent 1: data ---
data_agent = LlmAgent(
    model=_MODEL,
    name="data_agent",
    description="Resolves a satellite name to a NORAD ID if needed, then screens conjunctions and classifies risk.",
    instruction=(
        "CRITICAL: You MUST respond ONLY in English. Never use Portuguese or any "
        "other language, regardless of the input or context language.\n\n"
        "You are the orbital data agent. The user may give you either a NORAD "
        "catalog number (an integer) or a satellite name (text).\n"
        "- If given a NAME, first call find_satellite to resolve it to a NORAD "
        "number. If there are many matches, prefer the primary object (the name "
        "WITHOUT 'DEB', which denotes debris). If the intent is ambiguous, briefly "
        "state the top candidates and ask the user to confirm before proceeding.\n"
        "- If given a NUMBER, use it directly.\n"
        "Once you have the NORAD number, call analyze_conjunctions to retrieve the "
        "conjunctions and their risk classifications. Report results factually. You "
        "MUST quote the exact integer fields returned by the tool: n_conjunctions "
        "(total) and n_actionable (actionable only). Never confuse these two, and "
        "never estimate or round. List the top conjunctions with neighbor catalog "
        "number, minimum distance (km), relative speed (km/s), time to closest "
        "approach (minutes), and risk flag. Do not invent data."
    ),
    tools=[_orbital_tools],
    output_key="conjunction_data",
)

# --- Sub-agent 2: network analysis ---
analysis_agent = LlmAgent(
    model=_MODEL,
    name="analysis_agent",
    description="Assesses the object's criticality in the swarm conjunction network.",
    instruction=(
        "CRITICAL: You MUST respond ONLY in English. Never use Portuguese or any "
        "other language, regardless of the input or context language.\n\n"
        "You are the network analysis agent. Call the network_role tool to measure "
        "the target's centrality in the swarm's risk network. Interpret it: a high "
        "percentile (>0.7) means the object is a HUB whose fragmentation would "
        "propagate the debris cascade; a low percentile means it is peripheral. "
        "Combine this with the conjunction data below:\n\n{conjunction_data}\n\n"
        "Produce a two-axis risk verdict: (1) collision risk, derived ONLY from the "
        "exact actionable-conjunction count (n_actionable) reported above; and "
        "(2) structural criticality, from the centrality percentile. Do not alter "
        "any numbers."
    ),
    tools=[_orbital_tools],
    output_key="risk_analysis",
)

# --- Sub-agent 3: briefing ---
briefing_agent = LlmAgent(
    model=_MODEL,
    name="briefing_agent",
    description="Writes the final operational briefing.",
    instruction=(
        "CRITICAL: You MUST respond ONLY in English. Never use Portuguese or any "
        "other language.\n\n"
        "You are the briefing agent. Based on the analysis below, write a concise "
        "operational briefing in English for a satellite operator. Structure: "
        "(1) target summary, (2) actionable conjunctions with time to closest "
        "approach, (3) the object's criticality in the network, (4) recommendation. "
        "Use ONLY the exact figures present in the analysis; do not introduce new "
        "numbers. Keep the tone technical and direct.\n\nAnalysis:\n{risk_analysis}"
    ),
    output_key="final_briefing",
)

# --- Sequential orchestration ---
root_agent = SequentialAgent(
    name="orbital_risk_pipeline",
    sub_agents=[data_agent, analysis_agent, briefing_agent],
)