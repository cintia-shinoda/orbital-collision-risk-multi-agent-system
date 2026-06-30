# Orbital Collision-Risk Multi-Agent System

A multi-agent system that assesses satellite collision risk in real time. Given a NORAD catalog number, it screens the object against a live debris catalog, classifies each close approach with a model trained on real European Space Agency data, measures the object's structural criticality in the conjunction network, and produces an operational briefing — visualized on an interactive 3D globe.

**Capstone track:** Freestyle

**Built with:** Google ADK 2.x, MCP, Gemini, Skyfield, LightGBM, NetworkX, CesiumJS



https://github.com/user-attachments/assets/62da2d20-e5ff-4542-a890-c1e4864db8fb


---

## The problem

Low Earth orbit is crowded and getting worse. The 2009 collision between the active satellite Iridium 33 and the defunct Cosmos 2251 created thousands of debris fragments still circling today. Each fragment is a potential trigger for further collisions — a chain reaction known as the **Kessler syndrome**, which could render entire orbital bands unusable and threaten GPS, communications, and weather infrastructure.

Satellite operators need to answer a recurring, time-critical question: **given my object and the current state of the catalog, what is about to come dangerously close, and how critical is my object to the wider debris network?**

## Why agents

The question is not a single computation — it is a small investigation that chains specialized steps: fetch live orbital data, propagate trajectories, screen for close approaches, classify their risk, analyze network position, and communicate the result to a human. A multi-agent system mirrors that workflow: each agent owns one responsibility, and a coordinator runs them in a reliable, inspectable sequence. The natural-language briefing at the end is exactly what an operator needs — not a table dump, but a decision.

## Solution overview

The system runs a fixed three-stage pipeline orchestrated by an ADK `SequentialAgent`:

1. **Data agent** — calls an MCP tool to fetch orbital elements, propagate them, screen one-vs-all conjunctions, and classify each approach as actionable or not.
2. **Analysis agent** — calls an MCP tool to measure the target's centrality in the swarm's conjunction network (hub vs. peripheral), then fuses collision risk with structural criticality.
3. **Briefing agent** — synthesizes a concise operational briefing in natural language.

The two analytical engines are exposed to the agents through a **custom MCP server**, keeping the orbital/ML logic decoupled from the agent layer.

## Architecture




```
NORAD ID
   │
   ▼
SequentialAgent (coordinator)
   ├── data_agent      ──► MCP: analyze_conjunctions  (Skyfield + KDTree + LightGBM)
   ├── analysis_agent  ──► MCP: network_role          (NetworkX centrality)
   └── briefing_agent  ──► natural-language briefing (Gemini)
   │
   ▼
Operational briefing  +  3D CesiumJS visualization
```

### Two-axis risk model

A single number does not capture orbital risk. The system reports two complementary axes:

- **Collision risk** — per-conjunction, from the ML classifier (distance, relative speed, time to closest approach).
- **Structural criticality** — from network centrality: an object with high centrality is a *hub* whose fragmentation would propagate the debris cascade.

An object can be high-risk but peripheral, or low-risk but a structural hub. Both matter.

## The risk classifier

Trained on the **ESA Collision Avoidance Challenge** dataset (real Conjunction Data Messages, 2015–2019). The target label is the actionable-risk flag of the final CDM per event (ESA's operational threshold). Only features that the project's own orbital pipeline can reproduce were used — minimum distance, relative speed, and time to closest approach — so the model is deployable on live screening output, not just on the benchmark.

| Metric | Value |
|---|---|
| Training events | 13,154 |
| Actionable events | 3,935 |
| Accuracy | 0.82 |
| ROC-AUC | 0.86 |
| F1 (actionable) | 0.71 |
| Recall (actionable) | 0.74 |
| Precision (actionable) | 0.68 |

The ROC-AUC of 0.86 shows the model ranks risk reliably; the operating threshold can be shifted toward recall to prioritize never missing a dangerous approach. Run `python scripts/risk/evaluate.py` to reproduce these numbers.

## Course concepts demonstrated

- **Multi-agent system (ADK):** three specialized sub-agents under a `SequentialAgent` coordinator.
- **MCP server:** a custom FastMCP server exposes the orbital and risk tools.
- **Security:** all credentials live in a git-ignored `.env`; no keys in code.
- **Deployability:** reproducible local run; optional Cloud Run deployment documented below.

## Tech stack

| Layer | Tool |
|---|---|
| Agent framework | Google ADK 2.x |
| LLM | Gemini (Google AI Studio) |
| Tool protocol | MCP (FastMCP) |
| Orbital data | CelesTrak GP/JSON via httpx |
| Propagation | Skyfield (SGP4) |
| Conjunction screening | SciPy KDTree |
| Network analysis | NetworkX |
| Risk classifier | LightGBM (trained on ESA CDM) |
| Data | Polars, DuckDB/Parquet |
| Visualization | CesiumJS (CZML) |

## Setup

> Requires Python 3.11 and a Gemini API key (Google AI Studio). 
> On macOS, LightGBM needs OpenMP: `brew install libomp`.


# 1. Environment
```bash
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

# 2. Credentials (never committed)
```bash
echo "GOOGLE_API_KEY=your_key_here" > orbital_agent/.env
```

# 3. Build the data layer (debris catalog + conjunction network)
```bash
python scripts/orbital/fetch_tle.py
python scripts/orbital/conjunctions.py
python scripts/orbital/graph.py
```

# 4. (Optional) reproduce the classifier and its metrics
```bash
python scripts/risk/evaluate.py
```

# 5. Run the agent
```bash
adk web orbital_agent
```

Then ask, for example: *"Analyze the collision risk for object 22675."*

### 3D visualization

```bash
python viz/generate_czml.py
cd viz && python -m http.server 8080
```

open: http://localhost:8080/index.html


## Project structure

```bash

orbital-collision-risk-multi-agent-system/
├── orbital_agent/          # ADK multi-agent system + MCP server
│   ├── agent.py
│   └── mcp_server.py
│
├── scripts/
│   ├── orbital/            # fetch_tle, propagate, conjunctions
│   └── risk/               # graph, classifier, evaluate
│
└── viz/                    # generate_czml + CesiumJS page + CZML
```

## Limitations and next steps

- Conjunction screening uses a coarse time grid; precise time-of-closest-approach would need local temporal refinement around each candidate.
- The classifier uses only the three features reproducible from public TLE data; the ESA winners used orbit-determination covariance, which TLEs do not provide. This is a deliberate trade-off favoring deployability over peak accuracy.
- Risk and centrality are computed live per query; caching would reduce latency for interactive use.

## Data and credits

- Orbital elements: [CelesTrak](https://celestrak.org) (GP/JSON).
- Risk labels: ESA Collision Avoidance Challenge dataset (Uriot et al.), CC-BY-4.0, [Zenodo 4463683](https://zenodo.org/records/4463683).
