from langgraph.graph import StateGraph, END
import asyncio
import hashlib

from graph.state import GraphState
from graph.nodes.inlegalbert_node import inlegalbert_node
from graph.agents.specialists import risk_agent, finance_agent, rag_agent, compliance_agent
from graph.agents.judge import judge_agent
from graph.nodes.debate import run_debate
from db.schema import get_connection


# ── Node: ingest ─────────────────────────────────────────────────────────────

async def ingest_node(state: GraphState) -> GraphState:
    state["doc_hash"]       = hashlib.sha256(state["doc_text"].encode()).hexdigest()
    state["debate_history"] = []
    state["debate_round"]   = 0
    return state


# ── Node: parallel agents (1–4) ──────────────────────────────────────────────

async def parallel_agents_node(state: GraphState) -> GraphState:
    print("Running 4 agents in parallel...")
    results = await asyncio.gather(
        risk_agent(state),
        finance_agent(state),
        rag_agent(state),
        compliance_agent(state),
        return_exceptions=True,
    )
    state["risk_findings"]       = results[0] if not isinstance(results[0], Exception) else None
    state["finance_findings"]    = results[1] if not isinstance(results[1], Exception) else None
    state["rag_findings"]        = results[2] if not isinstance(results[2], Exception) else []
    state["compliance_findings"] = results[3] if not isinstance(results[3], Exception) else None
    print("All 4 agents complete.")
    return state


# ── Node: debate ──────────────────────────────────────────────────────────────

async def debate_node(state: GraphState) -> GraphState:
    print("Starting adversarial debate...")
    state["debate_history"] = await run_debate(state)
    print(f"Debate complete — {len(state['debate_history'])} arguments logged.")
    return state


# ── Node: judge (Agent 5) ─────────────────────────────────────────────────────

async def judge_node(state: GraphState) -> GraphState:
    print("Judge agent synthesizing verdict...")
    state["final_verdict"] = await judge_agent(state)
    print("Verdict ready.")
    return state


# ── Node: store verdict in DuckDB ─────────────────────────────────────────────

async def store_node(state: GraphState) -> GraphState:
    import json
    verdict = state.get("final_verdict", {})
    con = get_connection()
    con.execute("""
        INSERT OR REPLACE INTO verdicts
            (job_id, doc_name, doc_hash, risk_score, finance_score,
             compliance_score, overall_score,
             debate_log, final_verdict, audit_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        state["job_id"],
        state["doc_name"],
        state["doc_hash"],
        verdict.get("risk_breakdown", {}).get("legal_risk"),
        verdict.get("risk_breakdown", {}).get("financial_risk"),
        verdict.get("risk_breakdown", {}).get("compliance_risk"),
        verdict.get("overall_risk_score"),
        json.dumps(state.get("debate_history", [])),
        verdict.get("reasoning"),
        verdict.get("audit_hash"),
    ])
    con.close()
    print(f"Verdict stored for job {state['job_id']}")
    return state


# ── Build the graph ───────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(GraphState)

    g.add_node("ingest",          ingest_node)
    g.add_node("inlegalbert",     inlegalbert_node)   
    g.add_node("parallel_agents", parallel_agents_node)
    g.add_node("debate",          debate_node)
    g.add_node("judge",           judge_node)
    g.add_node("store",           store_node)

    g.set_entry_point("ingest")
    g.add_edge("ingest",          "inlegalbert")       #  updated
    g.add_edge("inlegalbert",     "parallel_agents")   #  updated
    g.add_edge("parallel_agents", "debate")
    g.add_edge("debate",          "judge")
    g.add_edge("judge",           "store")
    g.add_edge("store",           END)

    return g.compile()


compiled_graph = build_graph()