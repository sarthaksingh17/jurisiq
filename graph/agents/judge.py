import json
import hashlib
from openai import AsyncOpenAI
from config.settings import settings

client = AsyncOpenAI(
    api_key=settings.groq_api_key,
    base_url=settings.base_url,
)

async def judge_agent(state: dict) -> dict:
    system = """You are a senior legal judge reviewing a contract analysis.
You have findings from 4 specialist agents, a 3-round debate transcript,
and a judgment signal from InLegalBERT.
Return JSON:
{
  "overall_risk_score": 0-100,
  "risk_breakdown": {
    "legal_risk": 0-100,
    "financial_risk": 0-100,
    "compliance_risk": 0-100
  },
  "top_issues": [{"issue": str, "severity": "high|medium|low", "recommendation": str}],
  "final_recommendation": "approve|approve_with_conditions|reject",
  "confidence": 0-100,
  "reasoning": str
}"""

    context = f"""
INLEGALBERT JUDGMENT SIGNAL:
{json.dumps(state.get('judgment_signal', {}), indent=2)}

CONTRADICTIONS FOUND:
{json.dumps(state.get('contradictions', []), indent=2)}

RISK AGENT:
{json.dumps(state.get('risk_findings', {}), indent=2)}

FINANCE AGENT:
{json.dumps(state.get('finance_findings', {}), indent=2)}

RAG PRECEDENTS:
{json.dumps(state.get('rag_findings', []), indent=2)}

COMPLIANCE AGENT:
{json.dumps(state.get('compliance_findings', {}), indent=2)}

DEBATE TRANSCRIPT:
{json.dumps(state.get('debate_history', []), indent=2)}
"""

    resp = await client.chat.completions.create(
        model=settings.llm_model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": context},
        ],
        temperature=0.1,
    )

    verdict = json.loads(resp.choices[0].message.content)

    audit_payload = json.dumps({
        "job_id":   state["job_id"],
        "doc_hash": state["doc_hash"],
        "verdict":  verdict,
    }, sort_keys=True)
    verdict["audit_hash"] = hashlib.sha256(
        audit_payload.encode()
    ).hexdigest()

    return verdict