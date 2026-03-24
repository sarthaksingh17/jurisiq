import json
from openai import AsyncOpenAI
from config.settings import settings

client = AsyncOpenAI(
    api_key=settings.groq_api_key,
    base_url=settings.base_url,
)

async def _debate_round(round_num: int, agent_name: str,
                        own_findings: dict, challenger_findings: dict,
                        contradictions: list, prior_rounds: list) -> dict:
    system = f"""You are the {agent_name} in a legal contract debate, round {round_num} of 3.
Challenge or defend findings based on evidence.
Return JSON: {{"agent": str, "round": int, "argument": str, "confidence_delta": -10 to 10}}"""

    user = f"""
YOUR FINDINGS:
{json.dumps(own_findings, indent=2)}

OPPOSING FINDINGS:
{json.dumps(challenger_findings, indent=2)}

PRE-IDENTIFIED CONTRADICTIONS:
{json.dumps(contradictions, indent=2)}

PRIOR ROUNDS:
{json.dumps(prior_rounds, indent=2)}
"""
    resp = await client.chat.completions.create(
        model=settings.llm_model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0.4,
    )
    return json.loads(resp.choices[0].message.content)


async def run_debate(state: dict) -> list:
    history      = []
    contradictions = state.get("contradictions", [])

    for round_num in range(1, settings.debate_rounds + 1):
        print(f"  Debate round {round_num}/{settings.debate_rounds}...")

        risk_arg = await _debate_round(
            round_num, "Risk Agent",
            state.get("risk_findings", {}),
            state.get("compliance_findings", {}),
            contradictions, history
        )
        history.append(risk_arg)

        comp_arg = await _debate_round(
            round_num, "Compliance Agent",
            state.get("compliance_findings", {}),
            state.get("risk_findings", {}),
            contradictions, history
        )
        history.append(comp_arg)

        fin_arg = await _debate_round(
            round_num, "Finance Agent",
            state.get("finance_findings", {}),
            {"rag_evidence": state.get("rag_findings", [])},
            contradictions, history
        )
        history.append(fin_arg)

    return history