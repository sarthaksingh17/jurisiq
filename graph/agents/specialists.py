import json
from openai import AsyncOpenAI
from config.settings import settings
from services.retrieval import similarity_search

client = AsyncOpenAI(
    api_key=settings.groq_api_key,
    base_url=settings.base_url,
)

async def _llm(system: str, user: str) -> str:
    resp = await client.chat.completions.create(
        model=settings.llm_model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content


async def risk_agent(state: dict) -> dict:
    system = """You are a legal risk analyst. Analyse the contract for dangerous clauses.
Use the pre-extracted clause types and obligations provided.
Return JSON: {
  "high_risk_clauses": [{"clause": str, "reason": str, "severity": 1-10}],
  "risk_score": 0-100,
  "summary": str
}"""
    context = f"""
CONTRACT CLAUSES (pre-extracted by InLegalBERT):
{json.dumps(state.get('clause_types', [])[:20], indent=2)}

OBLIGATIONS FOUND:
{json.dumps(state.get('obligations', [])[:10], indent=2)}

CONTRADICTIONS FOUND:
{json.dumps(state.get('contradictions', []), indent=2)}

RAW CONTRACT (first 3000 chars):
{state['doc_text'][:3000]}
"""
    result = await _llm(system, context)
    return json.loads(result)


async def finance_agent(state: dict) -> dict:
    system = """You are a financial analyst reviewing a contract.
Use the pre-extracted obligations and entities provided.
Return JSON: {
  "financial_flags": [{"item": str, "concern": str, "severity": 1-10}],
  "finance_score": 0-100,
  "summary": str
}"""
    context = f"""
OBLIGATIONS (pre-extracted):
{json.dumps(state.get('obligations', [])[:15], indent=2)}

ENTITIES FOUND (amounts, dates):
{json.dumps(state.get('entities', {}), indent=2)}

RAW CONTRACT (first 3000 chars):
{state['doc_text'][:3000]}
"""
    result = await _llm(system, context)
    return json.loads(result)


async def rag_agent(state: dict) -> list:
    system = """Extract the 3 most legally significant clauses for precedent search.
Return JSON: {"clauses": ["clause1", "clause2", "clause3"]}"""

    context = f"""
CLAUSE TYPES (pre-extracted):
{json.dumps(state.get('clause_types', [])[:10], indent=2)}

CONTRACT (first 2000 chars):
{state['doc_text'][:2000]}
"""
    result = await _llm(system, context)
    clauses = json.loads(result).get("clauses", [])

    all_precedents = []
    seen = set()
    for clause in clauses:
        hits = similarity_search(clause, top_k=3)
        for hit in hits:
            key = hit["chunk_text"][:80]
            if key not in seen:
                seen.add(key)
                all_precedents.append(hit)

    return all_precedents[:5]


async def compliance_agent(state: dict) -> dict:
    system = """You are a compliance officer checking a contract against regulations.
Use the pre-identified statutes and clause types provided.
Return JSON: {
  "violations": [{"regulation": str, "clause": str, "severity": 1-10}],
  "compliance_score": 0-100,
  "summary": str
}"""
    context = f"""
APPLICABLE STATUTES (pre-identified by InLegalBERT):
{json.dumps(state.get('statutes', []), indent=2)}

CLAUSE TYPES:
{json.dumps(state.get('clause_types', [])[:20], indent=2)}

CONTRACT (first 3000 chars):
{state['doc_text'][:3000]}
"""
    result = await _llm(system, context)
    return json.loads(result)