from typing import TypedDict, Optional

class GraphState(TypedDict):
    # inputs
    job_id:    str
    doc_name:  str
    doc_text:  str
    doc_hash:  str

    # InLegalBERT enrichment
    segments:          Optional[dict]
    entities:          Optional[dict]
    clause_types:      Optional[list]
    obligations:       Optional[list]
    contradictions:    Optional[list]
    clause_embeddings: Optional[list]
    judgment_signal:   Optional[dict]

    # agent outputs
    risk_findings:        Optional[dict]
    finance_findings:     Optional[dict]
    rag_findings:         Optional[list]
    compliance_findings:  Optional[dict]

    # debate + verdict
    debate_history: list
    debate_round:   int
    final_verdict:  Optional[dict]
    error:          Optional[str]