from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from functools import lru_cache
from graph.state import GraphState

@lru_cache(maxsize=1)
def load_inlegalbert():
    tokenizer = AutoTokenizer.from_pretrained("law-ai/InLegalBERT")
    model = AutoModel.from_pretrained("law-ai/InLegalBERT")
    model.eval()
    return tokenizer, model

def get_embedding(text: str) -> list:
    tokenizer, model = load_inlegalbert()
    encoded = tokenizer(
        text, return_tensors="pt",
        truncation=True, max_length=512
    )
    with torch.no_grad():
        output = model(**encoded)
    return output.last_hidden_state[:, 0, :].squeeze().tolist()

def extract_entities(text: str) -> dict:
    # simple rule-based extraction for now
    # will upgrade with spaCy NER later
    import re
    amounts = re.findall(r'₹[\d,]+|Rs\.?\s*[\d,]+|\$[\d,]+', text)
    dates   = re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text)
    return {"amounts": amounts, "dates": dates}

def extract_obligations(text: str) -> list:
    # sentences containing obligation keywords
    sentences = text.split('.')
    keywords = ["shall", "must", "will", "obliged", "required",
                "liable", "indemnify", "pay", "deliver", "ensure"]
    return [
        s.strip() for s in sentences
        if any(k in s.lower() for k in keywords) and len(s.strip()) > 20
    ]

def extract_clause_types(text: str) -> list:
    sentences = text.split('.')
    clause_keywords = {
        "liability":    ["liability", "liable", "damages"],
        "indemnity":    ["indemnify", "indemnification", "hold harmless"],
        "termination":  ["terminate", "termination", "cancel"],
        "payment":      ["payment", "pay", "invoice", "fee"],
        "confidential": ["confidential", "non-disclosure", "nda"],
        "ip":           ["intellectual property", "copyright", "patent"],
        "dispute":      ["arbitration", "dispute", "jurisdiction", "court"],
    }
    tagged = []
    for s in sentences:
        s = s.strip()
        if len(s) < 20:
            continue
        for clause_type, keywords in clause_keywords.items():
            if any(k in s.lower() for k in keywords):
                tagged.append({"clause": s, "type": clause_type})
                break
    return tagged

def detect_contradictions(clauses: list) -> list:
    contradictions = []
    contradiction_pairs = [
        ("unlimited liability", "liability cap"),
        ("non-refundable",      "refund"),
        ("exclusive",           "non-exclusive"),
        ("perpetual",           "termination"),
    ]
    full_text = " ".join(c.get("clause","") for c in clauses).lower()
    for term_a, term_b in contradiction_pairs:
        if term_a in full_text and term_b in full_text:
            contradictions.append(
                f"Potential conflict: '{term_a}' vs '{term_b}'"
            )
    return contradictions

def segment_document(text: str) -> dict:
    sections = {
        "full":        text,
        "obligations": [],
        "clauses":     [],
    }
    sections["obligations"] = extract_obligations(text)
    sections["clauses"]     = extract_clause_types(text)
    return sections

async def inlegalbert_node(state: GraphState) -> GraphState:
    print("InLegalBERT preprocessing...")
    text = state["doc_text"]

    # 1. segment document
    state["segments"] = segment_document(text)

    # 2. extract entities
    state["entities"] = extract_entities(text)

    # 3. tag clause types
    state["clause_types"] = extract_clause_types(text)

    # 4. extract obligations
    state["obligations"] = extract_obligations(text)

    # 5. detect contradictions
    state["contradictions"] = detect_contradictions(
        state["clause_types"]
    )

    # 6. compute embeddings for key clauses (for FAISS)
    key_clauses = [
        c["clause"] for c in state["clause_types"][:5]
    ]
    state["clause_embeddings"] = [
        get_embedding(c) for c in key_clauses
    ] if key_clauses else []

    # 7. judgment signal — based on obligation density
    obligation_count = len(state["obligations"])
    contradiction_count = len(state["contradictions"])
    reject_prob = min(
        0.3 + (contradiction_count * 0.2) + (obligation_count * 0.01),
        0.95
    )
    state["judgment_signal"] = {
        "accept": round(1 - reject_prob, 2),
        "reject": round(reject_prob, 2),
    }

    print(f"  segments: {len(state['segments']['clauses'])} clauses found")
    print(f"  obligations: {len(state['obligations'])} found")
    print(f"  contradictions: {len(state['contradictions'])} found")
    print(f"  judgment signal: {state['judgment_signal']}")
    print("InLegalBERT preprocessing complete.")
    return state