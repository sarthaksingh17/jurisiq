# JURIS-IQ — Multi-Agent Legal Contract Intelligence  
<p align="center">
  <em>your AI Harvey Specter for contracts ⚖️</em>
</p>

A production-grade AI system that analyzes legal contracts using 5 specialized agents, adversarial debate, and legal-domain NLP — built with LangGraph, FAISS, InLegalBERT, and LLaMA 3.


<img width="2540" height="1252" alt="image" src="https://github.com/user-attachments/assets/9e673961-4400-4aac-ad3a-ee2e8227ace1" />


## What it does
Upload a contract PDF or paste contract text. JURIS-IQ deploys 5 AI agents that analyze it from every angle — legal risk, financial exposure, regulatory compliance, and precedent search — before a Judge agent synthesizes a final verdict with a confidence score.
### The output:
+ Overall risk score (0–100)
+ Legal, financial, and compliance breakdowns
+ Top issues with recommendations
+ Final recommendation: APPROVE / APPROVE WITH CONDITIONS / REJECT
+ Full 9-argument adversarial debate transcript
+ SHA-256 tamper-proof audit hash

## Architecture

<img width="686" height="651" alt="image" src="https://github.com/user-attachments/assets/4f2a8b3b-934f-443f-b95b-d23eb8e35073" />

## Tech Stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph (typed state machine) |
| LLM | LLaMA 3.3 70B via Groq API |
| Legal NLP | InLegalBERT (trained on 5.4M Indian court docs) |
| Vector search | FAISS (IndexFlatIP) |
| Metadata store | DuckDB (embedded, no server) |
| Legal corpus | CUAD dataset — 510 real commercial contracts |
| API | FastAPI + SSE streaming |
| Frontend | Streamlit |
| Async | asyncio.gather() for parallel agent execution |

## Project Structure
<img width="740" height="678" alt="image" src="https://github.com/user-attachments/assets/142f58dc-cca3-4a2e-8403-6027819d0214" />

## Quickstart
+ git clone https://github.com/yourusername/jurisiq
+ cd jurisiq
+ uv venv --python 3.11
+ .venv\Scripts\activate   # Windows
+ source .venv/bin/activate  # Mac/Linux
+ uv pip install -r requirements.txt


## Build the knowledge base (once)
python -m ingestion.ingest --limit 50

## The Adversarial Debate
After 4 agents produce findings, the debate engine runs 3 rounds:

+ Round 1 — each agent presents its top finding
+ Round 2 — agents challenge each other's conclusions
+ Round 3 — agents defend or revise with new evidence

## Dataset
#### CUAD (Contract Understanding Atticus Dataset)

+ 510 real commercial contracts from SEC EDGAR(used 50 in project)
+ 13,000+ expert-labeled clauses across 41 categories
+ Source: theatticusproject/cuad

#### InLegalBERT

+ Pre-trained on 5.4 million Indian court documents (1950–2019)
+ Used for legal-aware embeddings and document pre-processing
+ Source: law-ai/InLegalBERT

