import asyncio
import hashlib
import json
import uuid

import fitz
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from graph.graph import compiled_graph
from db.schema import get_connection, init_schema

app = FastAPI(title="JURIS-IQ", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs: dict = {}


@app.on_event("startup")
async def startup():
    init_schema()
    print("JURIS-IQ ready.")


class ContractRequest(BaseModel):
    text: str
    doc_name: str = "contract.txt"


def _build_initial_state(job_id: str, doc_name: str, text: str, doc_hash: str) -> dict:
    return {
        "job_id":              job_id,
        "doc_name":            doc_name,
        "doc_text":            text,
        "doc_hash":            doc_hash,
        "segments":            None,
        "entities":            None,
        "clause_types":        None,
        "obligations":         None,
        "contradictions":      None,
        "clause_embeddings":   None,
        "judgment_signal":     None,
        "risk_findings":       None,
        "finance_findings":    None,
        "rag_findings":        [],
        "compliance_findings": None,
        "debate_history":      [],
        "debate_round":        0,
        "final_verdict":       None,
        "error":               None,
    }


async def _run_graph(job_id: str, state: dict):
    try:
        final = await compiled_graph.ainvoke(state)
        jobs[job_id]["verdict"] = final.get("final_verdict")
        jobs[job_id]["status"]  = "complete"
        jobs[job_id]["events"].append({
            "type":    "verdict",
            "payload": final.get("final_verdict"),
        })
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"]  = str(e)
        print(f"Error for job {job_id}: {e}")


# ── POST /analyze — raw text ──────────────────────────────────────────────────

@app.post("/analyze")
async def analyze_contract(req: ContractRequest):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "running", "verdict": None, "events": []}
    state = _build_initial_state(
        job_id, req.doc_name, req.text,
        hashlib.sha256(req.text.encode()).hexdigest()
    )
    asyncio.create_task(_run_graph(job_id, state))
    return {"job_id": job_id, "status": "running"}


# ── POST /upload — PDF ────────────────────────────────────────────────────────

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDFs accepted")

    raw  = await file.read()
    doc  = fitz.open(stream=raw, filetype="pdf")
    text = "\n".join(page.get_text("text") for page in doc)
    doc.close()

    if not text.strip():
        raise HTTPException(400, "Could not extract text from PDF")

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "running", "verdict": None, "events": []}
    state = _build_initial_state(
        job_id, file.filename, text,
        hashlib.sha256(raw).hexdigest()
    )
    asyncio.create_task(_run_graph(job_id, state))
    return {"job_id": job_id, "status": "running", "doc_name": file.filename}


# ── GET /stream/{job_id} — SSE ────────────────────────────────────────────────

@app.get("/stream/{job_id}")
async def stream(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    async def generator():
        sent = 0
        while True:
            job    = jobs.get(job_id, {})
            events = job.get("events", [])
            while sent < len(events):
                yield f"data: {json.dumps(events[sent])}\n\n"
                sent += 1
            status = job.get("status", "running")
            yield f"data: {json.dumps({'type': 'status', 'status': status})}\n\n"
            if status in ("complete", "error"):
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── GET /verdict/{job_id} ─────────────────────────────────────────────────────

@app.get("/verdict/{job_id}")
async def get_verdict(job_id: str):
    con = get_connection()
    row = con.execute(
        "SELECT * FROM verdicts WHERE job_id = ?", [job_id]
    ).fetchone()
    con.close()
    if not row:
        raise HTTPException(404, "Verdict not found")
    cols = ["job_id", "doc_name", "doc_hash", "risk_score", "finance_score",
            "compliance_score", "overall_score", "debate_log",
            "final_verdict", "audit_hash", "created_at"]
    return dict(zip(cols, row))


# ── GET /health ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)