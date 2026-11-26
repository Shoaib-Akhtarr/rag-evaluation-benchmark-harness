from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from app.eval.metrics import eval_harness

app = FastAPI(title="Adversarial RAG Red-Team & Evaluation Harness", version="2.0.0")


class QueryReq(BaseModel):
    query: str
    top_k: int = 2


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "documents_indexed": len({c["doc_id"] for c in eval_harness.chunks}),
        "chunks_indexed": len(eval_harness.chunks),
    }


@app.post("/api/v1/ingest")
async def ingest_document(file: UploadFile = File(...)):
    raw = await file.read()
    added = eval_harness.ingest_bytes(file.filename, raw)
    return {"filename": file.filename, "chunks_indexed": added, "total_chunks": len(eval_harness.chunks)}


@app.get("/api/v1/eval/ci-gate")
def run_ci_gate():
    return eval_harness.compute_metrics()


@app.get("/api/v1/redteam/audit")
def run_redteam():
    return eval_harness.run_redteam_audit()
