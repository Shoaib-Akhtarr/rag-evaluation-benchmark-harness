from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from app.eval.metrics import eval_harness
from app.llm.client import llm_gateway
from app.registry.doc_store import registry

app = FastAPI(
    title="Adversarial RAG Red-Team & Evaluation Harness",
    version="3.0.0",
    description="Real Prompt-Injection Neural Shield + SQLite SHA-256 Canonical Registry + Live LLM Resilience Probe",
)


class QueryReq(BaseModel):
    query: str
    top_k: int = 2


@app.get("/health")
def health():
    llm_cfg = llm_gateway.resolve_provider()
    return {
        "status": "healthy",
        "documents_indexed": len({c["doc_id"] for c in eval_harness.chunks}),
        "chunks_indexed": len(eval_harness.chunks),
        "canonical_registry_chunks": registry.count_chunks(),
        "llm_provider": llm_cfg["provider"],
        "llm_model": llm_cfg["model"],
    }


@app.post("/api/v1/ingest")
async def ingest_document(file: UploadFile = File(...)):
    raw = await file.read()
    added = eval_harness.ingest_bytes(file.filename, raw)
    return {
        "filename": file.filename,
        "chunks_indexed": added,
        "total_chunks": len(eval_harness.chunks),
        "canonical_registry_chunks": registry.count_chunks(),
    }


@app.get("/api/v1/eval/ci-gate")
def run_ci_gate():
    return eval_harness.compute_metrics()


@app.get("/api/v1/redteam/audit")
def run_redteam():
    return eval_harness.run_redteam_audit()


@app.get("/api/v1/redteam/llm-probe")
def run_llm_probe():
    return eval_harness.run_llm_resilience_probe()
