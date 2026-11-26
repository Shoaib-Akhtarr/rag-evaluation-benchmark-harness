from fastapi import FastAPI
from app.eval.metrics import eval_harness

app = FastAPI(title="RAG Evaluation & CI/CD Harness (Stanford CUAD)", version="1.0.0")

@app.get("/health")
def health():
    return {"status": "healthy", "contracts_loaded": len(eval_harness.contracts)}

@app.get("/api/v1/eval/run")
def run_eval():
    return eval_harness.compute_metrics()
