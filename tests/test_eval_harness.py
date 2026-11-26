from fastapi.testclient import TestClient
from app.main import app
from app.eval.metrics import eval_harness

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["contracts_loaded"] >= 22

def test_cuad_ci_gate():
    metrics = eval_harness.compute_metrics()
    assert metrics["context_precision"] >= 0.90
    assert metrics["context_recall"] == 1.0
    assert metrics["faithfulness"] >= 0.95
    assert metrics["answer_relevance"] >= 0.90
    assert metrics["ci_gate_passed"] is True
