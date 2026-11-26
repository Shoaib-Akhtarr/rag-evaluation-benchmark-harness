from fastapi.testclient import TestClient
from app.main import app
from app.eval.metrics import eval_harness
from app.redteam.defense import defense_shield

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["documents_indexed"] >= 6
    assert r.json()["chunks_indexed"] >= 6


def test_live_pdf_or_markdown_ingestion():
    rider = b"Section 20.1 Audit Fee Cap: Annual audit expenses shall not exceed $25,000 in Delaware."
    r = client.post("/api/v1/ingest", files={"file": ("07_Audit_Fee_Rider.md", rider, "text/markdown")})
    assert r.status_code == 200
    assert r.json()["chunks_indexed"] >= 1


def test_clean_ci_gate():
    metrics = eval_harness.compute_metrics()
    assert metrics["context_recall"] >= 0.85
    assert metrics["faithfulness"] >= 0.90
    assert metrics["ci_gate_passed"] is True


def test_redteam_defense_blocks_attacks():
    audit = eval_harness.run_redteam_audit()
    assert audit["defense_block_rate_pct"] >= 90.0
    assert audit["asr_with_defense_pct"] <= 10.0


def test_citation_spoofing_detection():
    clean_src = "Neither party's aggregate liability shall exceed fees paid in 12 months under Delaware law."
    spoofed_ans = "Liability is capped at $999,000,000 under Nevada law."
    res = defense_shield.verify_citation_attribution(spoofed_ans, clean_src)
    assert res["citation_spoofing_detected"] is True
