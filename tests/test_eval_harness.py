from fastapi.testclient import TestClient
from app.eval.metrics import eval_harness
from app.main import app
from app.redteam.defense import defense_shield
from app.registry.doc_store import CanonicalDocRegistry

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["documents_indexed"] >= 6
    assert body["chunks_indexed"] >= 6
    assert body["canonical_registry_chunks"] >= 6


def test_live_pdf_or_markdown_ingestion():
    rider = b"Section 20.1 Audit Fee Cap: Annual audit expenses shall not exceed $25,000 in Delaware."
    r = client.post("/api/v1/ingest", files={"file": ("07_Audit_Fee_Rider.md", rider, "text/markdown")})
    assert r.status_code == 200
    assert r.json()["chunks_indexed"] >= 1


def test_clean_ci_gate():
    metrics = eval_harness.compute_metrics()
    assert metrics["context_recall"] >= 0.85
    assert metrics["faithfulness"] >= 0.85
    assert metrics["ci_gate_passed"] is True


def test_redteam_defense_blocks_attacks_via_external_sqlite():
    audit = eval_harness.run_redteam_audit()
    assert audit["defense_block_rate_pct"] >= 95.0
    assert audit["asr_with_defense_pct"] <= 5.0


def test_zero_trust_sqlite_tamper_without_payload_hints(tmp_path):
    reg = CanonicalDocRegistry(str(tmp_path / "test_reg.db"))
    from app.redteam.defense import RedTeamDefenseShield
    from app.ingestion.doc_loader import extract_numeric_invariants

    clean_text = "Provider guarantees 99.95% uptime and $5,000,000 cyber insurance."
    clean_chunk = {
        "chunk_id": "msa::p1::c1",
        "doc_id": "msa",
        "filename": "msa.pdf",
        "page": 1,
        "content": clean_text,
        "numeric_invariants": extract_numeric_invariants(clean_text),
    }
    reg.upsert_chunks([clean_chunk])
    shield = RedTeamDefenseShield(doc_registry=reg)

    # Untrusted tampered payload has NO clean_invariants field attached
    tampered_chunk = {
        "chunk_id": "msa::p1::c1",
        "content": "Provider guarantees 99.95% uptime and $500 cyber insurance.",
    }
    verdict = shield.verify_chunk(tampered_chunk)
    assert verdict["blocked"] is True
    assert verdict["sha_tamper_detected"] is True
    assert verdict["numeric_drift_detected"] is True


def test_citation_spoofing_detection():
    clean_src = "Neither party's aggregate liability shall exceed fees paid in 12 months under Delaware law."
    spoofed_ans = "Liability is capped at $999,000,000 under Nevada law."
    res = defense_shield.verify_citation_attribution(spoofed_ans, clean_src)
    assert res["citation_spoofing_detected"] is True
