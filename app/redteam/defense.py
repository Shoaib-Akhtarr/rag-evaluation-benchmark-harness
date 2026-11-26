import os
import re
from typing import Dict, Any, List
import joblib
import numpy as np
import torch
import torch.nn.functional as F
from app.ingestion.doc_loader import extract_numeric_invariants
from training.train_redteam_detector import AdversarialDetectorMLP, extract_anomaly_features

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARTIFACTS_DIR = os.path.join(ROOT, "artifacts")


class RedTeamDefenseShield:
    """
    Two-Layer Adversarial RAG Defense:
      1. Trained PyTorch Residual MLP Detector (prompt injection, entity swap, stale-date spoof).
      2. Deterministic Numeric Checksum & Span-Attribution Verifier (catches 1-digit drift & citation spoofing).
    """
    def __init__(self):
        vec_path = os.path.join(ARTIFACTS_DIR, "redteam_vectorizer.joblib")
        ckpt_path = os.path.join(ARTIFACTS_DIR, "redteam_detector.pt")
        self.vectorizer = joblib.load(vec_path) if os.path.exists(vec_path) else None
        self.model = None
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            self.model = AdversarialDetectorMLP(tfidf_dim=ckpt["tfidf_dim"], struct_dim=8)
            self.model.load_state_dict(ckpt["state_dict"])
            self.model.eval()

    def score_chunk(self, text: str) -> float:
        if self.model is None or self.vectorizer is None:
            return 0.0
        x_tfidf = self.vectorizer.transform([text]).toarray().astype(np.float32)
        x_struct = extract_anomaly_features(text).reshape(1, -1)
        x = torch.from_numpy(np.concatenate([x_tfidf, x_struct], axis=1))
        with torch.no_grad():
            probs = F.softmax(self.model(x), dim=-1)[0]
        return round(float(probs[1].item()), 4)

    def verify_chunk(self, chunk: Dict[str, Any], clean_invariants: List[str] = None) -> Dict[str, Any]:
        poison_prob = self.score_chunk(chunk["content"])
        current_nums = extract_numeric_invariants(chunk["content"])
        numeric_drift_detected = False
        if clean_invariants is not None and set(current_nums) != set(clean_invariants):
            numeric_drift_detected = True

        blocked = poison_prob >= 0.45 or numeric_drift_detected
        reasons = []
        if poison_prob >= 0.45:
            reasons.append(f"Neural Adversarial Score={poison_prob}")
        if numeric_drift_detected:
            reasons.append("Deterministic Numeric Checksum Mismatch")
        return {
            "chunk_id": chunk["chunk_id"],
            "blocked": blocked,
            "poison_probability": poison_prob,
            "numeric_drift_detected": numeric_drift_detected,
            "reasons": reasons,
        }

    def verify_citation_attribution(self, answer: str, cited_doc_text: str) -> Dict[str, Any]:
        """Checks that numeric invariants & key clauses in the answer actually exist in the cited source span."""
        ans_nums = set(extract_numeric_invariants(answer))
        src_nums = set(extract_numeric_invariants(cited_doc_text))
        unverified_nums = sorted(list(ans_nums - src_nums))
        ans_tokens = set(re.findall(r"[a-z0-9]{4,}", answer.lower()))
        src_tokens = set(re.findall(r"[a-z0-9]{4,}", cited_doc_text.lower()))
        overlap = len(ans_tokens & src_tokens) / max(1, len(ans_tokens))
        spoofed = len(unverified_nums) > 0 or overlap < 0.65
        return {
            "citation_spoofing_detected": spoofed,
            "span_overlap": round(overlap, 4),
            "unverified_numbers": unverified_nums,
        }


defense_shield = RedTeamDefenseShield()
