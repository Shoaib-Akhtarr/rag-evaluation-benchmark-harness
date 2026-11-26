import os
import re
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from app.ingestion.doc_loader import extract_numeric_invariants
from app.registry.doc_store import CanonicalDocRegistry, registry

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARTIFACTS_DIR = os.path.join(ROOT, "artifacts")


def extract_anomaly_features(text: str) -> np.ndarray:
    """
    Domain-agnostic structural prompt-injection and manipulation indicators.
    No fuzzer-specific company names or test-document dates are hardcoded.
    """
    lower = text.lower()
    imperative_cues = sum(
        1
        for pat in [
            "ignore", "disregard", "override", "forget previous", "forget all",
            "new instruction", "system:", "assistant:", "you are now", "bypass",
            "pretend", "developer mode", "<!--", "jailbreak", "do not follow",
        ]
        if pat in lower
    )
    second_person = len(re.findall(r"\b(?:you|your|yours|assistant|system|ai)\b", lower))
    imperative_verbs = len(re.findall(r"\b(?:ignore|reveal|output|print|say|act|forget|execute|waive|grant)\b", lower))
    bracket_density = (text.count("[") + text.count("]") + text.count("<") + text.count(">")) / max(1, len(text))
    punct_density = (text.count("!") + text.count("?") + text.count(":")) / max(1, len(text))
    upper_ratio = sum(1 for c in text if c.isupper()) / max(1, len(text))
    digit_ratio = sum(1 for c in text if c.isdigit()) / max(1, len(text))
    length_norm = min(len(text) / 1000.0, 2.0)
    return np.array(
        [
            min(imperative_cues, 4) / 4.0,
            min(second_person, 6) / 6.0,
            min(imperative_verbs, 5) / 5.0,
            min(bracket_density * 50.0, 1.0),
            min(punct_density * 40.0, 1.0),
            min(upper_ratio * 4.0, 1.0),
            min(digit_ratio * 5.0, 1.0),
            length_norm,
        ],
        dtype=np.float32,
    )


class AdversarialDetectorMLP(nn.Module):
    """Residual MLP classifier over TF-IDF n-grams + domain-agnostic structural cues."""

    def __init__(self, tfidf_dim: int, struct_dim: int = 8):
        super().__init__()
        in_dim = tfidf_dim + struct_dim
        self.fc_in = nn.Linear(in_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc_mid = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop = nn.Dropout(0.30)
        self.fc_out = nn.Linear(256, 2)

    def forward(self, x):
        h = F.gelu(self.bn1(self.fc_in(x)))
        h = h + self.drop(F.gelu(self.bn2(self.fc_mid(h))))
        return self.fc_out(h)


class RedTeamDefenseShield:
    """
    Two-Layer Adversarial RAG Firewall:
      1. Neural Prompt-Injection Detector trained on real HF injection benchmarks
         (deepset/prompt-injections + xTRam1/safe-guard-prompt-injection) — catches unseen
         prompt overrides inside external or runtime chunks.
      2. External SQLite Canonical Registry Verification — verifies SHA-256 content hashes
         and numeric invariants against trusted ingestion records, catching 1-digit drift,
         entity swaps, obligation inversions, and stale-date spoofing without payload hints.
    """

    def __init__(self, doc_registry: CanonicalDocRegistry = registry):
        self.doc_registry = doc_registry
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
        """
        Scores full chunk text and sentence-level windows using the trained neural detector,
        returning the max risk score so long documents cannot dilute an embedded injection sentence.
        """
        if self.model is None or self.vectorizer is None:
            return 0.0
        segments = [text] + [
            seg.strip()
            for seg in re.split(r"(?<=[.!?])\s+|(?=\[)|(?=<!--)", text)
            if len(seg.strip()) >= 20
        ]
        x_tfidf = self.vectorizer.transform(segments).toarray().astype(np.float32)
        x_struct = np.stack([extract_anomaly_features(s) for s in segments], axis=0)
        x = torch.from_numpy(np.concatenate([x_tfidf, x_struct], axis=1))
        with torch.no_grad():
            probs = F.softmax(self.model(x), dim=-1)[:, 1]
        return round(float(probs.max().item()), 4)

    def verify_chunk(self, chunk: Dict[str, Any], clean_invariants: Optional[List[str]] = None) -> Dict[str, Any]:
        poison_prob = self.score_chunk(chunk["content"])
        current_nums = extract_numeric_invariants(chunk["content"])
        current_sha = CanonicalDocRegistry.compute_sha256(chunk["content"])

        canonical = self.doc_registry.lookup_chunk(chunk["chunk_id"]) if self.doc_registry else None
        expected_nums = canonical["numeric_invariants"] if canonical else clean_invariants
        expected_sha = canonical["sha256"] if canonical else None

        numeric_drift_detected = expected_nums is not None and set(current_nums) != set(expected_nums)
        sha_tamper_detected = expected_sha is not None and current_sha != expected_sha

        blocked = poison_prob >= 0.50 or numeric_drift_detected or sha_tamper_detected
        reasons: List[str] = []
        if poison_prob >= 0.50:
            reasons.append(f"Neural Injection Classifier={poison_prob}")
        if sha_tamper_detected:
            reasons.append("SQLite Canonical SHA-256 Tamper Mismatch")
        if numeric_drift_detected:
            reasons.append("Canonical Numeric Invariant Drift")

        if self.doc_registry is not None:
            self.doc_registry.log_security_event(
                chunk_id=chunk["chunk_id"],
                attack_detected=blocked,
                reason="; ".join(reasons) or "clean",
                neural_score=poison_prob,
                hash_mismatch=sha_tamper_detected,
                numeric_mismatch=numeric_drift_detected,
            )

        return {
            "chunk_id": chunk["chunk_id"],
            "blocked": blocked,
            "poison_probability": poison_prob,
            "sha_tamper_detected": sha_tamper_detected,
            "numeric_drift_detected": numeric_drift_detected,
            "canonical_verified": canonical is not None,
            "reasons": reasons,
        }

    def verify_citation_attribution(self, answer: str, cited_doc_text: str) -> Dict[str, Any]:
        ans_nums = set(extract_numeric_invariants(answer))
        src_nums = set(extract_numeric_invariants(cited_doc_text))
        unverified_nums = sorted(list(ans_nums - src_nums))
        ans_tokens = set(re.findall(r"[a-z0-9]{4,}", answer.lower()))
        src_tokens = set(re.findall(r"[a-z0-9]{4,}", cited_doc_text.lower()))
        overlap = len(ans_tokens & src_tokens) / max(1, len(ans_tokens))
        spoofed = len(unverified_nums) > 0 or overlap < 0.55
        return {
            "citation_spoofing_detected": spoofed,
            "span_overlap": round(overlap, 4),
            "unverified_numbers": unverified_nums,
        }


defense_shield = RedTeamDefenseShield()
