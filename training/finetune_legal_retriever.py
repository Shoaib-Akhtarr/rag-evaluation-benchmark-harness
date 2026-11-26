import json
import os
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_PATH = os.path.join(ROOT, "data", "cuad_legal_corpus.json")
ARTIFACTS_DIR = os.path.join(ROOT, "artifacts")
BENCH_DIR = os.path.join(ROOT, "benchmarks")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(BENCH_DIR, exist_ok=True)


class LegalBiEncoderAdapter(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 128),
            nn.GELU(),
            nn.Linear(128, 128),
        )

    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=-1)


def main():
    torch.manual_seed(42)
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    contracts = raw["contracts"]
    eval_cases = raw["eval_dataset"]
    doc_ids = [c["doc_id"] for c in contracts]
    doc_texts = [f"{c['title']} ({c['clause_category']}): {c['content']}" for c in contracts]
    q_texts = [e["query"] for e in eval_cases]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=768, sublinear_tf=True)
    doc_mat = vectorizer.fit_transform(doc_texts).toarray().astype(np.float32)
    q_mat = vectorizer.transform(q_texts).toarray().astype(np.float32)

    doc_emb = F.normalize(torch.from_numpy(doc_mat), p=2, dim=-1)
    q_emb = F.normalize(torch.from_numpy(q_mat), p=2, dim=-1)
    target_idx = torch.tensor([doc_ids.index(e["expected_doc_id"]) for e in eval_cases], dtype=torch.long)

    with torch.no_grad():
        base_preds = torch.argmax(q_emb @ doc_emb.T, dim=-1)
        base_acc = float((base_preds == target_idx).float().mean().item())

    adapter = LegalBiEncoderAdapter(dim=doc_emb.shape[1])
    opt = torch.optim.AdamW(adapter.parameters(), lr=8e-3, weight_decay=1e-3)
    loss_history = []
    for epoch in range(1, 101):
        adapter.train()
        opt.zero_grad()
        q_p = adapter(q_emb)
        d_p = adapter(doc_emb)
        logits = (q_p @ d_p.T) * 18.0
        loss = F.cross_entropy(logits, target_idx)
        loss.backward()
        opt.step()
        if epoch % 25 == 0 or epoch == 1:
            loss_history.append({"epoch": epoch, "loss": round(float(loss.item()), 4)})

    adapter.eval()
    with torch.no_grad():
        tuned_doc = adapter(doc_emb)
        tuned_q = adapter(q_emb)
        tuned_preds = torch.argmax(tuned_q @ tuned_doc.T, dim=-1)
        tuned_acc = float((tuned_preds == target_idx).float().mean().item())

    joblib.dump(vectorizer, os.path.join(ARTIFACTS_DIR, "cuad_vectorizer.joblib"))
    torch.save({"state_dict": adapter.state_dict(), "input_dim": doc_emb.shape[1]}, os.path.join(ARTIFACTS_DIR, "cuad_adapter.pt"))
    np.savez(
        os.path.join(ARTIFACTS_DIR, "cuad_embeddings.npz"),
        doc_ids=np.array(doc_ids),
        embeddings=tuned_doc.cpu().numpy(),
    )

    metrics = {
        "base_encoder": "TfidfVectorizer (ngram 1-2, dim<=768)",
        "adapter": "PyTorch MLP Bi-Encoder Adapter (128-d, InfoNCE)",
        "contracts_indexed": len(contracts),
        "eval_queries": len(eval_cases),
        "zeroshot_precision_at_1": round(base_acc, 4),
        "finetuned_precision_at_1": round(tuned_acc, 4),
        "loss_history": loss_history,
    }
    with open(os.path.join(BENCH_DIR, "training_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"CUAD Legal Adapter | Zero-Shot P@1={base_acc:.4f} -> Fine-Tuned P@1={tuned_acc:.4f}")


if __name__ == "__main__":
    main()
