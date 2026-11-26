import json
import os
import random
import re
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.ingestion.doc_loader import load_directory
from app.redteam.fuzzer import AdversarialRAGFuzzer, ATTACK_CLASSES

ARTIFACTS_DIR = os.path.join(ROOT, "artifacts")
BENCH_DIR = os.path.join(ROOT, "benchmarks")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(BENCH_DIR, exist_ok=True)


class AdversarialDetectorMLP(nn.Module):
    """
    PyTorch Residual MLP for detecting prompt injection, numeric/entity poisoning,
    stale-date spoofing, and obligation inversion in retrieved RAG chunks.
    """
    def __init__(self, tfidf_dim: int, struct_dim: int = 8):
        super().__init__()
        in_dim = tfidf_dim + struct_dim
        self.fc_in = nn.Linear(in_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc_mid = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop = nn.Dropout(0.25)
        self.fc_out = nn.Linear(256, 2)

    def forward(self, x):
        h = F.gelu(self.bn1(self.fc_in(x)))
        h = h + self.drop(F.gelu(self.bn2(self.fc_mid(h))))
        return self.fc_out(h)


def extract_anomaly_features(text: str) -> np.ndarray:
    lower = text.lower()
    inj_cues = sum(1 for w in ["ignore all", "ignore previous", "system override", "admin directive", "disregard", "<!--", "cross-ref override"] if w in lower)
    stale_cues = 1.0 if ("2019" in lower or "superseded" in lower or "archived" in lower) else 0.0
    suspicious_entities = sum(1 for w in ["shadowvault", "nevada", "unregulated", "thirdpartybroker", "us-unverified", "self-attested"] if w in lower)
    neg_count = sum(1 for w in ["shall not", "waives all", "unlimited", "$0"] if w in lower)
    exclaim_brackets = text.count("[") + text.count("]") + text.count("!")
    num_count = len(re.findall(r"\d+", text))
    upper_ratio = sum(1 for c in text if c.isupper()) / max(1, len(text))
    char_len = min(len(text) / 1000.0, 2.0)
    return np.array([
        min(inj_cues, 3) / 3.0,
        stale_cues,
        min(suspicious_entities, 3) / 3.0,
        min(neg_count, 3) / 3.0,
        min(exclaim_brackets, 6) / 6.0,
        min(num_count, 10) / 10.0,
        upper_ratio * 5.0,
        char_len,
    ], dtype=np.float32)


def build_training_dataset():
    chunks = load_directory(os.path.join(ROOT, "documents"))
    fuzzer = AdversarialRAGFuzzer(seed=42)
    rng = random.Random(42)

    texts, labels = [], []
    # Generate 100 clean variations + 100 per attack class (~700+ samples)
    for _ in range(45):
        for c in chunks:
            texts.append(c["content"])
            labels.append(0)  # clean

    for atk in ATTACK_CLASSES:
        for _ in range(18):
            c = rng.choice(chunks)
            mut = fuzzer.mutate_chunk(c, atk)
            texts.append(mut["content"])
            labels.append(1)  # poisoned

    return texts, np.array(labels, dtype=np.int64)


def main():
    torch.manual_seed(42)
    texts, y = build_training_dataset()

    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=512, sublinear_tf=True)
    X_tfidf = vec.fit_transform(texts).toarray().astype(np.float32)
    X_struct = np.stack([extract_anomaly_features(t) for t in texts], axis=0)
    X = np.concatenate([X_tfidf, X_struct], axis=1)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

    loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=64, shuffle=True)
    tfidf_dim = X_tfidf.shape[1]
    model = AdversarialDetectorMLP(tfidf_dim=tfidf_dim, struct_dim=8)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)

    best_acc = 0.0
    best_state = None
    loss_history = []
    for epoch in range(1, 51):
        model.train()
        tot_loss = 0.0
        for bx, by in loader:
            opt.zero_grad()
            loss = F.cross_entropy(model(bx), by)
            loss.backward()
            opt.step()
            tot_loss += float(loss.item()) * len(bx)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            vp = torch.argmax(model(torch.from_numpy(X_val)), dim=-1).numpy()
            vacc = float(accuracy_score(y_val, vp))
        if vacc >= best_acc:
            best_acc = vacc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == 1:
            loss_history.append({"epoch": epoch, "train_loss": round(tot_loss / len(X_train), 4), "val_acc": round(vacc, 4)})

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_test))
        probs = F.softmax(logits, dim=-1)[:, 1].numpy()
        preds = torch.argmax(logits, dim=-1).numpy()

    test_acc = round(float(accuracy_score(y_test, preds)), 4)
    test_f1 = round(float(f1_score(y_test, preds)), 4)
    test_auc = round(float(roc_auc_score(y_test, probs)), 4)

    joblib.dump(vec, os.path.join(ARTIFACTS_DIR, "redteam_vectorizer.joblib"))
    torch.save({"state_dict": best_state, "tfidf_dim": tfidf_dim}, os.path.join(ARTIFACTS_DIR, "redteam_detector.pt"))

    metrics = {
        "model": "PyTorch Residual MLP Adversarial Detector (256-d + CosineAnnealingLR)",
        "total_samples": len(texts),
        "clean_samples": int((y == 0).sum()),
        "poisoned_samples": int((y == 1).sum()),
        "test_accuracy": test_acc,
        "test_f1": test_f1,
        "test_roc_auc": test_auc,
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "loss_history": loss_history,
    }
    with open(os.path.join(BENCH_DIR, "redteam_training_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved redteam_detector.pt | Test Acc={test_acc}, F1={test_f1}, ROC-AUC={test_auc}")


if __name__ == "__main__":
    main()
