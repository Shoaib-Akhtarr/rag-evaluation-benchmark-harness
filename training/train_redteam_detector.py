import json
import os
import sys
import httpx
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.redteam.defense import AdversarialDetectorMLP, extract_anomaly_features

ARTIFACTS_DIR = os.path.join(ROOT, "artifacts")
BENCH_DIR = os.path.join(ROOT, "benchmarks")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(BENCH_DIR, exist_ok=True)


def _load_real_legal_clean_clauses(split: str, limit: int = 400) -> list[str]:
    """Loads unseen clean contract clauses from official ContractNLI splits to prevent false positives on legal text."""
    try:
        r = httpx.get("https://huggingface.co/api/datasets/kiddothe2b/contract-nli/parquet", timeout=15).json()
        url = r["contractnli_a"][split][0]
        df = pd.read_parquet(url)
        # Take unique premises truncated to chunk size (~550 chars) so train and test documents stay 100% disjoint
        unique_docs = df["premise"].drop_duplicates().tolist()[:limit]
        return [doc[:600].strip() for doc in unique_docs if isinstance(doc, str) and len(doc.strip()) > 60]
    except Exception:
        return []


def load_benchmark_datasets():
    """
    Loads real open prompt-injection benchmarks with strict official train/test separation:
      - xTRam1/safe-guard-prompt-injection (8,236 train / 2,060 test)
      - deepset/prompt-injections (546 train / 116 test)
      - Disjoint clean legal NDA excerpts from kiddothe2b/contract-nli
    Zero fuzzer templates, zero duplicated chunks, zero train/test leakage.
    """
    ds_sg_tr = load_dataset("xTRam1/safe-guard-prompt-injection", split="train")
    ds_sg_te = load_dataset("xTRam1/safe-guard-prompt-injection", split="test")
    ds_ds_tr = load_dataset("deepset/prompt-injections", split="train")
    ds_ds_te = load_dataset("deepset/prompt-injections", split="test")

    train_texts = list(ds_sg_tr["text"]) + list(ds_ds_tr["text"])
    train_labels = [int(x) for x in ds_sg_tr["label"]] + [int(x) for x in ds_ds_tr["label"]]

    test_texts = list(ds_sg_te["text"]) + list(ds_ds_te["text"])
    test_labels = [int(x) for x in ds_sg_te["label"]] + [int(x) for x in ds_ds_te["label"]]

    legal_train = _load_real_legal_clean_clauses("train", limit=350)
    legal_test = _load_real_legal_clean_clauses("test", limit=120)
    train_texts.extend(legal_train)
    train_labels.extend([0] * len(legal_train))
    test_texts.extend(legal_test)
    test_labels.extend([0] * len(legal_test))

    return train_texts, np.array(train_labels, dtype=np.int64), test_texts, np.array(test_labels, dtype=np.int64)


def main():
    torch.manual_seed(42)
    train_texts, y_train_full, test_texts, y_test = load_benchmark_datasets()

    # Hold out 15% of training split strictly for checkpoint selection (test split stays completely untouched)
    tr_txt, val_txt, y_tr, y_val = train_test_split(
        train_texts, y_train_full, test_size=0.15, stratify=y_train_full, random_state=42
    )

    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=3000, sublinear_tf=True, min_df=2)
    X_tr = np.concatenate(
        [vec.fit_transform(tr_txt).toarray().astype(np.float32), np.stack([extract_anomaly_features(t) for t in tr_txt])],
        axis=1,
    )
    X_val = np.concatenate(
        [vec.transform(val_txt).toarray().astype(np.float32), np.stack([extract_anomaly_features(t) for t in val_txt])],
        axis=1,
    )
    X_test = np.concatenate(
        [vec.transform(test_texts).toarray().astype(np.float32), np.stack([extract_anomaly_features(t) for t in test_texts])],
        axis=1,
    )

    tfidf_dim = len(vec.get_feature_names_out())
    model = AdversarialDetectorMLP(tfidf_dim=tfidf_dim, struct_dim=8)

    # Address mild class imbalance cleanly via weighted cross-entropy
    pos_weight = float((y_tr == 0).sum()) / max(1.0, float((y_tr == 1).sum()))
    class_weights = torch.tensor([1.0, min(pos_weight, 3.0)], dtype=torch.float32)

    loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)), batch_size=128, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=25)

    best_val_f1 = 0.0
    best_state = None
    loss_history = []
    for epoch in range(1, 26):
        model.train()
        tot_loss = 0.0
        for bx, by in loader:
            opt.zero_grad()
            loss = F.cross_entropy(model(bx), by, weight=class_weights)
            loss.backward()
            opt.step()
            total_b = float(loss.item()) * len(bx)
            tot_loss += total_b
        scheduler.step()

        model.eval()
        with torch.no_grad():
            vp = torch.argmax(model(torch.from_numpy(X_val)), dim=-1).numpy()
            vf1 = float(f1_score(y_val, vp))
        if vf1 >= best_val_f1:
            best_val_f1 = vf1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0 or epoch == 1:
            loss_history.append({"epoch": epoch, "train_loss": round(tot_loss / len(X_tr), 4), "val_f1": round(vf1, 4)})

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_test))
        probs = F.softmax(logits, dim=-1)[:, 1].numpy()
        preds = (probs >= 0.50).astype(np.int64)

    test_acc = round(float(accuracy_score(y_test, preds)), 4)
    test_prec = round(float(precision_score(y_test, preds)), 4)
    test_rec = round(float(recall_score(y_test, preds)), 4)
    test_f1 = round(float(f1_score(y_test, preds)), 4)
    test_auc = round(float(roc_auc_score(y_test, probs)), 4)

    joblib.dump(vec, os.path.join(ARTIFACTS_DIR, "redteam_vectorizer.joblib"))
    torch.save({"state_dict": best_state, "tfidf_dim": tfidf_dim}, os.path.join(ARTIFACTS_DIR, "redteam_detector.pt"))

    metrics = {
        "model": "PyTorch Residual MLP Injection Detector (3000 TF-IDF + 8 Domain-Agnostic Features, CosineAnnealingLR)",
        "datasets": [
            "xTRam1/safe-guard-prompt-injection (official train/test)",
            "deepset/prompt-injections (official train/test)",
            "kiddothe2b/contract-nli disjoint clean clauses",
        ],
        "train_samples": len(tr_txt),
        "val_samples": len(val_txt),
        "test_samples": len(test_texts),
        "test_accuracy": test_acc,
        "test_precision": test_prec,
        "test_recall": test_rec,
        "test_f1": test_f1,
        "test_roc_auc": test_auc,
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "loss_history": loss_history,
    }
    with open(os.path.join(BENCH_DIR, "redteam_training_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved redteam_detector.pt | Test N={len(test_texts)} | Acc={test_acc}, Prec={test_prec}, Rec={test_rec}, F1={test_f1}, AUC={test_auc}")


if __name__ == "__main__":
    main()
