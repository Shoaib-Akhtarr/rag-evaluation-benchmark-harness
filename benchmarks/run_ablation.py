import json
import os
import sys
import joblib
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.eval.metrics import eval_harness
from training.finetune_legal_retriever import LegalBiEncoderAdapter


def eval_finetuned_adapter():
    vec = joblib.load(os.path.join(ROOT, "artifacts", "cuad_vectorizer.joblib"))
    ckpt = torch.load(os.path.join(ROOT, "artifacts", "cuad_adapter.pt"), map_location="cpu", weights_only=True)
    adapter = LegalBiEncoderAdapter(ckpt["input_dim"])
    adapter.load_state_dict(ckpt["state_dict"])
    adapter.eval()

    emb_data = np.load(os.path.join(ROOT, "artifacts", "cuad_embeddings.npz"))
    doc_ids = emb_data["doc_ids"].tolist()
    doc_emb = torch.from_numpy(emb_data["embeddings"])

    q_texts = [e["query"] for e in eval_harness.eval_dataset]
    q_mat = vec.transform(q_texts).toarray().astype(np.float32)
    q_t = F.normalize(torch.from_numpy(q_mat), p=2, dim=-1)
    with torch.no_grad():
        q_tuned = adapter(q_t)
        sims = q_tuned @ doc_emb.T
        top2 = torch.topk(sims, k=2, dim=-1).indices.tolist()

    prec, rec = [], []
    for item, idxs in zip(eval_harness.eval_dataset, top2):
        r_ids = [doc_ids[i] for i in idxs]
        exp = item["expected_doc_id"]
        prec.append(1.0 if r_ids[0] == exp else (0.5 if exp in r_ids else 0.0))
        rec.append(1.0 if exp in r_ids else 0.0)
    return {
        "context_precision": round(sum(prec) / len(prec), 4),
        "context_recall": round(sum(rec) / len(rec), 4),
    }


def main():
    bm25_metrics = eval_harness.compute_metrics()
    adapter_metrics = eval_finetuned_adapter()
    train_metrics = json.load(open(os.path.join(ROOT, "benchmarks", "training_metrics.json"), "r", encoding="utf-8"))

    ablation = {
        "contracts_loaded": len(eval_harness.contracts),
        "eval_cases": len(eval_harness.eval_dataset),
        "bm25_baseline": bm25_metrics,
        "finetuned_pytorch_biencoder": adapter_metrics,
        "zeroshot_vs_finetuned_p1": {
            "zeroshot_tfidf_p1": train_metrics["zeroshot_precision_at_1"],
            "finetuned_adapter_p1": train_metrics["finetuned_precision_at_1"],
        },
    }
    with open(os.path.join(ROOT, "benchmarks", "ablation_results.json"), "w", encoding="utf-8") as f:
        json.dump(ablation, f, indent=2)

    md = (
        "# Stanford CUAD RAG Evaluation & Ablation Report\n\n"
        f"- **Contracts:** {ablation['contracts_loaded']} | **Eval Cases:** {ablation['eval_cases']}\n\n"
        "| Retriever | Context Precision | Context Recall |\n"
        "|---|---:|---:|\n"
        f"| BM25Okapi | {bm25_metrics['context_precision']} | {bm25_metrics['context_recall']} |\n"
        f"| Fine-Tuned PyTorch Bi-Encoder Adapter | {adapter_metrics['context_precision']} | {adapter_metrics['context_recall']} |\n\n"
        f"- **Faithfulness:** {bm25_metrics['faithfulness']} | **Answer Relevance:** {bm25_metrics['answer_relevance']} | **CI Gate Passed:** {bm25_metrics['ci_gate_passed']}\n"
    )
    with open(os.path.join(ROOT, "benchmarks", "SUMMARY.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(json.dumps(ablation, indent=2))


if __name__ == "__main__":
    main()
