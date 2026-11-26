import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.eval.metrics import eval_harness


def main():
    ci_metrics = eval_harness.compute_metrics()
    rt_audit = eval_harness.run_redteam_audit()
    train_metrics = json.load(open(os.path.join(ROOT, "benchmarks", "redteam_training_metrics.json"), "r", encoding="utf-8"))

    ablation = {
        "clean_rag_ci_gate": ci_metrics,
        "pytorch_redteam_detector": {
            "test_accuracy": train_metrics["test_accuracy"],
            "test_f1": train_metrics["test_f1"],
            "test_roc_auc": train_metrics["test_roc_auc"],
        },
        "redteam_asr_before_vs_after": rt_audit,
    }
    with open(os.path.join(ROOT, "benchmarks", "ablation_results.json"), "w", encoding="utf-8") as f:
        json.dump(ablation, f, indent=2)

    md = (
        "# Adversarial RAG Red-Team & Defense Benchmark\n\n"
        f"- **Ingested Legal PDFs:** {ci_metrics['documents_indexed']} PDFs ({ci_metrics['chunks_indexed']} chunks)\n"
        f"- **PyTorch Adversarial Detector:** Test Accuracy = **{train_metrics['test_accuracy']}**, F1 = **{train_metrics['test_f1']}**, ROC-AUC = **{train_metrics['test_roc_auc']}**\n"
        f"- **Attack Success Rate (ASR) Without Defense:** **{rt_audit['asr_without_defense_pct']}%**\n"
        f"- **Attack Success Rate (ASR) With Defense:** **{rt_audit['asr_with_defense_pct']}%** (Block Rate: **{rt_audit['defense_block_rate_pct']}%**)\n"
    )
    with open(os.path.join(ROOT, "benchmarks", "SUMMARY.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(json.dumps(ablation, indent=2))


if __name__ == "__main__":
    main()
