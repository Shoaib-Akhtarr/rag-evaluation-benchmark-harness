import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.eval.metrics import eval_harness


def main():
    ci_metrics = eval_harness.compute_metrics()
    rt_audit = eval_harness.run_redteam_audit()
    llm_probe = eval_harness.run_llm_resilience_probe()
    train_metrics = json.load(open(os.path.join(ROOT, "benchmarks", "redteam_training_metrics.json"), "r", encoding="utf-8"))

    ablation = {
        "clean_rag_ci_gate": ci_metrics,
        "pytorch_redteam_detector": {
            "datasets": train_metrics.get("datasets", []),
            "test_samples": train_metrics.get("test_samples"),
            "test_accuracy": train_metrics["test_accuracy"],
            "test_precision": train_metrics.get("test_precision"),
            "test_recall": train_metrics.get("test_recall"),
            "test_f1": train_metrics["test_f1"],
            "test_roc_auc": train_metrics["test_roc_auc"],
        },
        "redteam_asr_before_vs_after": rt_audit,
        "live_llm_resilience_probe": llm_probe,
    }
    with open(os.path.join(ROOT, "benchmarks", "ablation_results.json"), "w", encoding="utf-8") as f:
        json.dump(ablation, f, indent=2)

    md = (
        "# Adversarial RAG Red-Team & Defense Benchmark\n\n"
        f"- **Ingested Legal PDFs:** {ci_metrics['documents_indexed']} PDFs ({ci_metrics['chunks_indexed']} chunks in SQLite Canonical Registry)\n"
        f"- **Unseen Test Benchmark (`safe-guard` + `deepset` + `ContractNLI` holdout, N={train_metrics.get('test_samples')}):** "
        f"Accuracy = **{train_metrics['test_accuracy']}**, F1 = **{train_metrics['test_f1']}**, ROC-AUC = **{train_metrics['test_roc_auc']}**\n"
        f"- **Attack Success Rate (ASR) Without Defense:** **{rt_audit['asr_without_defense_pct']}%**\n"
        f"- **Attack Success Rate (ASR) With SQLite + Neural Defense:** **{rt_audit['asr_with_defense_pct']}%** (Block Rate: **{rt_audit['defense_block_rate_pct']}%**)\n"
        f"- **Unseen Zero-Trust Injections Blocked:** {rt_audit['unseen_zero_trust_injections_blocked']}\n"
        f"- **Live LLM Probe (`{llm_probe.get('provider')}/{llm_probe.get('model')}`):** Executed = `{llm_probe.get('executed')}`, Blocked by Shield = `{llm_probe.get('with_defense_blocked')}`\n"
    )
    with open(os.path.join(ROOT, "benchmarks", "SUMMARY.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(json.dumps(ablation, indent=2))


if __name__ == "__main__":
    main()
