import sys
from app.eval.metrics import eval_harness

if __name__ == "__main__":
    res = eval_harness.compute_metrics()
    print("=== RAG EVALUATION HARNESS (Stanford CUAD) BENCHMARK ===")
    for k, v in res.items():
        print(f"  {k}: {v}")
    if not res["ci_gate_passed"]:
        sys.exit(1)
