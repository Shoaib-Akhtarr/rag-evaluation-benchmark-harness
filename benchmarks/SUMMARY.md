# Adversarial RAG Red-Team & Defense Benchmark

- **Ingested Legal PDFs:** 6 PDFs (8 chunks in SQLite Canonical Registry)
- **Unseen Test Benchmark (`safe-guard` + `deepset` + `ContractNLI` holdout, N=2296):** Accuracy = **0.9821**, F1 = **0.9707**, ROC-AUC = **0.9924**
- **Attack Success Rate (ASR) Without Defense:** **100.0%**
- **Attack Success Rate (ASR) With SQLite + Neural Defense:** **0.0%** (Block Rate: **100.0%**)
- **Unseen Zero-Trust Injections Blocked:** 3/3
- **Live LLM Probe (`groq/qwen/qwen3.8-27b`):** Executed = `True`, Blocked by Shield = `True`
