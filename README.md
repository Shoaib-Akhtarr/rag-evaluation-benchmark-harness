# 🛡️ Adversarial RAG Red-Team & Citation-Spoofing Defense

[![Python 3.11](https://img.shields.io/badge/Python-3.11+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Residual%20MLP-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/Tests-6%20Passed-brightgreen)](tests/)

Production **Pre-LLM Adversarial Firewall, SQLite Canonical Document Registry, and Live LLM Resilience Evaluation Harness** for enterprise legal and compliance RAG pipelines.

## Problem Solved
Standard RAG systems blindly inject retrieved document chunks into LLM prompts — exposing them to **indirect prompt injection**, **1-digit numeric drift** (`$5,000,000` -> `$500`), **entity swaps**, **stale-date spoofing**, **obligation inversions**, and **citation spoofing**.

## System Architecture
1. **PDF Ingestion & SQLite Immutable Registry ([`app/registry/doc_store.py`](app/registry/doc_store.py)):** Computes canonical SHA-256 hashes and numeric invariants at trusted ingestion time — runtime chunk verification never relies on metadata carried inside untrusted payloads.
2. **PyTorch Adversarial Injection Detector ([`training/train_redteam_detector.py`](training/train_redteam_detector.py)):** Residual MLP trained on **real open benchmarks** (`xTRam1/safe-guard-prompt-injection` + `deepset/prompt-injections` + disjoint `kiddothe2b/contract-nli` legal clauses) with domain-agnostic structural cues and strict official train/test separation (zero synthetic leakage).
3. **Live LLM Resilience & Attribution Gate ([`app/eval/metrics.py`](app/eval/metrics.py), [`app/llm/client.py`](app/llm/client.py)):** Measures Clean CI Gate metrics (Precision, Recall, Citation/Numeric Faithfulness) and probes live Groq/OpenAI/Ollama endpoints before and after firewall enforcement with token/latency/cost tracing.

## Verified Benchmark Results (`benchmarks/ablation_results.json`)
- **Unseen Real Test Split (`N = 2,296` across `safe-guard` + `deepset` + `ContractNLI` holdout):**
  - Test Accuracy: **0.9821** | Precision: **0.9869** | Recall: **0.9549** | Macro F1: **0.9707** | ROC-AUC: **0.9924**
- **6-Class Adversarial Fuzzer Suite (24 mutations verified against SQLite + Neural Shield):**
  - Attack Success Rate (Without Defense): **100.0%**
  - Attack Success Rate (With Defense): **0.0%** (Block Rate: **100.0%**)
  - Unseen Zero-Trust Runtime Injections Blocked: **3 / 3**
- **Live LLM CI Gate (`groq/qwen/qwen3.8-27b`):** Context Precision = **1.0**, Context Recall = **1.0**, Faithfulness = **1.0**, Answer Relevance = **0.8056**

## Quickstart
```bash
pip install -r requirements.txt
python training/train_redteam_detector.py
python benchmarks/run_ablation.py
pytest -v tests/
```

Run API + Streamlit Dashboard (or Docker Compose):
```bash
uvicorn app.main:app --port 8002
streamlit run streamlit_app.py
# or: docker compose up --build
```
