# 📈 RAG Evaluation & CI/CD Benchmark Harness (Stanford CUAD)

Automated Ragas/DeepEval-style evaluation suite over the **Stanford CUAD** legal contract corpus. Measures Context Precision, Context Recall, Faithfulness, and Answer Relevance with an automated CI/CD deployment gate.

## Run Tests & Benchmark
```bash
pytest -v tests/
python -m app.evaluate
```
