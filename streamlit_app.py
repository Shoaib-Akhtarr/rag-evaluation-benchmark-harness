import os
import sys
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.eval.metrics import eval_harness

st.set_page_config(page_title="RAG Evaluation & CI Gatekeeper", page_icon="⚖️", layout="wide")
st.title("⚖️ RAG Evaluation & CI/CD Gatekeeper (Stanford CUAD)")
st.caption("Context Precision · Context Recall · Faithfulness · Answer Relevance")

with st.sidebar:
    st.header("CUAD Legal Corpus")
    st.metric("Legal Contracts Loaded", len(eval_harness.contracts))
    st.metric("Evaluation Cases", len(eval_harness.eval_dataset))

st.subheader("1. Automated CI/CD Quality Gate")
if st.button("Run Full CUAD Evaluation Gate", type="primary"):
    metrics = eval_harness.compute_metrics()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Context Precision", f"{metrics['context_precision']:.2%}")
    c2.metric("Context Recall", f"{metrics['context_recall']:.2%}")
    c3.metric("Faithfulness", f"{metrics['faithfulness']:.2%}")
    c4.metric("Answer Relevance", f"{metrics['answer_relevance']:.2%}")

    if metrics["ci_gate_passed"]:
        st.success("CI GATE PASSED — Faithfulness >= 90% and Context Recall >= 85%")
    else:
        st.error("CI GATE FAILED — Deployment Blocked")

st.divider()
st.subheader("2. Interactive Legal Clause Retrieval")
presets = [e["query"] for e in eval_harness.eval_dataset]
chosen = st.selectbox("Sample CUAD Legal Query:", presets)
query = st.text_input("Query legal clauses:", value=chosen)
top_k = st.slider("Top-K Clauses:", 1, 5, 2)

if st.button("Retrieve & Ground Legal Answer"):
    docs = eval_harness.retrieve_top_k(query, k=top_k)
    answer = eval_harness.generate_grounded_answer(docs)
    st.subheader("Verbatim-Grounded Answer")
    st.info(answer)
    st.subheader("Retrieved Clauses")
    for d in docs:
        with st.expander(f"[{d['doc_id']}] {d['title']} ({d['clause_category']})"):
            st.write(d["content"])
