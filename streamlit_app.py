import os
import sys
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.eval.metrics import eval_harness
from app.redteam.defense import defense_shield
from app.redteam.fuzzer import ATTACK_CLASSES, AdversarialRAGFuzzer
from app.registry.doc_store import registry

st.set_page_config(page_title="Adversarial RAG Red-Team & Defense", page_icon="🛡️", layout="wide")
st.title("🛡️ Adversarial RAG Red-Team & Citation-Spoofing Defense")
st.caption("Real PDF Ingestion · SQLite SHA-256 Canonical Registry · PyTorch Injection Shield Trained on Public Benchmarks")

with st.sidebar:
    st.header("Ingest Real Documents")
    uploaded = st.file_uploader("Upload PDF, Markdown, or TXT", type=["pdf", "md", "txt"], accept_multiple_files=True)
    if uploaded:
        for uf in uploaded:
            added = eval_harness.ingest_bytes(uf.name, uf.read())
            st.success(f"Indexed `{uf.name}` ({added} chunks into canonical registry)")
    st.metric("Documents Indexed", len({c["doc_id"] for c in eval_harness.chunks}))
    st.metric("Page Chunks", len(eval_harness.chunks))
    st.metric("SQLite Canonical Chunks", registry.count_chunks())

tab1, tab2, tab3 = st.tabs(["1. Red-Team ASR Audit", "2. Live Attack Simulator", "3. Live LLM Resilience Probe"])

with tab1:
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Run Red-Team Security Audit", type="primary"):
            rt = eval_harness.run_redteam_audit()
            m1, m2, m3 = st.columns(3)
            m1.metric("ASR Without Defense", f"{rt['asr_without_defense_pct']}%")
            m2.metric("ASR With Defense", f"{rt['asr_with_defense_pct']}%")
            m3.metric("Defense Block Rate", f"{rt['defense_block_rate_pct']}%")
            st.caption(f"Unseen Zero-Trust Injections Blocked: {rt['unseen_zero_trust_injections_blocked']}")
            st.dataframe(pd.DataFrame(rt["breakdown_by_attack"]).T, use_container_width=True)
    with col_b:
        if st.button("Run Clean Retrieval CI Gate"):
            m = eval_harness.compute_metrics()
            st.metric("Context Precision", f"{m['context_precision']:.2%}")
            st.metric("Context Recall", f"{m['context_recall']:.2%}")
            st.metric("Faithfulness", f"{m['faithfulness']:.2%}")
            st.success("CI GATE PASSED") if m["ci_gate_passed"] else st.error("CI GATE FAILED")

with tab2:
    atk_choice = st.selectbox("Select Attack Class to Inject:", ATTACK_CLASSES)
    if eval_harness.chunks:
        chunk_idx = st.slider("Target PDF Chunk Index:", 0, len(eval_harness.chunks) - 1, 0)
        if st.button("Mutate Chunk & Test Defense"):
            orig = eval_harness.chunks[chunk_idx]
            mut = AdversarialRAGFuzzer(seed=42).mutate_chunk(orig, atk_choice)
            verdict = defense_shield.verify_chunk(mut)
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Clean PDF Chunk")
                st.info(orig["content"])
            with c2:
                st.subheader(f"Poisoned Chunk ({atk_choice})")
                st.warning(mut["content"])
            if verdict["blocked"]:
                st.success(f"BLOCKED by Defense Shield — {', '.join(verdict['reasons'])}")
            else:
                st.error("BYPASSED Defense Shield")

with tab3:
    if st.button("Probe Live LLM Hijack vs Shield"):
        probe = eval_harness.run_llm_resilience_probe()
        st.json(probe)
