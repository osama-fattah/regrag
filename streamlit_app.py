"""Live demo of RegRAG. Runs offline (no API keys) on the bundled synthetic SOP corpus.

Deploy: Streamlit Community Cloud -> New app -> this repo, branch main, file streamlit_app.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from regrag.embeddings import HashingEmbedder
from regrag.index import HybridIndex
from regrag.ingest import load_corpus
from regrag.pipeline import RegRAG
from regrag.reasoners import HeuristicReasoner

ROOT = Path(__file__).parent
EXAMPLES = [
    "How many days do we have to submit a serious ICSR?",
    "Are shared user accounts allowed on GxP systems?",
    "Which GAMP category applies to custom bespoke software?",
    "What happens to a biologic product that was frozen below 0 °C?",
    "Can I paste patient-identifiable data into a generative AI tool?",
    "What is the company's parental leave policy?",  # out of scope: should abstain
]

st.set_page_config(page_title="RegRAG demo", page_icon="📑", layout="wide")


@st.cache_resource
def get_rag() -> RegRAG:
    index = HybridIndex.build(load_corpus(ROOT / "data/sample_corpus"), HashingEmbedder())
    return RegRAG(index, HeuristicReasoner(idf=index.bm25.idf))


@st.cache_data
def eval_summary() -> dict | None:
    from regrag.eval.run import evaluate, load_gold

    _, summary = evaluate(get_rag(), load_gold(ROOT / "data/eval/gold.jsonl"))
    return summary


rag = get_rag()

with st.sidebar:
    st.header("About")
    st.markdown(
        "**RegRAG** answers questions about pharma SOPs and policies with citations, and "
        "**refuses** when the approved documents don't support an answer.\n\n"
        "This demo runs the **offline mode** (keyword + hashed-vector retrieval, extractive "
        "answers), so it needs no API keys. The same LangGraph agent runs on AWS Bedrock or "
        "Azure OpenAI with one setting.\n\n"
        "The 8 documents describe a **fictional company** and were written for this demo."
    )
    st.markdown("[Source code on GitHub](https://github.com/osama-fattah/regrag)")
    st.subheader("Evaluation (28-question gold set)")
    s = eval_summary()
    c1, c2 = st.columns(2)
    c1.metric("Retrieval hit@3", f"{s['hit@3']:.0%}")
    c2.metric("Abstention acc.", f"{s['abstention_accuracy']:.0%}")
    c1.metric("Citation precision", f"{s['citation_precision']:.0%}")
    c2.metric("Key-fact recall", f"{s['key_fact_recall']:.0%}")
    st.subheader("Documents")
    for doc in sorted({(c.doc_id, c.title) for c in rag.index.chunks}):
        st.caption(doc[1])

st.title("RegRAG: cited answers from GxP documents")
st.caption("Agentic RAG: hybrid retrieval, grounding checks, abstention, PII redaction, audit trail.")

if "q" not in st.session_state:
    st.session_state.q = EXAMPLES[0]

st.write("Try an example:")
cols = st.columns(3)
for i, ex in enumerate(EXAMPLES):
    if cols[i % 3].button(ex, key=f"ex{i}", use_container_width=True):
        st.session_state.q = ex

question = st.text_input("Your question", key="q")

if question.strip():
    a = rag.ask(question)
    if a.abstained:
        st.warning(f"**Abstained:** {a.abstain_reason}.\n\n{a.answer}")
    else:
        st.success("Grounded answer: every sentence is supported by the cited source.")
        st.markdown(a.answer)
        st.subheader("Sources")
        by_id = {c.chunk_id: c for c in rag.index.chunks}
        for c in a.citations:
            with st.expander(f"[{c.n}] {c.title} › {c.section}"):
                st.write(by_id[c.chunk_id].text)
    if a.redactions:
        st.info(f"Redacted before processing and logging: {', '.join(a.redactions)}")
    if a.injection_suspected:
        st.error("Possible prompt-injection pattern detected and flagged in the audit log.")

    with st.expander("Agent trace (LangGraph steps)"):
        for step in a.trace:
            st.code(step, language=None)
    with st.expander("Audit record (JSON)"):
        st.code(
            json.dumps({k: v for k, v in a.to_dict().items() if k not in ("trace", "answer")}, indent=2),
            language="json",
        )
