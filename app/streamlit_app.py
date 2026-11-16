"""Streamlit demo: ask a question, see the cited answer and how each passage was ranked.

Run: `make app`
"""

from __future__ import annotations

import re
import time

import streamlit as st

from filings_copilot.cli import settings_for
from filings_copilot.config import load_ablations
from filings_copilot.eval.runner import best_config
from filings_copilot.generate.answer import Copilot
from filings_copilot.llm import LLM
from filings_copilot.retrieve.factory import build_retriever

st.set_page_config(page_title="Filings Copilot", page_icon="📑", layout="wide")

EXAMPLES = [
    "How did NVIDIA's data center revenue change in fiscal 2025, and what drove it?",
    "Compare the cybersecurity risk governance described by JPMorgan and Microsoft in their 2024 10-Ks.",
    "What was Apple's total net sales by geographic segment in fiscal 2024?",
    "What capital expenditure guidance did ExxonMobil give for fiscal 2027?",
]


@st.cache_resource(show_spinner="Loading indexes and models…")
def load(config: str) -> Copilot:
    s = settings_for(config)
    llm = LLM(s.llm, s.resolve(s.llm.cache_path))
    return Copilot(build_retriever(s, llm), llm)


with st.sidebar:
    st.header("Pipeline")
    names = [s["name"] for s in load_ablations()]
    config = st.selectbox("Configuration", names, index=names.index(best_config()))
    st.caption("Switch configurations to see how each component changes the retrieved passages.")
    show_debug = st.toggle("Show retrieval ranks", value=True)

st.title("📑 Filings Copilot")
st.caption("Cited answers from SEC 10-K filings · hybrid search over contextualized chunks")

question = st.text_input("Ask about the filings", placeholder=EXAMPLES[0])
cols = st.columns(len(EXAMPLES))
for col, ex in zip(cols, EXAMPLES, strict=True):
    if col.button(ex[:48] + "…", use_container_width=True):
        question = ex

if question:
    copilot = load(config)
    box = st.empty()
    with st.spinner("Searching the filings and writing the answer…"):
        first_token: list[float] = []

        def show(text: str) -> None:
            # The answer appears as it is written; first words after ~2 s instead of ~8 s.
            if text and not first_token:
                first_token.append(time.perf_counter() - t0)
            box.markdown(re.sub(r"\[(\d+)\]", r"**[\1]**", text) + " ▌")

        t0 = time.perf_counter()
        ans = copilot.ask(question, on_text=show)

    linked = re.sub(r"\[(\d+)\]", r"**[\1]**", ans.answer)
    if not ans.answerable:
        box.empty()
        st.warning("The filings in the corpus don't contain enough to answer this fully.")
        st.markdown(linked)
    else:
        box.markdown(linked)
    m0, m1, m2, m3 = st.columns(4)
    m0.metric("First words after", f"{first_token[0]:.1f} s" if first_token else "–")
    m1.metric("Full answer", f"{ans.latency_ms / 1000:.1f} s")
    m2.metric("Cost", f"${ans.cost_usd:.4f}")
    m3.metric("Passages cited", f"{len(ans.citations)} / {len(ans.passages)}")

    st.subheader("Sources")
    for n, p in enumerate(ans.passages, 1):
        c = p.chunk
        cited = "✅ " if n in ans.citations else ""
        with st.expander(
            f"{cited}[{n}] {c.company} · FY{c.fiscal_year} · {c.section_title}",
            expanded=n in ans.citations[:2],
        ):
            if show_debug:
                st.caption(" · ".join(f"{k}: #{v}" for k, v in p.ranks.items()))
            st.markdown(c.text)
