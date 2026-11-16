"""Streamlit demo: ask a question, watch each step run, read the cited answer as it streams.

Run: `make app`
"""

from __future__ import annotations

import re
import time

import streamlit as st

from filings_copilot.cli import settings_for
from filings_copilot.config import load_ablations
from filings_copilot.eval.runner import best_config
from filings_copilot.generate.answer import Copilot, generate_answer
from filings_copilot.llm import LLM
from filings_copilot.retrieve.factory import build_retriever
from filings_copilot.schemas import RetrievedChunk

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


def md(text: str) -> str:
    """Model text to safe Markdown: "$" is literal (Streamlit would start LaTeX math) and
    runs of citations like [3][4] are bolded as one unit."""
    text = text.replace("$", r"\$")
    return re.sub(r"(?:\[\d+\])+", lambda m: f"**{m.group(0)}**", text)


def use_example(q: str) -> None:
    st.session_state.question = q


def show_sources(area, passages: list[RetrievedChunk], cited: list[int], debug: bool) -> None:
    with area.container():
        st.subheader("Sources")
        for n, p in enumerate(passages, 1):
            c = p.chunk
            mark = "✅ " if n in cited else ""
            with st.expander(
                f"{mark}[{n}] {c.company} · FY{c.fiscal_year} · {c.section_title}",
                expanded=n in cited[:2],
            ):
                if debug:
                    st.caption(" · ".join(f"{k}: #{v}" for k, v in p.ranks.items()))
                st.markdown(md(c.text))


with st.sidebar:
    st.header("Pipeline")
    names = [s["name"] for s in load_ablations()]
    config = st.selectbox("Configuration", names, index=names.index(best_config()))
    st.caption(
        "The default is the configuration that scored best in the evaluation. Switch to see "
        "how each component changes the retrieved passages."
    )
    show_debug = st.toggle("Show retrieval ranks", value=True)

st.title("📑 Filings Copilot")
st.caption(
    "Cited answers from 24 SEC 10-K filings (8 companies, fiscal 2023–2025) · "
    "hybrid search over contextualized chunks · answers abstain when the filings don't say"
)

st.text_input("Ask about the filings", key="question", placeholder="e.g. " + EXAMPLES[0])
st.caption("Or try one:")
cols = st.columns(len(EXAMPLES))
for col, ex in zip(cols, EXAMPLES, strict=True):
    col.button(ex, on_click=use_example, args=(ex,), use_container_width=True)

question = st.session_state.get("question", "").strip()
if question:
    copilot = load(config)
    llm, retriever = copilot.llm, copilot.retriever
    cost_before = llm.total.cost_usd
    t0 = time.perf_counter()

    status = st.status("Working on it…", expanded=True)
    answer_box = st.empty()
    metrics_area = st.empty()
    sources_area = st.empty()

    with status:
        st.write("Reading the question (company, year, sub-questions)…")
        plan = retriever.plan(question)
        t_plan = time.perf_counter() - t0
        parts = f"{len(plan.queries)} sub-questions" if len(plan.queries) > 1 else "one search"
        st.write(f"✓ Planned in {t_plan:.1f} s: {parts}")
        passages = retriever.retrieve(question, plan)
        t_search = time.perf_counter() - t0 - t_plan
        st.write(f"✓ Found {len(passages)} passages in {t_search:.1f} s")
        st.write("Writing the answer…")
    status.update(label="Writing the answer…")
    show_sources(sources_area, passages, [], show_debug)  # readable while the answer streams

    first_words: list[float] = []

    def stream(text: str) -> None:
        if text and not first_words:
            first_words.append(time.perf_counter() - t0)
        answer_box.markdown(md(text) + " ▌")

    ans = generate_answer(question, passages, llm, on_text=stream)
    total = time.perf_counter() - t0
    status.update(label=f"Done in {total:.1f} s", state="complete", expanded=False)

    with answer_box.container():
        if not ans.answerable:
            st.warning("The filings in the corpus don't contain enough to answer this fully.")
        st.markdown(md(ans.answer))
    with metrics_area.container():
        m0, m1, m2, m3 = st.columns(4)
        m0.metric("Passages ready after", f"{t_plan + t_search:.1f} s")
        m1.metric("First words after", f"{first_words[0]:.1f} s" if first_words else "–")
        m2.metric("Full answer", f"{total:.1f} s")
        m3.metric(
            "Cost",
            f"${llm.total.cost_usd - cost_before:.4f}",
            help=f"{len(ans.citations)} of {len(passages)} passages cited",
        )
    show_sources(sources_area, passages, ans.citations, show_debug)
