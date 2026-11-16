"""Runs the Streamlit script headless with stand-ins for the indexes and the API."""

import json

from conftest import FakeAnthropic
from streamlit.testing.v1 import AppTest
from test_llm_and_generation import passages

from filings_copilot.retrieve.query import QueryPlan


class FakeRetriever:
    def plan(self, question):
        return QueryPlan(original=question, tickers=["ACME"], fiscal_years=[2025])

    def retrieve(self, question, plan=None):
        return passages()


def _patch(monkeypatch, settings, reply):
    import streamlit as st

    import filings_copilot.eval.runner as runner
    import filings_copilot.llm as llm_mod
    import filings_copilot.retrieve.factory as factory

    st.cache_resource.clear()  # the app caches its Copilot across runs in one process
    fake = FakeAnthropic(lambda req: reply)
    real_llm = llm_mod.LLM
    monkeypatch.setattr(
        llm_mod, "LLM", lambda cfg, path=None, **kw: real_llm(cfg, None, client=fake)
    )
    monkeypatch.setattr(factory, "build_retriever", lambda s, llm: FakeRetriever())
    monkeypatch.setattr(runner, "best_config", lambda *a, **k: "+ filters")
    return fake


def test_example_button_fills_the_question_and_renders_safe_markdown(monkeypatch, settings):
    reply = json.dumps(
        {"answerable": True, "answer": "Sales were $391,035 million [3][4].", "citations": [3, 4]}
    )
    _patch(monkeypatch, settings, reply)
    at = AppTest.from_file("../app/streamlit_app.py", default_timeout=30).run()
    assert not at.exception
    at.button[2].click().run()  # third example
    assert not at.exception, at.exception
    assert at.text_input(key="question").value.startswith("What was Apple's")
    answer = next(m.value for m in at.markdown if "391,035" in m.value)
    assert r"\$391,035" in answer and "**[3][4]**" in answer  # no LaTeX, one bold run
    labels = [m.label for m in at.metric]
    assert labels == ["Passages ready after", "First words after", "Full answer", "Cost"]


def test_unanswerable_shows_warning(monkeypatch, settings):
    _patch(
        monkeypatch,
        settings,
        json.dumps({"answerable": False, "answer": "Not disclosed.", "citations": []}),
    )
    at = AppTest.from_file("../app/streamlit_app.py", default_timeout=30)
    at.session_state["question"] = "What is ACME's 2030 guidance?"
    at.run()
    assert not at.exception, at.exception
    alerts = [a.value for a in at.warning]  # st.warning inside a placeholder container
    assert any("don't contain enough" in a for a in alerts), alerts
