import json

import pytest
from conftest import FakeAnthropic

from filings_copilot.eval.dataset import leakage_ratio, locate
from filings_copilot.eval.generation_metrics import judge
from filings_copilot.generate.answer import generate_answer
from filings_copilot.llm import LLM, parse_json
from filings_copilot.schemas import Answer, Chunk, GoldQuestion, RetrievedChunk


def passages(n=3):
    return [
        RetrievedChunk(
            score=1.0,
            chunk=Chunk(
                chunk_id=f"c{i}",
                doc_id="D",
                ticker="ACME",
                company="Acme",
                fiscal_year=2025,
                item="7",
                section_title="Item 7. MD&A",
                start=0,
                end=10,
                text=f"Passage {i} text",
            ),
        )
        for i in range(n)
    ]


def test_parse_json_variants():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! {"a": [1, 2]} trailing') == {"a": [1, 2]}
    with pytest.raises(ValueError):
        parse_json("no json here")


def test_llm_disk_cache_and_cost(settings, tmp_path):
    fake = FakeAnthropic(lambda req: "hello")
    llm = LLM(settings.llm, tmp_path / "cache.sqlite", client=fake)
    assert llm.complete("hi", model="claude-haiku-5-5").text == "hello"
    again = llm.complete("hi", model="claude-haiku-5-5")
    assert again.cached and len(fake.calls) == 1
    assert llm.total.cost_usd == pytest.approx((1000 * 1.0 + 100 * 5.0) / 1e6)


def test_cached_prefix_uses_cache_control(settings):
    fake = FakeAnthropic(lambda req: "ok")
    LLM(settings.llm, None, client=fake).complete("q", cached_prefix="BIG DOC")
    content = fake.calls[0]["messages"][0]["content"]
    assert content[0]["cache_control"] == {"type": "ephemeral"} and content[1]["text"] == "q"


def test_generate_answer_filters_invalid_citations(settings):
    reply = json.dumps(
        {"answerable": True, "answer": "Revenue rose [1][7].", "citations": [1, 7, "x"]}
    )
    llm = LLM(settings.llm, None, client=FakeAnthropic(lambda req: reply))
    ans = generate_answer("What happened to revenue?", passages(), llm)
    assert ans.citations == [1] and ans.answerable
    prompt = llm.client.calls[0]["messages"][0]["content"][0]["text"]
    assert '<passage id="1" company="Acme"' in prompt


def test_judge_scores(settings):
    reply = json.dumps(
        {
            "correctness": 0.5,
            "claims": [{"claim": "a", "supported": True}, {"claim": "b", "supported": False}],
            "citation_support": {"1": True, "2": False},
        }
    )
    llm = LLM(settings.llm, None, client=FakeAnthropic(lambda req: reply))
    gold = GoldQuestion(qid="q", question="?", answer="ref", qtype="single_hop")
    ans = Answer(
        question="?", answer="a. b.", answerable=True, citations=[1, 2], passages=passages()
    )
    row = judge(gold, ans, llm)
    assert row == {
        "false_abstain": 0.0,
        "correctness": 0.5,
        "faithfulness": 0.5,
        "citation_precision": 0.5,
    }


def test_abstention_scored_without_judge(settings):
    llm = LLM(
        settings.llm,
        None,
        client=FakeAnthropic(lambda req: pytest.fail("judge should not be called")),
    )
    gold = GoldQuestion(qid="q", question="?", answer="(not in filings)", qtype="unanswerable")
    ans = Answer(question="?", answer="Not disclosed.", answerable=False)
    assert judge(gold, ans, llm) == {"abstain_correct": 1.0}


def test_locate_ignores_case_whitespace_and_pipes():
    text = "Intro.\n| Data Center | $60,922 | $47,525 |\nMore text here."
    span = locate(text, "data center  $60,922 $47,525")
    assert span is not None
    assert text[span[0] : span[1]].startswith("Data Center")
    assert text[span[0] : span[1]].endswith("47,525")
    assert locate(text, "something that is not there at all") is None


def test_leakage_ratio():
    assert leakage_ratio(
        "What was data center revenue growth?", "Data center revenue growth was strong"
    ) == pytest.approx(0.8)
    assert (
        leakage_ratio("How did AI demand affect sales?", "Data center revenue growth was strong")
        == 0.0
    )
