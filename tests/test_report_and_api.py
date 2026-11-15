import json

import pandas as pd
import pytest
from typer.testing import CliRunner

from filings_copilot.cli import app as cli_app
from filings_copilot.eval.report import END, START, build_report


def test_report_injects_into_readme(tmp_path):
    res = tmp_path / "results"
    (res / "runs").mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "config": "bm25",
                "n": 10,
                "recall@5": 0.5,
                "recall@10": 0.6,
                "mrr": 0.4,
                "ndcg@10": 0.45,
                "hit@1": 0.3,
                "recall@5[table]": 0.2,
                "p50_ms": 5.0,
            },
            {
                "config": "+ rerank",
                "n": 10,
                "recall@5": 0.7,
                "recall@10": 0.8,
                "mrr": 0.6,
                "ndcg@10": 0.65,
                "hit@1": 0.5,
                "recall@5[table]": 0.6,
                "p50_ms": 90.0,
                "Δrecall@5 95% CI": "[+0.1, +0.3]",
            },
        ]
    ).to_csv(res / "retrieval_ablation.csv", index=False)
    rows = [
        {
            "qid": "a",
            "qtype": "single_hop",
            "correctness": 1.0,
            "faithfulness": 1.0,
            "citation_precision": 1.0,
            "false_abstain": 0.0,
        },
        {"qid": "b", "qtype": "unanswerable", "abstain_correct": 1.0},
    ]
    (res / "runs" / "generation.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{START}\nold\n{END}\noutro")

    build_report(res, readme)

    text = readme.read_text()
    assert "old" not in text and "intro" in text and "outro" in text
    assert "| + rerank" in text and "0.700" in text
    assert "unanswerable (correct abstention)" in text
    assert (res / "ablation.png").exists() and (res / "REPORT.md").exists()


def test_cli_help_lists_commands():
    out = CliRunner().invoke(cli_app, ["--help"]).output
    for cmd in [
        "download",
        "parse",
        "chunk",
        "index",
        "gen-gold",
        "review",
        "eval-retrieval",
        "eval-generation",
        "report",
        "ask",
    ]:
        assert cmd in out


def test_api_retrieve(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import filings_copilot.api as api
    from filings_copilot.schemas import Chunk, RetrievedChunk

    class StubRetriever:
        cfg = type("C", (), {"top_k": 2})()
        last_timings = {"total_ms": 1.0}  # noqa: RUF012

        def retrieve(self, q):
            c = Chunk(
                chunk_id="c1",
                doc_id="D",
                ticker="ACME",
                company="Acme",
                fiscal_year=2025,
                item="7",
                section_title="Item 7",
                start=0,
                end=5,
                text="hello",
            )
            return [RetrievedChunk(chunk=c, score=0.9)]

    monkeypatch.setattr(api, "copilot", lambda: type("Co", (), {"retriever": StubRetriever()})())
    resp = TestClient(api.app).post("/retrieve", json={"question": "what?"})
    assert resp.status_code == 200
    assert resp.json()["passages"][0]["ticker"] == "ACME"


def _gold(n_single=40, n_table=12, n_multi=8, n_un=6):
    from filings_copilot.schemas import Evidence, GoldQuestion

    out = []
    for qtype, n in [
        ("single_hop", n_single),
        ("table", n_table),
        ("multi_hop", n_multi),
        ("unanswerable", n_un),
    ]:
        for i in range(n):
            ev = (
                []
                if qtype == "unanswerable"
                else [Evidence(doc_id="D", start=0, end=5, quote="revenue grew strongly")]
            )
            out.append(
                GoldQuestion(
                    qid=f"{qtype}-{i}",
                    question=f"How did revenue change {i}?",
                    answer="a",
                    qtype=qtype,
                    evidence=ev,
                )
            )
    return out


def test_stratified_sample_covers_every_type():
    from filings_copilot.eval.dataset import stratified_sample

    gold = _gold()
    ids = stratified_sample(gold, 20, seed=1)
    types = {q.qtype for q in gold if q.qid in set(ids)}
    assert types == {"single_hop", "table", "multi_hop", "unanswerable"}
    assert len(ids) == len(set(ids)) == 20
    assert ids == stratified_sample(gold, 20, seed=1)  # reproducible


def test_review_is_resumable_and_reported(tmp_path, monkeypatch):
    import filings_copilot.cli as cli
    from filings_copilot.config import load_settings
    from filings_copilot.eval.dataset import load_gold, save_gold
    from filings_copilot.eval.report import eval_set_summary

    gold_path = tmp_path / "eval" / "gold.jsonl"
    save_gold(_gold(), gold_path)
    settings = load_settings(overrides={"eval.gold_path": str(gold_path)})
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    runner = CliRunner()

    # First session: accept, drop, then quit.
    out = runner.invoke(cli_app, ["review", "--sample", "10"], input="a\nd\nq\n").output
    assert "Reviewed 2/" in out
    # Second session resumes where it stopped and edits the next question.
    out = runner.invoke(cli_app, ["review"], input="e\nBetter question?\nBetter answer\nq\n").output
    assert "Reviewed 3/" in out and "1 accepted, 1 edited, 1 dropped" in out

    gold = load_gold(gold_path)
    assert len(gold) == len(_gold()) - 1
    assert sum(q.verified for q in gold) == 2 and any(
        q.question == "Better question?" for q in gold
    )
    summary = "\n".join(eval_set_summary(gold_path))
    assert "65 questions" in summary and "3 questions was reviewed" in summary
    assert "Lexical leakage" in summary


def test_evidence_context_shows_table_headers():
    from filings_copilot.eval.dataset import evidence_context, locate

    text = (
        "Intro paragraph.\n\nThe following table summarizes revenue by market (in millions):\n\n"
        "|  |  |  |\n|---|---|---|\n|  | Jan 26, 2025 | Jan 28, 2024 |\n|  |  |  |\n"
        "| Gaming | $11,350 | $10,447 |\n| Data Center | $115,186 | $47,525 |\n\nNext paragraph."
    )
    start, end = locate(text, "Data Center | $115,186 | $47,525")
    ctx = evidence_context(text, start, end)
    shown = [line for line, _ in ctx]
    assert shown[0].startswith("The following table summarizes revenue")
    assert "|  | Jan 26, 2025 | Jan 28, 2024 |" in shown  # the year header is visible
    assert [line for line, ev in ctx if ev] == ["| Data Center | $115,186 | $47,525 |"]
    assert not any(set(line.strip()) <= {"|", "-", " "} for line in shown)  # layout rows hidden


def test_evidence_context_prose_highlights_quote():
    from filings_copilot.eval.dataset import evidence_context

    text = "A" * 300 + " Revenue grew 12% due to volume. " + "B" * 300
    s = text.index("Revenue")
    ctx = evidence_context(text, s, s + len("Revenue grew 12% due to volume."))
    assert [line for line, ev in ctx if ev] == ["Revenue grew 12% due to volume."]
    assert ctx[0][0].startswith("…") and ctx[-1][0].endswith("…")


def test_generation_table_counts_graded_answers():
    from filings_copilot.eval.report import generation_table

    df = pd.DataFrame(
        [
            {
                "qtype": "multi_hop",
                "correctness": 1.0,
                "faithfulness": 1.0,
                "citation_precision": 1.0,
                "false_abstain": 0.0,
            },
            {
                "qtype": "multi_hop",
                "correctness": None,
                "faithfulness": None,
                "citation_precision": None,
                "false_abstain": 0.0,
            },
            {
                "qtype": "multi_hop",
                "correctness": 0.0,
                "faithfulness": None,
                "citation_precision": None,
                "false_abstain": 1.0,
            },
        ]
    )
    row = generation_table(df).iloc[0]
    assert row["n"] == 3 and row["graded"] == 2 and row["correctness"] == 0.5


def test_best_config_picks_highest_recall(tmp_path):
    from filings_copilot.eval.runner import best_config

    pd.DataFrame(
        [
            {"config": "+ filters", "recall@5": 0.80, "p50_ms": 3000},
            {"config": "full − rerank", "recall@5": 0.83, "p50_ms": 2000},
            {"config": "+ decomposition", "recall@5": 0.83, "p50_ms": 5000},
            {"config": "not a real step", "recall@5": 0.99, "p50_ms": 1},
        ]
    ).to_csv(tmp_path / "retrieval_ablation.csv", index=False)
    assert best_config(tmp_path) == "full − rerank"  # tie on recall -> faster wins
    assert best_config(tmp_path / "missing") == "+ filters"


def test_ablation_reuses_saved_steps(tmp_path, monkeypatch):
    import filings_copilot.eval.runner as runner
    from filings_copilot.config import load_settings
    from filings_copilot.schemas import GoldQuestion

    settings = load_settings(overrides={"eval.results_dir": str(tmp_path)})
    monkeypatch.setattr(runner, "load_settings", lambda overrides=None: settings)
    monkeypatch.setattr(
        runner,
        "load_ablations",
        lambda: [{"name": "a", "overrides": {}}, {"name": "b", "overrides": {}}],
    )
    monkeypatch.setattr(runner, "ensure_chunks", lambda *a: None)
    computed = []

    def fake_run(s, gold):
        computed.append(1)
        return pd.DataFrame(
            [
                {
                    "qid": "q1",
                    "qtype": "single_hop",
                    "latency_ms": 1.0,
                    "recall@5": 1.0,
                    "recall@10": 1.0,
                    "mrr": 1.0,
                    "ndcg@10": 1.0,
                    "hit@1": 1.0,
                }
            ]
        )

    monkeypatch.setattr(runner, "run_retrieval", fake_run)
    gold = [GoldQuestion(qid="q1", question="?", answer="a", qtype="single_hop")]
    runner.run_ablation(gold)
    assert len(computed) == 2
    table = runner.run_ablation(gold)  # second run reuses both saved steps
    assert len(computed) == 2 and list(table["config"]) == ["a", "b"]
    runner.run_ablation(gold, force=["b"])
    assert len(computed) == 3


def test_eval_generation_and_report_end_to_end(tmp_path, monkeypatch):
    """The exact path `make eval` runs after retrieval: generation, judging, report."""
    import sys

    sys.path.insert(0, "tests")
    from conftest import FakeAnthropic, HashEmbedder, OverlapReranker, meta, mini_10k_html

    import filings_copilot.eval.runner as runner
    from filings_copilot.config import ChunkingConfig, load_settings
    from filings_copilot.eval.dataset import locate
    from filings_copilot.eval.report import build_report
    from filings_copilot.index.dense import DenseIndex
    from filings_copilot.ingest.chunking import chunk_document
    from filings_copilot.ingest.parse import parse_filing
    from filings_copilot.llm import LLM
    from filings_copilot.retrieve.pipeline import Retriever
    from filings_copilot.schemas import Evidence, GoldQuestion

    doc = parse_filing(mini_10k_html(), meta())
    chunks = chunk_document(doc, ChunkingConfig(max_tokens=120))
    settings = load_settings(
        overrides={
            "eval.results_dir": str(tmp_path / "results"),
            "llm.cache_path": str(tmp_path / "c.sqlite"),
            "retrieval.top_k": 4,
        }
    )
    emb = HashEmbedder()

    def fake_build(s, llm=None):
        return Retriever(
            s,
            chunks,
            DenseIndex.build(emb, [c.index_text for c in chunks]),
            emb,
            OverlapReranker(),
            llm,
        )

    def reply(req):
        if "answerable" in req["output_config"]["format"]["schema"]["properties"]:
            unans = "dividend" in req["messages"][0]["content"][-1]["text"]
            return {
                "answerable": not unans,
                "answer": "Not disclosed."
                if unans
                else "Data Center revenue was $60,922 million [1].",
                "citations": [] if unans else [1],
            }
        return {
            "correctness": 1,
            "claims": [{"claim": "x", "supported": True}],
            "citation_support": [{"passage": 1, "supports": True}],
        }

    fake = FakeAnthropic(reply)
    monkeypatch.setattr(runner, "build_retriever", fake_build)
    monkeypatch.setattr(
        runner, "LLM", lambda cfg, path=None, **kw: LLM(cfg, path, client=fake, **kw)
    )
    s, e = locate(doc.text, "Data Center | $60,922 | $47,525")
    gold = [
        GoldQuestion(
            qid="t1",
            question="What was Acme data center revenue in fiscal 2025?",
            answer="$60,922 million",
            qtype="table",
            evidence=[Evidence(doc_id=doc.meta.doc_id, start=s, end=e)],
        ),
        GoldQuestion(
            qid="u1",
            question="What dividend will Acme pay in 2030?",
            answer="(not in filings)",
            qtype="unanswerable",
        ),
    ]
    df = runner.run_generation(settings, gold, config_name="full − rerank")
    assert set(df["qid"]) == {"t1", "u1"}
    assert df.set_index("qid").loc["t1", "judge_failed"] == 0.0

    readme = tmp_path / "README.md"
    readme.write_text("<!-- RESULTS:START -->\n<!-- RESULTS:END -->\n")
    body = build_report(tmp_path / "results", readme)
    assert "`full − rerank`" in body and "graded" in body
    assert "unanswerable (correct abstention)" in body
