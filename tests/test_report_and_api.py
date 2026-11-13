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
