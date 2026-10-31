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
