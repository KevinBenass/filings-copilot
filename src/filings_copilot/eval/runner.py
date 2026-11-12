"""Experiment runners: retrieval ablation ladder and end-to-end generation eval."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import track

from filings_copilot.config import Settings, load_ablations, load_settings
from filings_copilot.eval.generation_metrics import judge
from filings_copilot.eval.retrieval_metrics import score_question
from filings_copilot.generate.answer import generate_answer
from filings_copilot.ingest.store import build_chunks, chunks_path
from filings_copilot.llm import LLM
from filings_copilot.retrieve.factory import build_retriever
from filings_copilot.schemas import GoldQuestion

console = Console()


def paired_bootstrap_ci(
    a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """95% CI of mean(b - a) over questions, resampling questions with replacement."""
    rng = np.random.default_rng(seed)
    diff = b - a
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def ensure_chunks(settings: Settings, allow_llm_cost: bool) -> None:
    if chunks_path(settings).exists():
        return
    if settings.chunking.context == "llm":
        _, est = build_chunks(settings, dry_run=True)
        if not allow_llm_cost:
            raise RuntimeError(
                f"Chunk variant {settings.chunking.variant} needs LLM contextualization "
                f"(estimated ${est:.2f}). Re-run with --allow-llm-cost."
            )
        console.print(f"Contextualizing chunks, estimated cost ${est:.2f}")
    build_chunks(settings)


def run_retrieval(settings: Settings, gold: list[GoldQuestion]) -> pd.DataFrame:
    retriever = build_retriever(settings)
    rows = []
    for q in track([g for g in gold if g.qtype != "unanswerable"], description="Retrieving"):
        t0 = time.perf_counter()
        hits = retriever.retrieve(q.question)
        latency = (time.perf_counter() - t0) * 1e3
        row: dict[str, Any] = {"qid": q.qid, "qtype": q.qtype, "latency_ms": latency}
        row.update(score_question([h.chunk for h in hits], q.evidence, settings.eval.ks))
        row["top_chunks"] = [h.chunk.chunk_id for h in hits]
        rows.append(row)
    return pd.DataFrame(rows)


def run_ablation(
    gold: list[GoldQuestion], allow_llm_cost: bool = False, only: list[str] | None = None
) -> pd.DataFrame:
    base = load_settings()
    out_dir = base.resolve(base.eval.results_dir)
    (out_dir / "runs").mkdir(parents=True, exist_ok=True)
    summary = []
    prev: pd.DataFrame | None = None
    for step in load_ablations():
        if only and step["name"] not in only:
            continue
        overrides = {**step["overrides"], "retrieval.top_k": max(base.eval.ks)}
        settings = load_settings(overrides=overrides)
        console.rule(step["name"])
        ensure_chunks(settings, allow_llm_cost)
        df = run_retrieval(settings, gold)
        df.to_json(
            out_dir / "runs" / f"retrieval_{_slug(step['name'])}.jsonl",
            orient="records",
            lines=True,
        )
        row = {"config": step["name"], "n": len(df)}
        for col in ["recall@5", "recall@10", "mrr", "ndcg@10", "hit@1"]:
            row[col] = df[col].mean()
        for qtype, g in df.groupby("qtype"):
            row[f"recall@5[{qtype}]"] = g["recall@5"].mean()
        row["p50_ms"] = df["latency_ms"].median()
        row["p95_ms"] = df["latency_ms"].quantile(0.95)
        if prev is not None:
            lo, hi = paired_bootstrap_ci(prev["recall@5"].to_numpy(), df["recall@5"].to_numpy())
            row["Δrecall@5 95% CI"] = f"[{lo:+.3f}, {hi:+.3f}]"
        prev = df
        summary.append(row)
    result = pd.DataFrame(summary)
    result.to_csv(out_dir / "retrieval_ablation.csv", index=False)
    return result


def run_generation(
    settings: Settings,
    gold: list[GoldQuestion],
    limit: int | None = None,
    workers: int | None = None,
) -> pd.DataFrame:
    llm = LLM(settings.llm, settings.resolve(settings.llm.cache_path))
    retriever = build_retriever(settings, llm)
    items = gold[:limit] if limit else gold
    # Retrieval runs serially (models are not thread-safe); generation and judging in parallel.
    retrieved = [
        (q, retriever.retrieve(q.question)) for q in track(items, description="Retrieving")
    ]

    def run(pair):
        q, passages = pair
        ans = generate_answer(q.question, passages, llm)
        row = {
            "qid": q.qid,
            "qtype": q.qtype,
            "answer": ans.answer,
            "answerable": ans.answerable,
            "citations": ans.citations,
            "gen_latency_ms": ans.latency_ms,
        }
        row.update(judge(q, ans, llm))
        return row

    with ThreadPoolExecutor(max_workers=workers or settings.llm.max_concurrency) as pool:
        rows = list(
            track(
                pool.map(run, retrieved), total=len(retrieved), description="Generating + judging"
            )
        )
    df = pd.DataFrame(rows)
    out_dir = settings.resolve(settings.eval.results_dir)
    (out_dir / "runs").mkdir(parents=True, exist_ok=True)
    df.to_json(out_dir / "runs" / "generation.jsonl", orient="records", lines=True)
    usage = llm.total
    meta = {
        "answer_model": settings.llm.answer_model,
        "judge_model": settings.llm.judge_model,
        "config": settings.chunking.variant,
        "n": len(df),
        "llm_calls": usage.calls,
        "cache_hits": usage.cache_hits,
        "cost_usd": round(usage.cost_usd, 4),
    }
    (out_dir / "generation_meta.json").write_text(json.dumps(meta, indent=2))
    return df


def _slug(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_").lower()


def results_dir() -> Path:
    s = load_settings()
    return s.resolve(s.eval.results_dir)
