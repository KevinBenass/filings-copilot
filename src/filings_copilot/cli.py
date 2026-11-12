"""Command line entry point: `fcopilot --help`."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from filings_copilot.config import Settings, load_ablations, load_settings

app = typer.Typer(add_completion=False, help="Filings Copilot: RAG over SEC 10-K filings.")
console = Console()

ConfigOpt = Annotated[
    str | None, typer.Option("--config", "-c", help="Named config from configs/ablations.yaml")
]
SetOpt = Annotated[
    list[str] | None, typer.Option("--set", help="Override, e.g. --set retrieval.top_k=5")
]
BEST_CONFIG = "+ filters"


def _coerce(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return {"true": True, "false": False}.get(v.lower(), v)


def settings_for(config: str | None = None, sets: list[str] | None = None) -> Settings:
    overrides: dict = {}
    if config:
        steps = {s["name"]: s["overrides"] for s in load_ablations()}
        if config not in steps:
            raise typer.BadParameter(f"Unknown config {config!r}. Options: {list(steps)}")
        overrides.update(steps[config])
    for kv in sets or []:
        k, _, v = kv.partition("=")
        overrides[k] = _coerce(v)
    return load_settings(overrides=overrides)


@app.command()
def download() -> None:
    """Download the configured 10-K filings from SEC EDGAR."""
    from filings_copilot.ingest.edgar import download_corpus

    s = load_settings()
    paths, missing = download_corpus(
        s.corpus.tickers,
        s.corpus.form,
        s.corpus.fiscal_years,
        s.sec_user_agent,
        s.resolve(s.corpus.raw_dir),
    )
    console.print(f"[green]{len(paths)} filings in {s.corpus.raw_dir}")
    for ticker, years in missing.items():
        console.print(f"[yellow]Not found: {ticker} fiscal years {', '.join(map(str, years))}")


@app.command()
def parse() -> None:
    """Parse raw HTML filings into text, tables and sections; write the corpus table."""
    from filings_copilot.eval.report import write_corpus_table
    from filings_copilot.ingest.parse import ITEM_TITLES, load_parsed, parse_corpus

    s = load_settings()
    parse_corpus(s.resolve(s.corpus.raw_dir), s.resolve(s.corpus.parsed_dir))
    docs = load_parsed(s.resolve(s.corpus.parsed_dir))
    key = {"7": "MD&A", "7A": "Mkt risk", "8": "Financials"}
    table = Table("doc", "chars", "sections", *key.values())
    for d in docs:
        items = {sec.item for sec in d.sections}
        table.add_row(
            d.meta.doc_id,
            f"{len(d.text):,}",
            " ".join(sec.item for sec in d.sections),
            *("✓" if k in items else "[red]✗" for k in key),
        )
    console.print(table)
    unknown = sorted({sec.item for d in docs for sec in d.sections} - set(ITEM_TITLES) - {"0"})
    if unknown:
        console.print(f"[yellow]Unrecognized items: {unknown}")
    path = write_corpus_table(docs, s.resolve(s.eval.results_dir))
    console.print(f"[green]Corpus table -> {path} and README")


@app.command()
def inspect(
    doc_id: str,
    pattern: Annotated[
        str, typer.Argument(help="Regex, e.g. 'item\\s*7' or 'discussion and analysis'")
    ] = r"item\s*7",
) -> None:
    """Show how a filing's text looks around a pattern (for debugging section detection)."""
    from filings_copilot.ingest.parse import ParsedFiling, section_report

    s = load_settings()
    path = s.resolve(s.corpus.parsed_dir) / f"{doc_id}.json"
    doc = ParsedFiling.model_validate_json(path.read_text())
    console.print(
        f"{doc_id}: {len(doc.text):,} chars, sections {[sec.item for sec in doc.sections]}"
    )
    for line in section_report(doc.text, pattern):
        console.print(line, markup=False, highlight=False)


@app.command()
def chunk(
    config: ConfigOpt = None,
    set_: SetOpt = None,
    dry_run: Annotated[bool, typer.Option(help="Only print the LLM cost estimate")] = False,
) -> None:
    """Chunk the parsed corpus (and contextualize with the LLM if configured)."""
    from filings_copilot.ingest.store import build_chunks, load_chunks

    s = settings_for(config, set_)
    path, est = build_chunks(s, dry_run=dry_run)
    if s.chunking.context == "llm":
        console.print(f"LLM contextualization estimate: ${est:.2f}")
    if not dry_run:
        console.print(f"[green]{len(load_chunks(path)):,} chunks -> {path}")


@app.command()
def index(config: ConfigOpt = None, set_: SetOpt = None, force: bool = False) -> None:
    """Embed a chunk variant into a dense index."""
    from filings_copilot.retrieve.factory import build_dense_index

    idx = build_dense_index(settings_for(config, set_), force=force)
    console.print(f"[green]Dense index ready: {idx.size:,} vectors")


@app.command("gen-gold")
def gen_gold() -> None:
    """Generate the synthetic evaluation set (requires ANTHROPIC_API_KEY)."""
    from filings_copilot.eval.dataset import build_gold_set, leakage_ratio, save_gold
    from filings_copilot.llm import LLM

    s = load_settings()
    llm = LLM(s.llm, s.resolve(s.llm.cache_path))
    gold = build_gold_set(s, llm)
    save_gold(gold, s.resolve(s.eval.gold_path))
    counts = {
        t: sum(q.qtype == t for q in gold)
        for t in ("single_hop", "table", "multi_hop", "unanswerable")
    }
    leak = [leakage_ratio(q.question, q.evidence[0].quote) for q in gold if q.evidence]
    console.print(
        f"[green]{len(gold)} questions {counts}, mean lexical leakage {sum(leak) / max(1, len(leak)):.2f}, cost ${llm.total.cost_usd:.2f}"
    )


@app.command()
def review(start: int = 0) -> None:
    """Human review loop for the gold set: accept, edit or drop each question."""
    from filings_copilot.eval.dataset import load_gold, save_gold

    s = load_settings()
    path = s.resolve(s.eval.gold_path)
    gold = load_gold(path)
    kept = gold[:start]
    for i, q in enumerate(gold[start:], start):
        console.rule(f"{i + 1}/{len(gold)}  {q.qid}  [{q.qtype}]")
        console.print(f"[bold]Q:[/] {q.question}\n[bold]A:[/] {q.answer}")
        for ev in q.evidence:
            console.print(f"[dim]{ev.doc_id}:[/] {ev.quote}")
        choice = typer.prompt("[a]ccept / [e]dit / [d]rop / [q]uit", default="a")
        if choice == "q":
            kept += gold[i:]
            break
        if choice == "d":
            continue
        if choice == "e":
            q.question = typer.prompt("Question", default=q.question)
            q.answer = typer.prompt("Answer", default=q.answer)
        q.verified = True
        kept.append(q)
        save_gold(kept + gold[i + 1 :], path)  # save progress after every item
    save_gold(kept, path)
    console.print(f"[green]{sum(q.verified for q in kept)}/{len(kept)} verified")


@app.command("eval-retrieval")
def eval_retrieval(
    allow_llm_cost: Annotated[
        bool, typer.Option(help="Permit LLM chunk contextualization")
    ] = False,
    only: Annotated[list[str] | None, typer.Option(help="Run only these ablation steps")] = None,
) -> None:
    """Run the retrieval ablation ladder over the gold set (no LLM except optional steps)."""
    from filings_copilot.eval.dataset import load_gold
    from filings_copilot.eval.runner import run_ablation

    s = load_settings()
    df = run_ablation(load_gold(s.resolve(s.eval.gold_path)), allow_llm_cost, only)
    console.print(df.to_string(index=False))


@app.command("eval-generation")
def eval_generation(config: ConfigOpt = BEST_CONFIG, limit: int | None = None) -> None:
    """End-to-end eval: retrieve, answer with Claude, grade with an LLM judge."""
    from filings_copilot.eval.dataset import load_gold
    from filings_copilot.eval.report import generation_table
    from filings_copilot.eval.runner import run_generation

    s = settings_for(config)
    df = run_generation(s, load_gold(s.resolve(s.eval.gold_path)), limit)
    console.print(generation_table(df).to_string(index=False))


@app.command()
def report() -> None:
    """Write results/REPORT.md, the ablation chart and the README results block."""
    from filings_copilot.eval.report import build_report
    from filings_copilot.eval.runner import results_dir

    console.print(Markdown(build_report(results_dir())))


@app.command()
def ask(
    question: str,
    config: ConfigOpt = BEST_CONFIG,
    generate: Annotated[bool, typer.Option(help="Generate an answer (needs API key)")] = True,
) -> None:
    """Ask a question. Use --no-generate to inspect retrieval only."""
    from filings_copilot.generate.answer import Copilot
    from filings_copilot.llm import LLM
    from filings_copilot.retrieve.factory import build_retriever

    s = settings_for(config)
    llm = LLM(s.llm, s.resolve(s.llm.cache_path))
    retriever = build_retriever(s, llm)
    if not generate:
        for n, h in enumerate(retriever.retrieve(question), 1):
            c = h.chunk
            console.print(
                f"[bold][{n}] {c.ticker} FY{c.fiscal_year} · {c.section_title}[/] ranks={h.ranks}"
            )
            console.print(c.text[:500] + ("…" if len(c.text) > 500 else ""), "\n")
        return
    ans = Copilot(retriever, llm).ask(question)
    console.print(Markdown(ans.answer))
    for n in ans.citations:
        c = ans.passages[n - 1].chunk
        console.print(f"[dim][{n}] {c.company} FY{c.fiscal_year}, {c.section_title}")
    console.print(f"[dim]{ans.latency_ms:.0f} ms · ${ans.cost_usd:.4f}")


if __name__ == "__main__":
    app()
