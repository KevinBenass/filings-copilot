"""Build the evaluation set: synthetic questions grounded in verbatim evidence spans.

Four question types, because aggregate scores hide where RAG systems fail:
  single_hop    one fact from a prose passage
  table         one figure that must be read from a financial table
  multi_hop     a comparison across two passages (two companies, or two years)
  unanswerable  plausible questions the filings do not answer (tests abstention)

Each generated question must come with a verbatim quote, which is located in the filing to
get a character span. Questions whose quote cannot be found are discarded, so every gold
label is mechanically checkable. A review CLI (`fcopilot review`) then lets a human accept,
edit or reject each item; the README reports how many were human-verified.
"""

from __future__ import annotations

import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rich.progress import Progress

from filings_copilot.config import Settings
from filings_copilot.ingest.chunking import n_tokens, split_blocks
from filings_copilot.ingest.parse import load_parsed
from filings_copilot.llm import LLM
from filings_copilot.schemas import Evidence, GoldQuestion, ParsedFiling

# Sections analysts actually query; weights shape the question mix.
SECTION_WEIGHTS = {"1": 2, "1A": 2, "1C": 0.5, "7": 4, "7A": 1, "8": 3, "5": 0.5}

SINGLE_PROMPT = """You are writing evaluation questions for a search system over SEC filings.

Filing: {company} ({ticker}), Form 10-K, fiscal year {fy}, {section}
<passage>
{passage}
</passage>

Write one question a financial analyst would realistically ask that is answered by this
passage{table_hint}. The question must be self-contained: name the company and fiscal year,
and do not copy long phrases from the passage (paraphrase as an analyst would). Then give a
concise answer and the exact supporting text copied verbatim from the passage (one sentence
or one table row, at most 40 words).

If the passage is boilerplate, a cross-reference or has no substantive fact, return
{{"skip": true}}.
Otherwise respond with JSON only:
{{"question": "...", "answer": "...", "quote": "..."}}"""

MULTI_PROMPT = """You are writing evaluation questions for a search system over SEC filings.

Passage A, {a_company} ({a_ticker}) fiscal year {a_fy}, {a_section}:
<passage_a>
{a_text}
</passage_a>

Passage B, {b_company} ({b_ticker}) fiscal year {b_fy}, {b_section}:
<passage_b>
{b_text}
</passage_b>

Write one comparative question an analyst would ask that needs a fact from BOTH passages
(e.g. compare a figure, a risk, a strategy). Name both companies/years explicitly. Give a
concise answer and one verbatim supporting quote from each passage (at most 40 words each).
If no meaningful comparison exists, return {{"skip": true}}.
Otherwise respond with JSON only:
{{"question": "...", "answer": "...", "quote_a": "...", "quote_b": "..."}}"""

UNANSWERABLE_PROMPT = """Write {n} realistic analyst questions about {company} ({ticker}) that its
Form 10-K annual reports for fiscal years {years} would NOT answer, but that sound like they
might: e.g. internal targets never disclosed, results for fiscal year {future_year}, exact
figures for a segment the company does not report separately, or details of unannounced
products. Each must name the company. Respond with JSON only:
{{"questions": ["...", "..."]}}"""


# ------------------------------------------------------------------ span location
def _normalize_with_map(text: str) -> tuple[str, list[int]]:
    out_chars: list[str] = []
    idx_map: list[int] = []
    prev_space = False
    for i, ch in enumerate(text):
        if ch.isspace() or ch == "|":
            if not prev_space:
                out_chars.append(" ")
                idx_map.append(i)
            prev_space = True
            continue
        out_chars.append(ch.lower())
        idx_map.append(i)
        prev_space = False
    return "".join(out_chars), idx_map


def locate(text: str, quote: str, lo: int = 0, hi: int | None = None) -> tuple[int, int] | None:
    """Find ``quote`` in ``text[lo:hi]`` ignoring case, whitespace and table pipes."""
    hi = len(text) if hi is None else hi
    region = text[lo:hi]
    norm_region, idx_map = _normalize_with_map(region)
    norm_quote, _ = _normalize_with_map(quote.strip().strip('"'))
    norm_quote = norm_quote.strip()
    if len(norm_quote) < 12:
        return None
    pos = norm_region.find(norm_quote)
    if pos == -1:
        return None
    start = lo + idx_map[pos]
    end = lo + idx_map[min(pos + len(norm_quote) - 1, len(idx_map) - 1)] + 1
    return start, end


# ------------------------------------------------------------------ sampling
def _sample_blocks(docs: list[ParsedFiling], rng: random.Random, want_table: bool, n: int):
    pool = []
    for doc in docs:
        for sec in doc.sections:
            w = SECTION_WEIGHTS.get(sec.item, 0)
            if not w:
                continue
            for b in split_blocks(doc.text[sec.start : sec.end], sec.start):
                ok = (
                    b.is_table if want_table else (not b.is_table and 60 <= n_tokens(b.text) <= 450)
                )
                if want_table and ok:
                    ok = 3 <= b.text.count("\n") <= 40 and any(ch.isdigit() for ch in b.text)
                if ok:
                    pool.append((w, doc, sec, b))
    if not pool:
        return []
    weights = [p[0] for p in pool]
    return [pool[i][1:] for i in _weighted_unique(rng, weights, min(n, len(pool)))]


def _weighted_unique(rng: random.Random, weights: list[float], n: int) -> list[int]:
    keys = [(rng.random() ** (1.0 / w), i) for i, w in enumerate(weights)]
    return [i for _, i in sorted(keys, reverse=True)[:n]]


# ------------------------------------------------------------------ generation
def _single(llm: LLM, doc: ParsedFiling, sec, block, qtype: str, qid: str) -> GoldQuestion | None:
    m = doc.meta
    try:
        out = llm.complete_json(
            SINGLE_PROMPT.format(
                company=m.company,
                ticker=m.ticker,
                fy=m.fiscal_year,
                section=sec.title,
                passage=block.text,
                table_hint=" and requires reading a specific figure from the table"
                if qtype == "table"
                else "",
            ),
            model=llm.cfg.answer_model,
            max_tokens=400,
            temperature=0.7,
        )
    except ValueError:
        return None
    if out.get("skip") or not out.get("question"):
        return None
    span = locate(doc.text, out.get("quote", ""), block.start, block.end)
    if span is None:
        return None
    return GoldQuestion(
        qid=qid,
        question=out["question"],
        answer=out["answer"],
        qtype=qtype,
        evidence=[Evidence(doc_id=m.doc_id, start=span[0], end=span[1], quote=out["quote"])],
        tickers=[m.ticker],
        fiscal_years=[m.fiscal_year],
    )


def _pairs(docs: list[ParsedFiling], rng: random.Random, n: int):
    """Pairs of blocks from the same item: across companies (same year) or across years."""
    by_key: dict[tuple[str, str], list[ParsedFiling]] = {}
    for d in docs:
        by_key.setdefault(("year", str(d.meta.fiscal_year)), []).append(d)
        by_key.setdefault(("ticker", d.meta.ticker), []).append(d)
    groups = [g for g in by_key.values() if len(g) >= 2]
    pairs = []
    for _ in range(n * 4):
        if len(pairs) >= n or not groups:
            break
        a_doc, b_doc = rng.sample(rng.choice(groups), 2)
        item = rng.choice(["1", "1A", "7", "7A"])
        sa = next((s for s in a_doc.sections if s.item == item), None)
        sb = next((s for s in b_doc.sections if s.item == item), None)
        if not sa or not sb:
            continue
        ba = [
            b
            for b in split_blocks(a_doc.text[sa.start : sa.end], sa.start)
            if not b.is_table and 60 <= n_tokens(b.text) <= 400
        ]
        bb = [
            b
            for b in split_blocks(b_doc.text[sb.start : sb.end], sb.start)
            if not b.is_table and 60 <= n_tokens(b.text) <= 400
        ]
        if ba and bb:
            pairs.append((a_doc, sa, rng.choice(ba), b_doc, sb, rng.choice(bb)))
    return pairs


def _multi(llm: LLM, pair, qid: str) -> GoldQuestion | None:
    a_doc, sa, ba, b_doc, sb, bb = pair
    am, bm = a_doc.meta, b_doc.meta
    try:
        out = llm.complete_json(
            MULTI_PROMPT.format(
                a_company=am.company,
                a_ticker=am.ticker,
                a_fy=am.fiscal_year,
                a_section=sa.title,
                a_text=ba.text,
                b_company=bm.company,
                b_ticker=bm.ticker,
                b_fy=bm.fiscal_year,
                b_section=sb.title,
                b_text=bb.text,
            ),
            model=llm.cfg.answer_model,
            max_tokens=500,
            temperature=0.7,
        )
    except ValueError:
        return None
    if out.get("skip") or not out.get("question"):
        return None
    span_a = locate(a_doc.text, out.get("quote_a", ""), ba.start, ba.end)
    span_b = locate(b_doc.text, out.get("quote_b", ""), bb.start, bb.end)
    if not span_a or not span_b:
        return None
    return GoldQuestion(
        qid=qid,
        question=out["question"],
        answer=out["answer"],
        qtype="multi_hop",
        evidence=[
            Evidence(doc_id=am.doc_id, start=span_a[0], end=span_a[1], quote=out["quote_a"]),
            Evidence(doc_id=bm.doc_id, start=span_b[0], end=span_b[1], quote=out["quote_b"]),
        ],
        tickers=sorted({am.ticker, bm.ticker}),
        fiscal_years=sorted({am.fiscal_year, bm.fiscal_year}),
    )


def build_gold_set(settings: Settings, llm: LLM, workers: int | None = None) -> list[GoldQuestion]:
    cfg = settings.eval
    rng = random.Random(cfg.seed)
    docs = load_parsed(settings.resolve(settings.corpus.parsed_dir))
    # Over-sample: some generations are skipped or fail span verification.
    singles = _sample_blocks(docs, rng, want_table=False, n=int(cfg.n_single_hop * 1.4))
    tables = _sample_blocks(docs, rng, want_table=True, n=int(cfg.n_table * 1.6))
    pairs = _pairs(docs, rng, int(cfg.n_multi_hop * 1.6))

    jobs = (
        [("single_hop", (d, s, b)) for d, s, b in singles]
        + [("table", (d, s, b)) for d, s, b in tables]
        + [("multi_hop", p) for p in pairs]
    )

    def run(i_job):
        i, (qtype, payload) = i_job
        qid = f"{qtype[:2]}-{i:04d}"
        return (
            _multi(llm, payload, qid)
            if qtype == "multi_hop"
            else _single(llm, *payload, qtype, qid)
        )

    results: list[GoldQuestion] = []
    with (
        Progress() as progress,
        ThreadPoolExecutor(max_workers=workers or llm.cfg.max_concurrency) as pool,
    ):
        task = progress.add_task("Generating questions", total=len(jobs))
        for q in pool.map(run, enumerate(jobs)):
            progress.advance(task)
            if q:
                results.append(q)

    caps = {"single_hop": cfg.n_single_hop, "table": cfg.n_table, "multi_hop": cfg.n_multi_hop}
    gold: list[GoldQuestion] = []
    for qtype, cap in caps.items():
        gold += [q for q in results if q.qtype == qtype][:cap]

    tickers = sorted({d.meta.ticker for d in docs})
    years = sorted({d.meta.fiscal_year for d in docs})
    companies = {d.meta.ticker: d.meta.company for d in docs}
    per = max(1, -(-cfg.n_unanswerable // len(tickers)))
    for t in tickers:
        try:
            out = llm.complete_json(
                UNANSWERABLE_PROMPT.format(
                    n=per,
                    company=companies[t],
                    ticker=t,
                    years=", ".join(map(str, years)),
                    future_year=max(years) + 2,
                ),
                model=llm.cfg.answer_model,
                max_tokens=500,
                temperature=0.7,
            )
        except ValueError:
            continue
        for q in out.get("questions", [])[:per]:
            gold.append(
                GoldQuestion(
                    qid=f"un-{len(gold):04d}",
                    question=q,
                    answer="(not in filings)",
                    qtype="unanswerable",
                    tickers=[t],
                )
            )
    gold = [q for q in gold if q.qtype != "unanswerable"] + [
        q for q in gold if q.qtype == "unanswerable"
    ][: cfg.n_unanswerable]
    return gold


def save_gold(gold: list[GoldQuestion], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(q.model_dump_json() + "\n" for q in gold))


def load_gold(path: Path) -> list[GoldQuestion]:
    return [
        GoldQuestion.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def leakage_ratio(question: str, quote: str) -> float:
    """Share of the quote's content words that appear verbatim in the question.

    High values mean the question parrots the passage, which inflates lexical retrieval
    scores. Reported in the eval README so readers can judge the gold set.
    """
    words = lambda s: {w for w in re.findall(r"[a-z]{4,}", s.lower())}  # noqa: E731
    q, e = words(question), words(quote)
    return len(q & e) / len(e) if e else 0.0
