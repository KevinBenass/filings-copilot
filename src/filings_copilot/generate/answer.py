"""Grounded answer generation with inline citations and an explicit abstain path."""

from __future__ import annotations

import re
import time

from filings_copilot.llm import LLM
from filings_copilot.retrieve.pipeline import Retriever
from filings_copilot.schemas import Answer, RetrievedChunk

SYSTEM = """You are a financial analyst answering questions strictly from excerpts of SEC
filings. Rules:
- Use only the numbered passages. Never use outside knowledge, even if you are confident.
- Cite every factual sentence with passage numbers in square brackets, e.g. [2] or [1][4].
- Quote figures exactly as written, with units and the period they refer to. If you compute
  something (growth rate, ratio), show the inputs and cite them.
- If the passages do not contain the information needed, set "answerable" to false and say
  briefly what is missing. A partial answer is fine if you state what is missing.
Reply in the required JSON format."""

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answerable": {
            "type": "boolean",
            "description": "false if the passages do not contain the information needed",
        },
        "answer": {"type": "string", "description": "The answer, citing passages like [2]."},
        "citations": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["answerable", "answer", "citations"],
    "additionalProperties": False,
}


def format_passages(passages: list[RetrievedChunk]) -> str:
    blocks = []
    for n, p in enumerate(passages, 1):
        c = p.chunk
        blocks.append(
            f'<passage id="{n}" company="{c.company}" ticker="{c.ticker}" '
            f'fiscal_year="{c.fiscal_year}" section="{c.section_title}">\n{c.text}\n</passage>'
        )
    return "\n\n".join(blocks)


def generate_answer(question: str, passages: list[RetrievedChunk], llm: LLM) -> Answer:
    t0 = time.perf_counter()
    prompt = f"{format_passages(passages)}\n\nQuestion: {question}"
    data: dict = {}
    cost = 0.0
    # Structured output constrained by a JSON schema: no free-text JSON to parse. Retry once with
    # more room if a long answer is cut off.
    for max_tokens in (2048, 4096):
        try:
            data, usage = llm.complete_structured(
                prompt,
                ANSWER_SCHEMA,
                model=llm.cfg.answer_model,
                system=SYSTEM,
                max_tokens=max_tokens,
            )
            cost += usage.cost_usd
            break
        except ValueError:
            continue
    answer = str(data.get("answer", "")).strip() or "(no answer generated)"
    answerable = bool(data.get("answerable", True))
    listed = [str(c).strip("[] ") for c in data.get("citations", [])]
    # Citations are what the model listed plus any [n] markers in the text, so a forgotten
    # list entry doesn't count against an answer whose sentences are cited.
    cites = [int(c) for c in listed if c.isdigit()] + [
        int(n) for n in re.findall(r"\[(\d+)\]", answer)
    ]
    cites = sorted({c for c in cites if 1 <= c <= len(passages)})
    return Answer(
        question=question,
        answer=answer,
        answerable=answerable,
        citations=cites,
        passages=passages,
        latency_ms=(time.perf_counter() - t0) * 1e3,
        cost_usd=cost,
    )


class Copilot:
    """Retrieval + generation, the object the API and UI talk to."""

    def __init__(self, retriever: Retriever, llm: LLM) -> None:
        self.retriever = retriever
        self.llm = llm

    def ask(self, question: str) -> Answer:
        t0 = time.perf_counter()
        before = self.llm.total.cost_usd
        passages = self.retriever.retrieve(question)
        ans = generate_answer(question, passages, self.llm)
        ans.latency_ms = (time.perf_counter() - t0) * 1e3
        ans.cost_usd = self.llm.total.cost_usd - before
        return ans
