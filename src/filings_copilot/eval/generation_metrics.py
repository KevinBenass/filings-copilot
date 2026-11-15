"""LLM-as-judge for answer quality, with the judge's job made as narrow as possible.

One judge call per question returns:
  correctness       0 / 0.5 / 1 against the reference answer (incorrect / partial / correct)
  claims            the answer split into atomic factual claims, each marked supported or
                    not by the retrieved passages  -> faithfulness = supported / total
  citation_support  for each cited passage, whether it supports a claim it is cited for
                    -> citation precision
Abstention is scored without the judge: unanswerable questions should yield
answerable=false, and answerable ones should not.

Known limitation (stated in the README): the judge is itself an LLM. We use a different model tier than the generator, and spot-check a sample by hand.
"""

from __future__ import annotations

from statistics import mean

from filings_copilot.generate.answer import format_passages
from filings_copilot.llm import LLM
from filings_copilot.schemas import Answer, GoldQuestion

JUDGE_PROMPT = """You are grading an answer produced by a retrieval-augmented system over SEC filings.

{passages}

Question: {question}
Reference answer: {reference}
System answer: {answer}
Passages cited by the system: {cited}

Tasks:
1. correctness: compare the system answer with the reference. 1 = same facts (numbers may
   be rounded or phrased differently), 0.5 = partially correct or incomplete, 0 = wrong or
   missing the key fact.
2. claims: split the system answer into atomic factual claims. For each, decide whether the
   passages above support it. Keep each claim short (at most 12 words); cover every claim.
3. citation_support: for each cited passage id, does that passage support at least one claim?

Reply in the required JSON format."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "correctness": {"type": "number", "enum": [0, 0.5, 1]},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"claim": {"type": "string"}, "supported": {"type": "boolean"}},
                "required": ["claim", "supported"],
                "additionalProperties": False,
            },
        },
        "citation_support": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"passage": {"type": "integer"}, "supports": {"type": "boolean"}},
                "required": ["passage", "supports"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["correctness", "claims", "citation_support"],
    "additionalProperties": False,
}


def judge(gold: GoldQuestion, answer: Answer, llm: LLM) -> dict[str, float | None]:
    if gold.qtype == "unanswerable":
        return {"abstain_correct": 0.0 if answer.answerable else 1.0}

    row: dict[str, float | None] = {"false_abstain": 0.0 if answer.answerable else 1.0}
    if not answer.answerable:
        row.update(correctness=0.0, faithfulness=None, citation_precision=None)
        return row
    prompt = JUDGE_PROMPT.format(
        passages=format_passages(answer.passages),
        question=gold.question,
        reference=gold.answer,
        answer=answer.answer,
        cited=answer.citations or "none",
    )
    out = None
    # Long answers have many claims: give the judge room, and retry once with more if its
    # JSON is still cut off. Failures are recorded, never silently dropped from averages.
    for max_tokens in (3000, 8000):
        try:
            out, _ = llm.complete_structured(
                prompt, JUDGE_SCHEMA, model=llm.cfg.judge_model, max_tokens=max_tokens
            )
            break
        except ValueError:
            continue
    if out is None:
        return {
            **row,
            "judge_failed": 1.0,
            "correctness": None,
            "faithfulness": None,
            "citation_precision": None,
        }
    row["judge_failed"] = 0.0
    claims = out.get("claims") or []
    raw_support = out.get("citation_support") or []
    if isinstance(raw_support, dict):  # tolerate {"<id>": bool} as well as the schema's list
        support = {str(k): v for k, v in raw_support.items()}
    else:
        support = {
            str(c.get("passage")): c.get("supports") for c in raw_support if isinstance(c, dict)
        }
    row["correctness"] = float(out.get("correctness", 0))
    row["faithfulness"] = mean(bool(c.get("supported")) for c in claims) if claims else None
    row["citation_precision"] = (
        mean(bool(v) for v in support.values()) if support else (0.0 if answer.answer else None)
    )
    return row
