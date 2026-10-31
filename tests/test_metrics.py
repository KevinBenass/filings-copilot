import math

import numpy as np
import pytest

from filings_copilot.eval.retrieval_metrics import covers, gains, ndcg_at_k, score_question
from filings_copilot.eval.runner import paired_bootstrap_ci
from filings_copilot.retrieve.fusion import rrf
from filings_copilot.schemas import Chunk, Evidence


def chunk(doc="D", start=0, end=100, cid="c"):
    return Chunk(
        chunk_id=cid,
        doc_id=doc,
        ticker="T",
        company="Co",
        fiscal_year=2025,
        item="7",
        section_title="Item 7",
        start=start,
        end=end,
        text="x",
    )


def test_covers_requires_same_doc_and_overlap():
    ev = Evidence(doc_id="D", start=40, end=80)
    assert covers(chunk(start=0, end=100), ev)
    assert not covers(chunk(doc="E", start=0, end=100), ev)
    assert not covers(chunk(start=0, end=45), ev)  # only 5/40 of evidence, 5/45 of chunk
    assert covers(chunk(start=50, end=70), ev)  # small chunk entirely inside evidence


def test_gains_do_not_reward_duplicates():
    evs = [Evidence(doc_id="D", start=0, end=50), Evidence(doc_id="E", start=0, end=50)]
    ranked = [chunk(start=0, end=60), chunk(start=10, end=60), chunk(doc="E", start=0, end=60)]
    assert gains(ranked, evs) == [1, 0, 1]


def test_score_question_values():
    evs = [Evidence(doc_id="D", start=0, end=50)]
    ranked = [chunk(doc="X"), chunk(doc="X"), chunk(start=0, end=60)]
    row = score_question(ranked, evs, [1, 3])
    assert row["mrr"] == pytest.approx(1 / 3)
    assert row["hit@1"] == 0 and row["hit@3"] == 1
    assert row["recall@3"] == 1
    assert row["ndcg@3"] == pytest.approx(1 / math.log2(4))


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k([1, 1, 0], n_evidence=2, k=3) == pytest.approx(1.0)


def test_rrf_prefers_documents_ranked_well_by_both():
    fused = [d for d, _ in rrf([[1, 2, 3], [3, 1, 4]])]
    assert fused[0] == 1 and set(fused) == {1, 2, 3, 4}


def test_paired_bootstrap_ci_brackets_true_difference():
    rng = np.random.default_rng(0)
    a = rng.random(400)
    lo, hi = paired_bootstrap_ci(a, a + 0.1)
    assert lo == pytest.approx(0.1) and hi == pytest.approx(0.1)
    lo, hi = paired_bootstrap_ci(a, rng.random(400))
    assert lo < 0 < hi
