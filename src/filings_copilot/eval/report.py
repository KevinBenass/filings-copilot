"""Render results into results/REPORT.md, a chart, and the README results block."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from filings_copilot.config import ROOT

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def _fmt(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype.kind == "f":
            fmt = "{:.0f}" if c.endswith("_ms") else "{:.3f}"
            out[c] = out[c].map(lambda v, fmt=fmt: "" if pd.isna(v) else fmt.format(v))
    return out.fillna("")


def ablation_chart(df: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = range(len(df))
    ax.plot(x, df["recall@5"], marker="o", color="#2563eb", label="Recall@5")
    ax.plot(x, df["ndcg@10"], marker="s", color="#059669", label="nDCG@10")
    ax.plot(x, df["mrr"], marker="^", color="#d97706", label="MRR")
    ax.set_xticks(list(x), df["config"], rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Retrieval ablation: each step adds one component")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def generation_table(df: pd.DataFrame) -> pd.DataFrame:
    answerable = df[df["qtype"] != "unanswerable"]
    rows = []
    for qtype, g in answerable.groupby("qtype"):
        rows.append(
            {
                "question type": qtype,
                "n": len(g),
                "correctness": g["correctness"].mean(),
                "faithfulness": g["faithfulness"].mean(),
                "citation precision": g["citation_precision"].mean(),
                "false abstention": g["false_abstain"].mean(),
            }
        )
    rows.append(
        {
            "question type": "**all answerable**",
            "n": len(answerable),
            "correctness": answerable["correctness"].mean(),
            "faithfulness": answerable["faithfulness"].mean(),
            "citation precision": answerable["citation_precision"].mean(),
            "false abstention": answerable["false_abstain"].mean(),
        }
    )
    un = df[df["qtype"] == "unanswerable"]
    if len(un):
        rows.append(
            {
                "question type": "unanswerable (correct abstention)",
                "n": len(un),
                "correctness": un["abstain_correct"].mean(),
            }
        )
    return pd.DataFrame(rows)


def build_report(results_dir: Path, readme: Path | None = None) -> str:
    parts: list[str] = []
    abl_path = results_dir / "retrieval_ablation.csv"
    if abl_path.exists():
        abl = pd.read_csv(abl_path)
        ablation_chart(abl, results_dir / "ablation.png")
        main_cols = [
            c
            for c in [
                "config",
                "recall@5",
                "recall@10",
                "mrr",
                "ndcg@10",
                "hit@1",
                "Δrecall@5 95% CI",
                "p50_ms",
            ]
            if c in abl
        ]
        type_cols = ["config"] + [c for c in abl.columns if c.startswith("recall@5[")]
        parts += [
            "### Retrieval ablation",
            f"Evaluated on {int(abl['n'].iloc[0])} answerable questions. "
            "Δ CI is a paired bootstrap 95% interval for the change in Recall@5 versus the row above.",
            "",
            "![Retrieval ablation](results/ablation.png)",
            "",
            _fmt(abl[main_cols]).to_markdown(index=False, disable_numparse=True),
            "",
            "Recall@5 by question type:",
            "",
            _fmt(abl[type_cols]).to_markdown(index=False, disable_numparse=True),
        ]
    gen_path = results_dir / "runs" / "generation.jsonl"
    meta_path = results_dir / "generation_meta.json"
    if gen_path.exists():
        gen = pd.read_json(gen_path, lines=True)
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        parts += [
            "",
            "### End-to-end answer quality",
            f"Answer model `{meta.get('answer_model', '?')}`, judge `{meta.get('judge_model', '?')}`, "
            f"{meta.get('n', len(gen))} questions, total eval cost ${meta.get('cost_usd', 0):.2f}.",
            "",
            _fmt(generation_table(gen)).to_markdown(index=False, disable_numparse=True),
        ]
    if not parts:
        parts = ["_No results yet. Run `make eval` to populate this section._"]
    body = "\n".join(parts)
    (results_dir / "REPORT.md").write_text(
        "# Results\n\n" + body.replace("results/ablation.png", "ablation.png") + "\n"
    )

    readme = readme or ROOT / "README.md"
    if readme.exists():
        text = readme.read_text()
        if START in text and END in text:
            new = re.sub(
                re.escape(START) + r".*?" + re.escape(END),
                lambda _: f"{START}\n{body}\n{END}",
                text,
                flags=re.DOTALL,
            )
            readme.write_text(new)
    return body
