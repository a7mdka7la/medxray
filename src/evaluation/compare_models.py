"""Cross-model evaluation runner.

Runs every report-generation strategy on a held-out split and every QA
backend on the synthetic QA dataset, then writes a markdown table summary.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd
from tqdm import tqdm

from ..pipeline.rag_qa import RAGQAPipeline
from ..pipeline.report_generation import ReportGenerator, Strategy
from . import metrics as M


REPORT_STRATEGIES: list[Strategy] = ["medgemma_direct", "colpali_rag", "clip_rag"]
QA_BACKENDS = ["colpali", "clip"]


@dataclass
class EvalRow:
    setting: str
    bleu4: float
    rouge_l: float
    bertscore_f1: float
    clinical_f1: float
    avg_latency_s: float
    n: int


# ----------------------------------------------------------------------
# Report generation
# ----------------------------------------------------------------------


def evaluate_report_generation(
    test_df: pd.DataFrame,
    generator: ReportGenerator,
    strategies: Iterable[Strategy] = REPORT_STRATEGIES,
) -> List[EvalRow]:
    rows: list[EvalRow] = []
    for strategy in strategies:
        hyps, refs, lats = [], [], []
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"report:{strategy}"):
            try:
                result = generator.generate(row["image_path"], strategy=strategy)
            except Exception as e:
                tqdm.write(f"[skip] {row.get('image_id')}: {e}")
                continue
            hyps.append(result.raw_text)
            refs.append(row["text"])
            lats.append(result.latency_s)

        if not hyps:
            continue
        rows.append(
            EvalRow(
                setting=f"report::{strategy}",
                bleu4=M.bleu_score(hyps, refs),
                rouge_l=M.rouge_l_score(hyps, refs),
                bertscore_f1=M.bertscore_f1(hyps, refs),
                clinical_f1=M.clinical_token_accuracy(hyps, refs),
                avg_latency_s=float(sum(lats) / len(lats)),
                n=len(hyps),
            )
        )
    return rows


# ----------------------------------------------------------------------
# QA
# ----------------------------------------------------------------------


def evaluate_qa(qa_records: list[dict], backends: Iterable[str] = QA_BACKENDS) -> List[EvalRow]:
    rows: list[EvalRow] = []
    for backend in backends:
        pipe = RAGQAPipeline(backend=backend)
        hyps, refs, lats = [], [], []
        for rec in tqdm(qa_records, desc=f"qa:{backend}"):
            try:
                res = pipe.answer(rec["image_path"], rec["question"], k=3)
            except Exception as e:
                tqdm.write(f"[skip] {rec.get('image_id')}: {e}")
                continue
            hyps.append(res.answer)
            refs.append(rec["answer"])
            lats.append(res.latency_s)
        if not hyps:
            continue
        rows.append(
            EvalRow(
                setting=f"qa::{backend}",
                bleu4=M.bleu_score(hyps, refs),
                rouge_l=M.rouge_l_score(hyps, refs),
                bertscore_f1=M.bertscore_f1(hyps, refs),
                clinical_f1=M.clinical_token_accuracy(hyps, refs),
                avg_latency_s=float(sum(lats) / len(lats)),
                n=len(hyps),
            )
        )
    return rows


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------


def to_markdown(rows: list[EvalRow]) -> str:
    if not rows:
        return "_No results._\n"
    head = "| Setting | n | BLEU-4 | ROUGE-L | BERTScore F1 | Clinical F1 | Avg latency (s) |\n"
    head += "|---|---:|---:|---:|---:|---:|---:|\n"
    lines = []
    for r in rows:
        lines.append(
            f"| `{r.setting}` | {r.n} | {r.bleu4:.3f} | {r.rouge_l:.3f} | "
            f"{r.bertscore_f1:.3f} | {r.clinical_f1:.3f} | {r.avg_latency_s:.2f} |"
        )
    return head + "\n".join(lines) + "\n"


def save_results(rows: list[EvalRow], json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps([r.__dict__ for r in rows], indent=2))
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(to_markdown(rows))
