"""Run cross-model evaluation and write the markdown table into docs/.

Usage:
    python -m scripts.run_evaluation --split data/sample_reports.csv --qa data/qa_dataset/qa.jsonl
    python -m scripts.run_evaluation --strategies clip_rag,medgemma_direct
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import get_args

import pandas as pd

from src.config import OUTPUTS_DIR, QA_DATASET_DIR, ROOT, SAMPLE_REPORTS_CSV
from src.evaluation.compare_models import (
    evaluate_qa,
    evaluate_report_generation,
    save_results,
    to_markdown,
)
from src.pipeline.report_generation import ReportGenerator, Strategy


def _load_qa(path: Path) -> list[dict]:
    records: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=str(SAMPLE_REPORTS_CSV))
    ap.add_argument("--qa", default=str(QA_DATASET_DIR / "qa.jsonl"))
    ap.add_argument("--limit_reports", type=int, default=5)
    ap.add_argument("--limit_qa", type=int, default=10)
    ap.add_argument(
        "--strategies",
        default="clip_rag,medgemma_direct",
        help="Comma-separated subset of report-generation strategies to run",
    )
    ap.add_argument("--qa_backends", default="clip", help="Comma-separated QA retriever backends")
    ap.add_argument("--skip_qa", action="store_true")
    ap.add_argument("--skip_report", action="store_true")
    ap.add_argument("--skip_bertscore", action="store_true", help="Skip BERTScore (needs ~400MB DistilBERT download)")
    args = ap.parse_args()

    if args.skip_bertscore:
        from src.evaluation import metrics as _m

        _m.bertscore_f1 = lambda *_, **__: float("nan")  # type: ignore[assignment]

    test_df = pd.read_csv(args.split).head(args.limit_reports)
    qa = _load_qa(Path(args.qa))[: args.limit_qa]

    valid = set(get_args(Strategy))
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip() in valid]
    qa_backends = [b.strip() for b in args.qa_backends.split(",") if b.strip()]

    report_rows = (
        [] if args.skip_report else evaluate_report_generation(test_df, ReportGenerator(), strategies)
    )
    qa_rows = [] if args.skip_qa else evaluate_qa(qa, backends=qa_backends)

    rows = report_rows + qa_rows
    save_results(rows, OUTPUTS_DIR / "eval.json", OUTPUTS_DIR / "eval.md")

    # Append into the written report doc so the comparison table stays in sync.
    report_md = ROOT / "docs" / "REPORT.md"
    if report_md.exists():
        body = report_md.read_text()
        marker = "<!-- AUTO-METRICS -->"
        end_marker = "<!-- /AUTO-METRICS -->"
        new_table = "\n" + to_markdown(rows) + "\n"
        if marker in body and end_marker in body:
            head, _, rest = body.partition(marker)
            _, _, tail = rest.partition(end_marker)
            new_body = head + marker + new_table + end_marker + tail
        else:
            new_body = body + "\n\n" + marker + new_table + end_marker + "\n"
        report_md.write_text(new_body)

    print("done.")
    print(to_markdown(rows))


if __name__ == "__main__":
    main()
