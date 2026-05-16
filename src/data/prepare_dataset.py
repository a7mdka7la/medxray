"""MIMIC-CXR loader + cleaning.

The Kaggle drop of MIMIC-CXR (simhadrisadaram/mimic-cxr-dataset) ships a CSV
with columns roughly like:

  image, study_id, subject_id, text, ...

`image` is the relative path to the JPG and `text` is the free-text report.
This module produces a clean, lightweight dataframe with image paths +
normalized reports + the three parsed sections (findings/impression/recs).

Usage:
    python -m src.data.prepare_dataset \\
        --raw_csv path/to/mimic.csv \\
        --images_root path/to/files \\
        --out data/sample_reports.csv \\
        --limit 500
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..utils.text_utils import normalize_report, parse_structured_report


REPORT_COL_CANDIDATES = ("text", "report", "findings_text")
IMAGE_COL_CANDIDATES = ("image", "image_path", "path", "filename")
ID_COL_CANDIDATES = ("image_id", "study_id", "id")


def _first_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of {candidates} present in CSV columns: {list(df.columns)}")


def prepare(raw_csv: str, images_root: str, limit: int | None) -> pd.DataFrame:
    df = pd.read_csv(raw_csv)

    img_col = _first_col(df, IMAGE_COL_CANDIDATES)
    report_col = _first_col(df, REPORT_COL_CANDIDATES)
    id_col = _first_col(df, ID_COL_CANDIDATES) if any(c in df.columns for c in ID_COL_CANDIDATES) else None

    if id_col is None:
        df = df.reset_index(drop=True)
        df["image_id"] = df.index.astype(str)
        id_col = "image_id"

    out = pd.DataFrame()
    out["image_id"] = df[id_col].astype(str)
    out["image_path"] = df[img_col].astype(str).apply(lambda p: str(Path(images_root) / p))
    out["text"] = df[report_col].astype(str).map(normalize_report)

    # Skip empty reports — they cannot be evaluated and confuse the retriever.
    out = out[out["text"].str.len() > 20].reset_index(drop=True)

    parsed = out["text"].map(parse_structured_report)
    out["findings"] = parsed.map(lambda d: d.get("findings", ""))
    out["impression"] = parsed.map(lambda d: d.get("impression", ""))
    out["recommendations"] = parsed.map(lambda d: d.get("recommendations", ""))

    if limit is not None:
        out = out.head(limit)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_csv", required=True)
    ap.add_argument("--images_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    df = prepare(args.raw_csv, args.images_root, args.limit)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows -> {args.out}")


if __name__ == "__main__":
    main()
