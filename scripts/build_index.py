"""Build the retrieval index from `data/sample_reports.csv`.

Usage:
    python -m scripts.build_index --backend colpali
    python -m scripts.build_index --backend clip
    python -m scripts.build_index --backend both
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.config import INDEX_DIR, SAMPLE_REPORTS_CSV
from src.models.clip_model import CLIPWrapper
from src.models.colpali import ColPaliWrapper
from src.retrieval.vector_store import ColPaliStore, FlatCLIPStore, _MetaRow
from src.utils.image_utils import load_image


def _rows_from_csv(csv_path: Path):
    df = pd.read_csv(csv_path)
    needed = {"image_id", "image_path", "text"}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns in {csv_path}: {missing}")
    return df


def build_colpali(df: pd.DataFrame, out_dir: Path, batch_size: int = 4) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    enc = ColPaliWrapper().load()
    store = ColPaliStore()
    rows = df.to_dict(orient="records")
    for i in tqdm(range(0, len(rows), batch_size), desc="colpali encode"):
        batch = rows[i : i + batch_size]
        imgs = [load_image(r["image_path"]) for r in batch]
        embs = enc.encode_images(imgs)
        store.add(
            list(embs),
            [_MetaRow(image_id=str(r["image_id"]), image_path=r["image_path"], report=r["text"]) for r in batch],
        )
    store.save(out_dir)
    print(f"colpali index -> {out_dir}  (docs={len(store.meta)})")


def build_clip(df: pd.DataFrame, out_dir: Path, batch_size: int = 16) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    enc = CLIPWrapper().load()
    rows = df.to_dict(orient="records")
    store = FlatCLIPStore(dim=None)
    for i in tqdm(range(0, len(rows), batch_size), desc="clip encode"):
        batch = rows[i : i + batch_size]
        imgs = [load_image(r["image_path"]) for r in batch]
        embs = enc.encode_images(imgs)
        meta = [_MetaRow(image_id=str(r["image_id"]), image_path=r["image_path"], report=r["text"]) for r in batch]
        store.add(embs, meta)
    store.save(out_dir)
    print(f"clip index -> {out_dir}  (docs={len(store.meta)})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default=str(SAMPLE_REPORTS_CSV))
    ap.add_argument("--out_root", default=str(INDEX_DIR))
    ap.add_argument("--backend", choices=["colpali", "clip", "both"], default="colpali")
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    df = _rows_from_csv(Path(args.reports))
    out_root = Path(args.out_root)

    if args.backend in {"colpali", "both"}:
        build_colpali(df, out_root / "colpali", batch_size=args.batch_size)
    if args.backend in {"clip", "both"}:
        build_clip(df, out_root / "clip", batch_size=max(args.batch_size, 16))


if __name__ == "__main__":
    main()
